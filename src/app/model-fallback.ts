/**
 * Provider fallback for a ticket's Executor Session: switches the session's model route when the current one fails on quota,
 * credentials or an exhausted rate limit, records each switch for the ticket, keeps HANDOFF.md, and lets a person reset to the
 * primary route. The chain lives in the plugin configuration, so changing it needs no build or reinstall.
 */
import type { Context } from '@deepseek-ai/cordis'
import { ReasoningEffortId } from '@deepseek-ai/dsh-llm'
import type { LlmFailure } from '@deepseek-ai/dsh-llm'
import type { Agent, ModelSelection, RequestErrorAction } from '@deepseek-ai/dsh-agent'
import { existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import type { FallbackModel, ModelFallback as ModelFallbackConfig } from '../deployment.ts'

/** One model switch of a ticket, as the ticket report shows it. */
export interface FallbackSwitch {
  /** Route the request failed on, as `provider/model`. */
  from: string
  /** Route the session continues on, as `provider/model`. */
  to: string
  /** Failure code and at most a short provider message. */
  reason: string
  /** When the switch happened. */
  at: string
}
/** The durable per-ticket fallback accounting, restored by the ticket's next Assignment. */
export interface FallbackRecord { active?: ModelSelection; switches: FallbackSwitch[]; exhausted?: boolean }
/** What `status` returns for a Session that has switched at least once or stopped on the chain's end. */
export interface FallbackStatus {
  /** The route the Session uses now, as `provider/model`. */
  active: string
  /** The first route of the configured chain. */
  primary: string | null
  /** Every switch of the ticket, oldest first. */
  switches: FallbackSwitch[]
  /** Switches the ticket may still make before it stops. */
  remaining: number
  /** Whether the chain or the switch limit ended the fallback and the Session returned to the primary route. */
  exhausted: boolean
}
/** The host hooks `ModelFallback` reads: configuration, durability, delivery and the token meter. */
export interface ModelFallbackHost {
  readonly context: Context
  readonly lifetime: AbortSignal
  readonly config: ModelFallbackConfig
  /** Persists one ticket's accounting; the ticket's next Assignment restores it. */
  save(ticketId: string, record: FallbackRecord): void
  /** Reads the ticket's persisted accounting, if any. */
  load(ticketId: string): FallbackRecord | undefined
  /** Delivers a notice into the agent's next pre-step; `requestId` deduplicates it across retries. */
  notify(agent: Agent, requestId: string, text: string, summary: string): Promise<void>
  /** Reads the Session's live context pressure from the token meter, when the deployment mounts one. */
  pressure(session: Agent['session']): { projectedTokens?: number; contextWindow?: number } | undefined
  /** The ticket's title and goal, for the handoff file. */
  describe(ticketId: string): { title: string; goal: string } | undefined
}
/** Failure codes that switch the model at once: a retry on the same route cannot succeed. */
const SWITCH_CODES = new Set(['QUOTA', 'ACCOUNT_QUOTA', 'INVALID_CREDENTIAL'])
/** A rate limit switches only once DSH's own retry policy gave up on the route. */
const RATE_LIMIT = 'RATE_LIMIT'
/** A route as `provider/model`, the form switches and the handoff file use. */
const routeKey = (route: Pick<ModelSelection, 'provider' | 'model'>): string => `${route.provider}/${route.model}`
const sameRoute = (left: Pick<ModelSelection, 'provider' | 'model'>, right: Pick<ModelSelection, 'provider' | 'model'>): boolean =>
  left.provider === right.provider && left.model === right.model
/** A failure reason short enough for a ticket card: the code, and the provider message's first line cut to 120 characters. */
const reasonOf = (failure: LlmFailure): string => {
  const line = failure.message.split('\n')[0]?.trim().slice(0, 120) ?? ''
  return line && line !== failure.code ? `${failure.code}: ${line}` : failure.code
}
/** Resolve the chain's plain configuration values into a request selection DSH accepts. */
const selectionOf = (route: FallbackModel): ModelSelection => ({ provider: route.provider, model: route.model,
  ...(route.reasoningEffort === undefined ? {} : { reasoningEffort: ReasoningEffortId(route.reasoningEffort) }) })

/** One Executor Session the controller watches, with its recorded switches. */
interface Attached {
  agent: Agent
  ticketId: string
  directory: string
  /** The fallback route in force, or `undefined` while the Session runs its own role model. */
  active?: ModelSelection
  switches: FallbackSwitch[]
  exhausted: boolean
  /** Whether the context-threshold notice already went out for the current window. */
  warned: boolean
  /** Invalidates a switch still resolving a route after a person's reset or a newer switch. */
  generation: number
  disposers: (() => void)[]
}
/** The part of DSH's `agent/request-error` payload the fallback reads. */
interface RequestFailure { failure: LlmFailure; signal: AbortSignal }

/**
 * Watches a ticket's Executor Session for provider failures and context pressure, switches its model along the configured
 * chain, writes and asks the worker to maintain HANDOFF.md, and exposes each switch to the ticket card and the worker's report.
 */
export class ModelFallback {
  private readonly attached = new Map<Agent, Attached>()
  private readonly bySession = new Map<string, Attached>()
  constructor(private readonly host: ModelFallbackHost) {}
  /** Whether the configured chain can fall back at all. */
  private enabled(): boolean { return this.host.config.models.length > 1 }
  /** Attach to a ticket's Executor Session: install the model override and failure handler, restoring an earlier Assignment's accounting. */
  attach(agent: Agent, ticketId: string, directory: string): void {
    if (!this.enabled() || this.attached.has(agent)) return
    const record = this.host.load(ticketId)
    const active = record?.active && this.host.config.models.some(route => sameRoute(route, record.active!)) ? record.active : undefined
    const state: Attached = { agent, ticketId, directory, active, switches: record?.switches ?? [], exhausted: record?.exhausted ?? false,
      warned: false, generation: 0, disposers: [] }
    this.attached.set(agent, state)
    this.bySession.set(agent.id, state)
    state.disposers.push(
      agent.ctx.on('agent/request-error', (payload, next) => this.failed(state, payload, next)),
      // A later step must use the fallback route; this wins over DSH's own role selection only while a fallback is in force.
      agent.ctx.on('agent/request', async (_payload, next) => {
        const selected = state.active
        if (!selected) return next()
        const resolved = await next()
        const { reasoningEffort: _inherited, ...rest } = resolved
        return { ...rest, provider: selected.provider, model: selected.model, ...(selected.reasoningEffort === undefined ? {} : { reasoningEffort: selected.reasoningEffort }) }
      }, { prepend: true }),
      agent.ctx.on('system-prompt/assemble', async (_assembly, _context, next) => {
        const selected = state.active
        const assembled = await next()
        return selected === undefined ? assembled : { ...assembled, variables: { ...assembled.variables, provider: selected.provider, model: selected.model } }
      }, { prepend: true }),
    )
  }
  /** Release one Session's listeners; its ticket accounting stays durable for a later Assignment. */
  detach(agent: Agent): void {
    const state = this.attached.get(agent)
    if (!state) return
    this.attached.delete(agent)
    if (this.bySession.get(agent.id) === state) this.bySession.delete(agent.id)
    for (const dispose of state.disposers.splice(0)) { try { dispose() } catch (error) { this.host.context.logger.warn('dsh-superagent could not release a model fallback listener: %s', String(error)) } }
  }
  /**
   * Follow one Session event: a finished compaction asks the worker to read HANDOFF.md (the summary may be the only record of
   * the work), and a usage sample below the threshold asks it to write one while the current model can still see the work.
   */
  observe(agent: Agent, event: { type: string; seq: number; data: unknown }): void {
    const state = this.attached.get(agent)
    if (!state) return
    if (event.type === 'compaction/end') {
      if ((event.data as { error?: string }).error !== undefined) return
      state.warned = false
      this.notice(state, `fallback-read:${state.ticketId}:${event.seq}`, readNotice(this.fileOf(state)), 'read HANDOFF.md after a compaction')
        .catch(() => { /* The notice is best effort; the ticket continues without it. */ })
      return
    }
    if (event.type !== 'assistant/message' || state.warned) return
    const pressure = this.host.pressure(agent.session)
    const window = pressure?.contextWindow, used = pressure?.projectedTokens
    if (window === undefined || used === undefined || window <= 0) return
    const remaining = (window - used) / window
    if (remaining >= this.host.config.contextThreshold) return
    state.warned = true
    this.notice(state, `fallback-write:${state.ticketId}:${state.switches.length}:${event.seq}`,
      writeNotice(this.fileOf(state), Math.max(0, Math.round(remaining * 100))), 'write HANDOFF.md before the context runs out')
      .catch(() => { /* The notice is best effort; the ticket continues without it. */ })
  }
  /** The Session's fallback status for a ticket card, or `{}` before its first switch. */
  status(sessionId: string | undefined): { model_fallback?: FallbackStatus } {
    const state = sessionId === undefined ? undefined : this.bySession.get(sessionId)
    if (!state) return {}
    const switches = state.switches
    if (!switches.length && !state.exhausted) return {}
    const primary = this.host.config.models[0]
    return { model_fallback: { active: routeKey(state.active ?? this.current(state)), primary: primary ? routeKey(primary) : null,
      switches: [...switches], remaining: Math.max(0, this.host.config.maxSwitches - switches.length), exhausted: state.exhausted } }
  }
  /**
   * Return every watched ticket's Session to the first route of the chain on a person's explicit request, clearing its switch
   * count; automatic fallback still applies to later provider failures. A switch still resolving a route is abandoned.
   */
  reset(): { reset: number; model: string | null } {
    const primary = this.host.config.models[0]
    if (!primary) return { reset: 0, model: null }
    let reset = 0
    for (const state of this.attached.values()) {
      state.generation++
      state.active = selectionOf(primary)
      state.switches = []
      state.exhausted = false
      state.warned = false
      state.agent.session.append('model/selection', state.active)
      this.save(state)
      this.notice(state, `fallback-reset:${state.ticketId}:${Date.now()}`, resetNotice(routeKey(primary)), 'model reset to the primary route')
        .catch(() => { /* The notice is best effort. */ })
      reset++
    }
    return { reset, model: routeKey(primary) }
  }
  /** The route a Session currently uses: the fallback route in force, else its last request header, else the primary. */
  private current(state: Attached): ModelSelection {
    if (state.active) return state.active
    const header = state.agent.session.requestHeader()?.config
    if (header) return { provider: header.provider, model: header.model,
      ...(header.reasoningEffort === undefined ? {} : { reasoningEffort: header.reasoningEffort }) }
    return selectionOf(this.host.config.models[0]!)
  }
  /** Handle one failed request: let DSH's retry policy decide first, then switch on a terminal code no other recovery owned. */
  private async failed(state: Attached, payload: RequestFailure, next: () => Promise<RequestErrorAction>): Promise<RequestErrorAction> {
    const downstream = await next()
    if (downstream !== undefined) return downstream
    if (payload.signal.aborted) return undefined
    const code = payload.failure.code
    if (!SWITCH_CODES.has(code) && code !== RATE_LIMIT) return undefined
    const generation = ++state.generation
    try {
      return await this.switch(state, code, payload.failure, generation)
    } catch (error) {
      this.host.context.logger.warn('dsh-superagent could not switch the model of ticket %s: %s', state.ticketId, String(error))
      return undefined
    }
  }
  /** Resolve the next usable route and continue the failed step on it, or stop on the limit or the chain's end. */
  private async switch(state: Attached, code: string, failure: LlmFailure, generation: number): Promise<RequestErrorAction> {
    const chain = this.host.config.models
    const current = this.current(state)
    const from = routeKey(current)
    const reason = reasonOf(failure)
    if (state.switches.length >= this.host.config.maxSwitches)
      return this.stop(state, code, `the ${this.host.config.maxSwitches}-switch limit for this ticket is reached`, from, reason)
    const found = chain.findIndex(route => sameRoute(route, current))
    let selected: FallbackModel | undefined
    const unavailable: string[] = []
    for (let index = found < 0 ? 0 : found + 1; index < chain.length; index++) {
      const candidate = chain[index]!
      if (sameRoute(candidate, current)) continue
      try {
        const resolved = await this.host.context.llm.resolveCallConfig({ provider: candidate.provider, model: candidate.model,
          ...(candidate.reasoningEffort === undefined ? {} : { reasoningEffort: ReasoningEffortId(candidate.reasoningEffort) }) })
        selected = { provider: resolved.provider, model: resolved.model,
          ...(resolved.reasoningEffort === undefined ? {} : { reasoningEffort: resolved.reasoningEffort }) }
        break
      } catch (error) {
        unavailable.push(`${routeKey(candidate)} (${error instanceof Error ? error.message : String(error)})`)
      }
    }
    // A reset, a newer switch, or the Session ending while the route resolved: the newer decision stands.
    if (state.generation !== generation) return undefined
    if (!selected) return this.stop(state, code, unavailable.length
      ? `no fallback route resolves: ${unavailable.join('; ')}` : 'the fallback chain has no further route', from, reason)
    state.active = selectionOf(selected)
    state.exhausted = false
    state.switches.push({ from, to: routeKey(selected), reason, at: new Date().toISOString() })
    state.agent.session.append('model/selection', state.active)
    this.save(state)
    this.writeHandoff(state, from, routeKey(selected), reason)
    const count = state.switches.length
    await this.notice(state, `fallback-switch:${state.ticketId}:${count}`,
      switchNotice(from, routeKey(selected), reason, count, this.host.config.maxSwitches, this.fileOf(state)),
      `model switched ${from} → ${routeKey(selected)}`)
    return { kind: 'retry' }
  }
  /** End the fallback for this failure: return the Session to the primary route and let the error reach the ticket. */
  private stop(state: Attached, code: string, why: string, from: string, reason: string): RequestErrorAction {
    state.exhausted = true
    const primary = this.host.config.models[0]
    if (primary) {
      state.active = selectionOf(primary)
      state.agent.session.append('model/selection', state.active)
      this.writeHandoff(state, from, routeKey(primary), reason)
    }
    this.save(state)
    this.host.context.logger.warn('dsh-superagent model fallback for ticket %s stopped at %s: %s (%s)', state.ticketId, from, why, code)
    return undefined
  }
  /** The handoff file path in the ticket directory. */
  private fileOf(state: Attached): string { return join(state.directory, this.host.config.handoffFile) }
  /** Write a factual continuation note before the old route becomes unavailable; a worker-written file is never overwritten. */
  private writeHandoff(state: Attached, from: string, to: string, reason: string): void {
    const file = this.fileOf(state)
    try {
      if (existsSync(file)) return
      mkdirSync(state.directory, { recursive: true })
      const ticket = this.host.describe(state.ticketId)
      writeFileSync(file, handoffText(ticket?.title ?? state.ticketId, ticket?.goal ?? '', state.ticketId, from, to, reason,
        state.switches.length, this.host.config.maxSwitches), { encoding: 'utf8', mode: 0o600 })
    } catch (error) {
      this.host.context.logger.warn('dsh-superagent could not write the handoff file of ticket %s: %s', state.ticketId, String(error))
    }
  }
  /** Persist the accounting; a failed write must not break the failed step's recovery. */
  private save(state: Attached): void {
    try { this.host.save(state.ticketId, { switches: state.switches, exhausted: state.exhausted, ...(state.active ? { active: state.active } : {}) }) }
    catch (error) { this.host.context.logger.warn('dsh-superagent could not record the model fallback of ticket %s: %s', state.ticketId, String(error)) }
  }
  /** Deliver a notice once, ignoring a delivery failure: the switch itself already happened. */
  private async notice(state: Attached, requestId: string, text: string, summary: string): Promise<void> {
    try { await this.host.notify(state.agent, requestId, text, summary) }
    catch (error) { this.host.context.logger.warn('dsh-superagent could not deliver a model fallback notice of ticket %s: %s', state.ticketId, String(error)) }
  }
}

/** The notice asking the worker for a handoff while its current model can still see the work. */
function writeNotice(file: string, percentLeft: number): string {
  return `[model fallback] The context window of the current model is nearly full: about ${percentLeft}% is left. Before continuing, `
    + `write or update ${file} in your working directory (do not commit it) with: the ticket goal; the files, commits and pull request `
    + 'already produced; the checks that passed and those that failed; what remains; known problems; and the exact stopping point. '
    + 'Keep it under 150 lines, because the next model reads it.'
}
/** The notice asking the worker to read a handoff after a compaction or a switch. */
function readNotice(file: string): string {
  return `[model fallback] The context was compacted. Read ${file} in your working directory before continuing, then update it with `
    + 'anything it no longer records.'
}
/** The notice a switch leaves for the worker, which also makes the switch visible in its next report. */
function switchNotice(from: string, to: string, reason: string, count: number, limit: number, file: string): string {
  return `[model fallback] The model for this ticket switched from ${from} to ${to} (${reason}). This is switch ${count} of ${limit}. `
    + `Read ${file} in your working directory before continuing and update it as you work. Include this switch, with the from and to `
    + 'routes and the reason, in your next orchestrator_report so the ticket report shows which part of the work each model did.'
}
/** The notice a person's reset leaves for the worker. */
function resetNotice(primary: string): string {
  return `[model fallback] A person reset the model of this ticket to ${primary}. This notice is informational; continue the assignment with this route.`
}
/** A factual, bounded continuation note the controller can write without a model. */
function handoffText(title: string, goal: string, ticketId: string, from: string, to: string, reason: string, count: number, limit: number): string {
  return `# HANDOFF

Continuation note for ticket ${ticketId}: ${title}. This file is not committed.

## Goal

${goal || 'Continue the ticket\'s assignment in this working directory.'}

## Done

The fallback controller cannot summarize work, so read the session transcript for what was done before this point.

## Checks

Read the session transcript for the checks that passed and those that failed.

## Remaining

Continue the assignment. Update this file with files, commits, pull requests, checks, known problems and the stopping point.

## Known problems

The model route ${from} failed with ${reason}.

## Stopping point

Model switched from ${from} to ${to} at ${new Date().toISOString()} (switch ${count} of ${limit}).`
}
