/** Model fallback: switching on a terminal provider failure, its limits, the handoff file, context pressure and a person's reset. */
import { afterEach, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import { agentEvents, type Agent, type ModelSelection } from '@deepseek-ai/dsh-agent'
import { Session, SessionId } from '@deepseek-ai/dsh-session'
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ModelFallback, type FallbackRecord, type ModelFallbackHost } from '../src/app/model-fallback.ts'
import type { ModelFallback as ModelFallbackConfig } from '../src/deployment.ts'

const contexts: Context[] = []
const directories: string[] = []
afterEach(async () => {
  await Promise.all(contexts.splice(0).map(ctx => ctx.fiber.dispose()))
  for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true })
  vi.restoreAllMocks()
})

const chain: ModelFallbackConfig = { models: [
  { provider: 'primary', model: 'one' }, { provider: 'second', model: 'two' }, { provider: 'third', model: 'three' }],
  maxSwitches: 2, contextThreshold: 0.15, handoffFile: 'HANDOFF.md' }
type RequestError = Parameters<ReturnType<typeof agentEvents>['waterfall']>[1]
type Next = () => Promise<{ kind: 'retry' } | undefined>

interface Fixture { ctx: Context; agent: Agent; session: Session; fallback: ModelFallback; resolve: ReturnType<typeof vi.fn>; notify: ReturnType<typeof vi.fn>
  records: Map<string, FallbackRecord>; directory: string
  fail: (code: string, message?: string, next?: Next) => Promise<{ kind: 'retry' } | undefined>
  request: () => Promise<ModelSelection | undefined> }
/**
 * A managed worker Session and a controller over it: a real Context and Session, a fake LLM route resolver, an in-memory
 * persistent record, a notice spy, and a handoff directory.
 */
async function fixture(options: { config?: Partial<ModelFallbackConfig>; pressure?: () => { projectedTokens?: number; contextWindow?: number } | undefined
  records?: Map<string, FallbackRecord> } = {}): Promise<Fixture> {
  const ctx = new Context(); contexts.push(ctx); await ctx.plugin(SystemPrompt)
  const resolve = vi.fn(async (value: ModelSelection) => value)
  ctx.reflect.provide('llm', { resolveCallConfig: resolve })
  const session = Session.create(SessionId('worker-ticket'))
  const agent = { ctx, session, id: 'worker-ticket' } as unknown as Agent
  const records = options.records ?? new Map<string, FallbackRecord>()
  const notify = vi.fn(async () => {})
  const host: ModelFallbackHost = {
    context: ctx, lifetime: new AbortController().signal,
    config: { ...chain, ...options.config },
    save: (id, record) => { records.set(id, record) },
    load: id => records.get(id),
    notify,
    pressure: () => options.pressure?.(),
    describe: () => ({ title: 'Add fallback', goal: 'Continue the fallback work' }),
  }
  const directory = mkdtempSync(join(tmpdir(), 'fallback-')); directories.push(directory)
  const fallback = new ModelFallback(host)
  fallback.attach(agent, 'ticket-1', directory)
  const fail = async (code: string, message = 'provider failed', next: Next = async () => undefined) =>
    await agentEvents(ctx, agent).waterfall('agent/request-error', { turn: 1, step: 1, provider: 'primary',
      failure: { code, message }, retryPolicy: undefined, signal: new AbortController().signal } as RequestError, next) as { kind: 'retry' } | undefined
  const request = () => agentEvents(ctx, agent).waterfall('agent/request', { turn: 1, step: 1, signal: new AbortController().signal },
    async () => ({ provider: 'host', model: 'default' }))
  return { ctx, agent, session, fallback, resolve, notify, records, directory, fail, request }
}
/** The route of the Session's newest durable model selection. */
const selected = (session: Session): ModelSelection | undefined => {
  const events = session.snapshotEvents().filter(event => (event as { type: string }).type === 'model/selection')
  return (events.at(-1) as unknown as { data: ModelSelection } | undefined)?.data
}

it('switches to the next route on a quota failure, writes the handoff, and records the switch for the card and the report', async () => {
  const { fallback, session, notify, records, directory, fail, request } = await fixture()
  expect(await request()).toEqual({ provider: 'host', model: 'default' })
  expect(await fail('QUOTA', 'insufficient balance')).toEqual({ kind: 'retry' })
  expect(selected(session)).toEqual({ provider: 'second', model: 'two' })
  expect(records.get('ticket-1')).toMatchObject({ active: { provider: 'second', model: 'two' },
    switches: [{ from: 'primary/one', to: 'second/two', reason: expect.stringContaining('QUOTA') }] })
  expect(readFileSync(join(directory, 'HANDOFF.md'), 'utf8')).toContain('second/two')
  expect(notify).toHaveBeenCalledWith(expect.anything(), 'fallback-switch:ticket-1:1',
    expect.stringContaining('from primary/one to second/two'), expect.stringContaining('second/two'))
  expect(notify.mock.calls[0]![2]).toContain('orchestrator_report')
  expect(await request()).toEqual({ provider: 'second', model: 'two' })
  expect(fallback.status('worker-ticket')).toMatchObject({ model_fallback: { active: 'second/two', primary: 'primary/one',
    remaining: 1, exhausted: false } })
})

it('switches on invalid credentials and on a rate limit only when DSH\'s retry policy gave up', async () => {
  const { fail, session } = await fixture()
  expect(await fail('INVALID_CREDENTIAL', 'bad key')).toEqual({ kind: 'retry' })
  expect(selected(session)).toEqual({ provider: 'second', model: 'two' })
  expect(await fail('RATE_LIMIT', 'too many requests', async () => ({ kind: 'retry' }))).toEqual({ kind: 'retry' })
  expect(selected(session)).toEqual({ provider: 'second', model: 'two' })
  expect(await fail('RATE_LIMIT', 'still limited')).toEqual({ kind: 'retry' })
  expect(selected(session)).toEqual({ provider: 'third', model: 'three' })
})

it('leaves every other failure to the ordinary retry and error path', async () => {
  const { fail, session, notify } = await fixture()
  expect(await fail('INVALID_REQUEST', 'bad request')).toBeUndefined()
  expect(await fail('SERVER', 'boom')).toBeUndefined()
  expect(selected(session)).toBeUndefined()
  expect(notify).not.toHaveBeenCalled()
})

it('stops at the switch limit and returns the Session to the primary route', async () => {
  const { fallback, session, records, fail } = await fixture({ config: { maxSwitches: 1 } })
  expect(await fail('QUOTA', 'out of quota')).toEqual({ kind: 'retry' })
  expect(await fail('QUOTA', 'out of quota')).toBeUndefined()
  expect(selected(session)).toEqual({ provider: 'primary', model: 'one' })
  expect(records.get('ticket-1')).toMatchObject({ exhausted: true, active: { provider: 'primary', model: 'one' },
    switches: [{ from: 'primary/one', to: 'second/two' }] })
  expect(fallback.status('worker-ticket')).toMatchObject({ model_fallback: { exhausted: true, remaining: 0,
    active: 'primary/one' } })
})

it('stops on the chain\'s end and on a route the provider no longer offers, returning to the primary route', async () => {
  const { fallback, session, fail } = await fixture({ config: { models: [
    { provider: 'primary', model: 'one' }, { provider: 'second', model: 'two' }] } })
  expect(await fail('QUOTA')).toEqual({ kind: 'retry' })
  expect(await fail('QUOTA')).toBeUndefined()
  expect(selected(session)).toEqual({ provider: 'primary', model: 'one' })
  expect(fallback.status('worker-ticket')).toMatchObject({ model_fallback: { exhausted: true } })
})

it('skips a route that does not resolve and uses the next available one', async () => {
  const { fail, resolve, session } = await fixture()
  resolve.mockImplementation(async (value: ModelSelection) => {
    if (value.model === 'two') throw new Error('unknown model')
    return value
  })
  expect(await fail('QUOTA')).toEqual({ kind: 'retry' })
  expect(selected(session)).toEqual({ provider: 'third', model: 'three' })
})

it('returns every ticket to the primary route on a person\'s reset and clears its switch count', async () => {
  const { fallback, session, notify, records, fail } = await fixture()
  await fail('QUOTA')
  expect(fallback.reset()).toEqual({ reset: 1, model: 'primary/one' })
  expect(selected(session)).toEqual({ provider: 'primary', model: 'one' })
  expect(records.get('ticket-1')).toMatchObject({ switches: [], exhausted: false })
  expect(notify).toHaveBeenCalledWith(expect.anything(), expect.stringContaining('fallback-reset:ticket-1'),
    expect.stringContaining('reset'), expect.any(String))
  expect(await fail('QUOTA')).toEqual({ kind: 'retry' })
  expect(records.get('ticket-1')?.switches).toHaveLength(1)
})

it('asks the worker to write the handoff below the context threshold, once per window, and to read it after a compaction', async () => {
  let pressure: { projectedTokens?: number; contextWindow?: number } | undefined = { projectedTokens: 900, contextWindow: 1000 }
  const { fallback, agent, notify } = await fixture({ pressure: () => pressure })
  fallback.observe(agent, { type: 'assistant/message', seq: 10, data: {} })
  await vi.waitFor(() => { expect(notify).toHaveBeenCalledTimes(1) })
  expect(notify.mock.calls[0]![2]).toContain('10%')
  expect(notify.mock.calls[0]![2]).toContain('HANDOFF.md')
  fallback.observe(agent, { type: 'assistant/message', seq: 11, data: {} })
  expect(notify).toHaveBeenCalledTimes(1)
  pressure = { projectedTokens: 950, contextWindow: 1000 }
  fallback.observe(agent, { type: 'compaction/end', seq: 12, data: {} })
  await vi.waitFor(() => { expect(notify).toHaveBeenCalledTimes(2) })
  expect(notify.mock.calls[1]![2]).toContain('compacted')
  // A failed compaction changes nothing, and a healthy window never asks for a handoff.
  fallback.observe(agent, { type: 'compaction/end', seq: 13, data: { error: 'summary failed' } })
  pressure = { projectedTokens: 100, contextWindow: 1000 }
  fallback.observe(agent, { type: 'assistant/message', seq: 14, data: {} })
  expect(notify).toHaveBeenCalledTimes(2)
})

it('never overwrites a handoff the worker wrote itself', async () => {
  const { directory, fail } = await fixture()
  writeFileSync(join(directory, 'HANDOFF.md'), 'my own handoff\n')
  expect(await fail('QUOTA')).toEqual({ kind: 'retry' })
  expect(readFileSync(join(directory, 'HANDOFF.md'), 'utf8')).toBe('my own handoff\n')
})

it('restores the recorded route and switches of an earlier Assignment on the same ticket', async () => {
  const records = new Map<string, FallbackRecord>([['ticket-1', { active: { provider: 'second', model: 'two' },
    switches: [{ from: 'primary/one', to: 'second/two', reason: 'QUOTA', at: '2026-09-21T08:00:00.000Z' }] }]])
  const { fallback, session, request } = await fixture({ records })
  expect(await request()).toEqual({ provider: 'second', model: 'two' })
  expect(fallback.status('worker-ticket')).toMatchObject({ model_fallback: { active: 'second/two',
    switches: [{ from: 'primary/one', to: 'second/two' }] } })
  expect(session.snapshotEvents().some(event => (event as { type: string }).type === 'model/selection')).toBe(false)
})

it('does nothing while the chain is empty and keeps no status before a switch', async () => {
  const { fallback, fail, notify, records } = await fixture({ config: { models: [] } })
  expect(await fail('QUOTA')).toBeUndefined()
  expect(notify).not.toHaveBeenCalled()
  expect(records.size).toBe(0)
  expect(fallback.status('worker-ticket')).toEqual({})
  expect(fallback.reset()).toEqual({ reset: 0, model: null })
  expect(existsSync(join(tmpdir(), 'HANDOFF.md'))).toBe(false)
})
