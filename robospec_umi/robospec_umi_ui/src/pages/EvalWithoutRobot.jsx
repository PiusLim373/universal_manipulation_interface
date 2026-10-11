import { useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  Upload, Loader2, Shuffle, Play, RotateCcw, AlertTriangle, ChevronDown, ChevronRight,
} from 'lucide-react'
import Page from '@/components/Page'
import EpisodePlayer, { Waiting } from '@/components/EpisodePlayer'
import ForecastGraphs, { ForecastOverlay } from '@/components/ForecastGraphs'
import { Button } from '@/components/ui/button'
import { api, sse, uploadCheckpoint } from '@/lib/api'
import { cn, fmtMB } from '@/lib/utils'

// the honest test is the held-out one; trained-on episodes mostly show memory
const SPLIT = {
  held_out: { label: 'held out', color: '#38bdf8', tip: 'never trained on' },
  trained_on: { label: 'trained on', color: '#a1a1aa', tip: 'in the training set' },
  not_in_dataset: { label: 'not in dataset', color: '#f59e0b', tip: 'not in the dataset this checkpoint trained on' },
}
const VERIFY = {
  verified: ['verified', '#22c55e', 'recomputed from the recording, it reproduces the dataset exactly'],
  'steps only': ['steps only', '#f59e0b', 'the dataset is not here, so only the step counts were checked'],
  mismatch: ['mismatch', '#ef4444', ''],
}

const splitOf = (row) => (!row.segments ? 'not_in_dataset'
  : row.segments.some((s) => s.held_out) ? 'held_out' : 'trained_on')
const better = (m, s) => (s > 0 ? Math.round((1 - m / s) * 100) : null)

function Tag({ color, children, title }) {
  return (
    <span className="text-[10px] px-1.5 rounded border whitespace-nowrap" title={title}
          style={{ color, borderColor: `${color}80` }}>{children}</span>
  )
}

/* ------------------------------------------------------------ checkpoint */
function CheckpointUpload({ onDone }) {
  const input = useRef(null)
  const [up, setUp] = useState(null)
  const busy = up && !up.done && !up.err
  const start = async (f) => {
    input.current.value = ''
    setUp({ name: f.name, loaded: 0, total: f.size })
    try {
      const r = await uploadCheckpoint(f, (loaded, total) => setUp((u) => ({ ...u, loaded, total })))
      setUp((u) => ({ ...u, done: r.path }))
      onDone(r.path)
    } catch (e) { setUp((u) => ({ ...u, err: e.message })) }
  }
  return (
    <div className="space-y-1.5">
      <input ref={input} type="file" accept=".ckpt" className="hidden"
             onChange={(e) => e.target.files[0] && start(e.target.files[0])} />
      <Button size="sm" variant="outline" className="w-full" disabled={busy}
              onClick={() => input.current.click()} title="a .ckpt trained on another PC">
        {busy ? <Loader2 className="size-4 animate-spin" /> : <Upload className="size-4" />}
        Upload checkpoint
      </Button>
      {up && (
        <div className="text-[11px] space-y-1">
          <p className="font-mono truncate">{up.name}</p>
          {busy && <>
            <div className="h-1 rounded bg-muted overflow-hidden">
              <div className="h-full bg-primary" style={{ width: `${(100 * up.loaded) / (up.total || 1)}%` }} />
            </div>
            <p className="text-muted-foreground tabular-nums">
              {fmtMB(up.loaded / 1e6)} / {fmtMB(up.total / 1e6)}
            </p>
          </>}
          {up.err && <p className="text-destructive">{up.err}</p>}
          {up.done && <p className="text-muted-foreground break-all">filed as data/outputs/{up.done}</p>}
        </div>
      )}
    </div>
  )
}

/* -------------------------------------------------------------- episodes */
function EpRow({ row, selected, onPick }) {
  const s = SPLIT[splitOf(row)]
  const h = row.result && Object.values(row.result)[0]
  return (
    <button type="button" disabled={!row.here} onClick={() => onPick(row.key)}
            className={cn('w-full text-left px-3 py-1.5 border-b border-border/50 transition-colors',
              selected ? 'bg-primary/10 border-l-2 border-l-primary' : 'hover:bg-muted/40',
              !row.here && 'opacity-40 cursor-not-allowed')}>
      <div className="flex items-center gap-2 text-xs">
        <span className="size-1.5 rounded-full shrink-0" style={{ background: s.color }} />
        <span className="font-mono truncate flex-1">{row.key}</span>
        {!row.here && <span className="text-[10px] text-muted-foreground">not here</span>}
      </div>
      {h && (
        <p className="text-[10px] text-muted-foreground tabular-nums pl-3.5">
          {h.s.toFixed(2)} s ahead: {h.pos_mm.toFixed(1)} mm · still {h.pos_still_mm.toFixed(1)}
        </p>
      )}
    </button>
  )
}

function Group({ title, rows, ep, onPick, open: openDefault = true }) {
  const [open, setOpen] = useState(openDefault)
  if (!rows.length) return null
  return (
    <div>
      <button type="button" onClick={() => setOpen((o) => !o)}
              className="w-full flex items-center gap-1.5 px-2 py-1.5 text-[11px] font-medium
                         text-muted-foreground bg-muted/60 hover:bg-muted sticky top-0 z-10">
        {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
        {title}
        <span className="ml-auto font-normal tabular-nums">{rows.length}</span>
      </button>
      {open && rows.map((r) => <EpRow key={r.key} row={r} selected={r.key === ep} onPick={onPick} />)}
    </div>
  )
}

/* --------------------------------------------------------------- results */
function Metrics({ r }) {
  const far = r.horizon - 1
  return (
    <div className="flex gap-3 flex-wrap">
      {Object.entries(r.metrics).map(([split, m]) => (
        <div key={split} className="rounded-md border border-border px-3 py-2 text-xs space-y-1">
          <div className="flex items-center gap-2">
            <Tag color={SPLIT[split].color}>{SPLIT[split].label}</Tag>
            <span className="text-muted-foreground">{m.forecasts} forecasts</span>
          </div>
          <table className="tabular-nums">
            <thead className="text-[10px] text-muted-foreground">
              <tr><th className="pr-3 text-left font-normal">ahead</th>
                <th className="pr-3 text-left font-normal">position</th>
                <th className="pr-3 text-left font-normal">rotation</th>
                <th className="text-left font-normal">gripper</th></tr>
            </thead>
            <tbody>
              {m.ahead.filter((a) => a.step === r.n_action_steps || a.step === far).map((a) => (
                <tr key={a.step}>
                  <td className="pr-3">{a.s.toFixed(2)} s</td>
                  {[['pos_mm', 'pos_still_mm', 'mm'], ['rot_deg', 'rot_still_deg', '°'],
                    ['grip_mm', 'grip_still_mm', 'mm']].map(([e, w, u]) => {
                    const b = better(a[e], a[w])
                    return (
                      <td key={e} className="pr-3" title={`standing still: ${a[w].toFixed(1)} ${u}`}>
                        {a[e].toFixed(1)} {u}
                        <span className="text-muted-foreground"> / {a[w].toFixed(1)}</span>
                        {b != null && (
                          <span className={b >= 0 ? 'text-[#22c55e]' : 'text-destructive'}>
                            {' '}{b >= 0 ? `${b}% better` : `${-b}% worse`}
                          </span>
                        )}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      <p className="text-[11px] text-muted-foreground max-w-xs self-center">
        model / standing still, and how much of standing still&apos;s error the model removes.
        Open loop: every forecast starts from the recorded pose.
      </p>
    </div>
  )
}

/* ------------------------------------------------------------------ page */
export default function EvalWithoutRobot() {
  const [params, setParams] = useSearchParams()
  const ckpt = params.get('ckpt')
  const ep = params.get('ep')
  const [ckpts, setCkpts] = useState(null)
  const [info, setInfo] = useState(null)
  const [infoErr, setInfoErr] = useState(null)
  const [job, setJob] = useState(null)
  const [got, setGot] = useState(null)      // {id, res, tl, err} for the shown (ckpt, ep)
  const [runErr, setRunErr] = useState(null)
  const [tries, setTries] = useState(0)

  const pick = (patch) => setParams((p) => {
    const n = new URLSearchParams(p)
    for (const [k, v] of Object.entries(patch)) {
      if (v == null) n.delete(k)
      else n.set(k, v)
    }
    return n
  }, { replace: true })

  const loadCkpts = (select) => api.evalCheckpoints().then((list) => {
    setCkpts(list)
    const first = list.find((c) => !c.error)
    if (select) pick({ ckpt: select, ep: null })
    else if (!ckpt && first) pick({ ckpt: first.path })
  }).catch((e) => setInfoErr(e.message))
  useEffect(() => { loadCkpts() }, []) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => sse('/api/eval/state', setJob), [])

  // a finished run refreshes the list's numbers and the result shown
  const mine = job && job.path === ckpt && job.episode === ep
  const finished = job?.status === 'done' ? job.at : null

  useEffect(() => {
    if (!ckpt) return undefined
    let dead = false
    api.evalCheckpoint(ckpt).then((i) => { if (!dead) { setInfo(i); setInfoErr(null) } })
      .catch((e) => { if (!dead) { setInfo(null); setInfoErr(e.message) } })
    return () => { dead = true }
  }, [ckpt, finished])

  const id = `${ckpt}|${ep}`
  useEffect(() => {
    if (!ckpt || !ep) return undefined
    let dead = false
    api.evalResult(ckpt, ep)
      .then(async (res) => {
        const tl = await api.timeline(ep)
        if (!dead) setGot({ id, res, tl })
      })
      .catch((e) => { if (!dead) setGot({ id, err: e.status === 404 ? null : e.message }) })
    return () => { dead = true }
  }, [ckpt, ep, finished, tries]) // eslint-disable-line react-hooks/exhaustive-deps
  const cur = got?.id === id ? got : null

  const run = async (force) => {
    setRunErr(null)
    try { await api.evalRun(ckpt, ep, force) } catch (e) { setRunErr(e.message) }
  }

  const rows = useMemo(() => {
    const eps = info?.episodes || []
    return {
      held: eps.filter((r) => splitOf(r) === 'held_out'),
      trained: eps.filter((r) => splitOf(r) === 'trained_on'),
      others: info?.others || [],
    }
  }, [info])
  const random = () => {
    const pool = [...rows.held, ...rows.trained].filter((r) => r.here)
    if (pool.length) pick({ ep: pool[Math.floor(Math.random() * pool.length)].key })
  }
  const row = [...rows.held, ...rows.trained, ...rows.others].find((r) => r.key === ep)
  const busy = job && (job.status === 'prep' || job.status === 'running')
  const runs = useMemo(() => {
    const m = new Map()
    for (const c of ckpts || []) m.set(c.run, [...(m.get(c.run) || []), c])
    return [...m.entries()]
  }, [ckpts])

  const r = cur?.res
  const split = row ? SPLIT[splitOf(row)] : null
  const verify = r && VERIFY[r.verify.status]

  return (
    <Page title="Eval Without Robot" back="/evaluation" fill wide>
      <div className="flex-1 flex min-h-0 gap-4">
        {/* ---------------- left: checkpoint + episodes */}
        <div className="w-72 shrink-0 flex flex-col min-h-0 border border-border rounded-lg overflow-hidden">
          <div className="p-3 border-b border-border space-y-2">
            <p className="text-sm font-medium">Checkpoint</p>
            {ckpts === null ? <p className="text-xs text-muted-foreground">loading…</p>
              : ckpts.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  No checkpoints in data/outputs/ yet. Upload one trained on another PC.
                </p>
              ) : (
                <select value={ckpt || ''} onChange={(e) => pick({ ckpt: e.target.value, ep: null })}
                        className="w-full bg-background border border-border rounded-md px-2 py-1.5 text-xs font-mono">
                  {runs.map(([run, cs]) => (
                    <optgroup key={run} label={run}>
                      {cs.map((c) => (
                        <option key={c.path} value={c.path} disabled={!!c.error}>
                          {c.name}{c.error ? ' (unreadable)' : ''}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>
              )}
            <CheckpointUpload onDone={(path) => loadCkpts(path)} />
            {infoErr && <p className="text-xs text-destructive">{infoErr}</p>}
            {info && (
              <div className="text-[11px] text-muted-foreground space-y-0.5">
                <p>epoch {info.epoch} · step {info.global_step}</p>
                <p className="font-mono break-all">trained on {info.dataset}</p>
                {!info.dataset_here && (
                  <p className="text-[#f59e0b]">
                    its project file is not here, so no recording can be told apart as held out
                  </p>
                )}
              </div>
            )}
          </div>
          <div className="px-3 py-2 border-b border-border flex items-center">
            <span className="text-sm font-medium flex-1">Episodes</span>
            <Button size="xs" variant="outline" onClick={random}
                    disabled={!rows.held.length && !rows.trained.length}
                    title="any held-out or trained-on episode">
              <Shuffle className="size-3" /> Random
            </Button>
          </div>
          <div className="flex-1 overflow-y-auto">
            <Group title="Held out" rows={rows.held} ep={ep} onPick={(k) => pick({ ep: k })} />
            <Group title="Trained on" rows={rows.trained} ep={ep} onPick={(k) => pick({ ep: k })} />
            <Group title="Other recordings" rows={rows.others} ep={ep} onPick={(k) => pick({ ep: k })}
                   open={false} />
          </div>
        </div>

        {/* ---------------- right: result */}
        <div className="flex-1 min-w-0 flex flex-col min-h-0 gap-3">
          {!ep ? (
            <div className="flex-1 grid place-items-center text-sm text-muted-foreground">
              Pick an episode, or a random one.
            </div>
          ) : <>
            <div className="flex items-center gap-2 text-sm flex-wrap">
              <span className="font-medium font-mono">{ep}</span>
              {split && <Tag color={split.color} title={split.tip}>{split.label}</Tag>}
              {verify && (
                <Tag color={verify[1]} title={r.verify.detail || verify[2]}>
                  {verify[0]}{r.verify.input_diff != null && ` · input ${r.verify.input_diff}/255`}
                </Tag>
              )}
              {r?.stale && <Tag color="#f59e0b" title="the checkpoint, the recording's prep or the intrinsics changed since">
                out of date</Tag>}
              {r?.trim && <span className="text-xs text-muted-foreground">trim {r.trim.map((v) => v ?? '…').join(' – ')} s</span>}
              <span className="flex-1" />
              {runErr && <span className="text-xs text-destructive">{runErr}</span>}
              {r && (
                <Button size="sm" variant="outline" disabled={busy} onClick={() => run(true)}>
                  <RotateCcw className="size-4" /> Re-run
                </Button>
              )}
            </div>
            {r && <Metrics r={r} />}

            {mine && busy ? (
              <Waiting label={ep} prep={{ status: 'running', stage: job.stage, pct: job.pct }}
                       note="Prepping the recording if needed, then a forecast at every observation step." />
            ) : mine && job.status === 'failed' && !r ? (
              <Waiting label={ep} err={job.err} onRetry={() => run(false)} />
            ) : cur?.err ? (
              <Waiting label={ep} err={cur.err} onRetry={() => setTries((n) => n + 1)} />
            ) : !cur ? (
              <div className="flex-1 grid place-items-center">
                <Loader2 className="size-6 animate-spin text-muted-foreground" />
              </div>
            ) : !r ? (
              <div className="flex-1 flex flex-col items-center justify-center gap-3 bg-black/40 rounded-md">
                <p className="text-sm">Not evaluated yet</p>
                <p className="text-xs text-muted-foreground max-w-sm text-center">
                  The model forecasts the next {info ? ((info.horizon - 1) * info.step_s).toFixed(2) : '…'} s at
                  every observation step, from the raw wrist video and the recorded TCP.
                </p>
                {busy && <p className="text-xs text-[#f59e0b] flex items-center gap-1">
                  <AlertTriangle className="size-3.5" /> evaluating {job.episode}; one at a time</p>}
                <Button onClick={() => run(false)} disabled={busy || !row?.here}>
                  <Play className="size-4" /> Run
                </Button>
              </div>
            ) : (
              <EpisodePlayer key={`${id}|${r.made_at}`} tl={cur.tl}
                             plan={{ spans: r.segments.map((s) => s.span) }}
                             src={(f) => api.media(ep, f, cur.tl.meta.prepared_at)}
                             graphs={({ time }) => <ForecastGraphs tl={cur.tl} r={r} time={time} />}
                             sceneOverlay={({ time }) => <ForecastOverlay r={r} time={time} />}
                             spanLabel="evaluated"
                             note={<span>
                               forecasts dashed, the playhead&apos;s bold · scene: <span className="text-white">recorded</span>
                               {' '}vs <span className="text-[#f59e0b]">forecast</span>, next {((r.horizon - 1) * r.step_s).toFixed(2)} s
                             </span>} />
            )}
          </>}
        </div>
      </div>
    </Page>
  )
}
