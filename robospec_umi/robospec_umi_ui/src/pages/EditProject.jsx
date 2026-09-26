import { useCallback, useEffect, useRef, useState } from 'react'
import { Navigate, useNavigate, useParams } from 'react-router-dom'
import {
  FolderOpen, ShieldCheck, Scissors, PackageCheck, Loader2, AlertTriangle, CheckCircle2,
  XCircle, RotateCcw, ChevronDown, ChevronRight, Copy, Check, Database,
} from 'lucide-react'
import Page from '@/components/Page'
import Stepper from '@/components/Stepper'
import EpisodeTrimmer from '@/components/EpisodeTrimmer'
import IntrinsicLock from '@/components/IntrinsicLock'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { api, sse } from '@/lib/api'
import { cn } from '@/lib/utils'

const STAGES = [
  { id: 'select', label: 'Select Sessions', icon: FolderOpen },
  { id: 'verify', label: 'Verify Sessions', icon: ShieldCheck },
  { id: 'edit',   label: 'Edit Sessions', icon: Scissors },
  { id: 'export', label: 'Export Training Dataset', icon: PackageCheck },
]
const idx = (id) => STAGES.findIndex((s) => s.id === id)

function ErrorLine({ children }) {
  return children
    ? <div className="rounded-md bg-destructive/10 text-destructive px-3 py-2 text-sm">{children}</div>
    : null
}

/* ------------------------------------------------------ 1: select sessions */
function SelectStage({ project, onDone }) {
  const [rows, setRows] = useState(null)
  const [picked, setPicked] = useState(() => new Set(project?.sessions || []))
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.editSessions().then(setRows).catch((e) => { setErr(e.message); setRows([]) })
  }, [])

  const toggle = (name) => setPicked((p) => {
    const n = new Set(p)
    if (n.has(name)) n.delete(name)
    else n.add(name)
    return n
  })
  const total = (rows || []).filter((r) => picked.has(r.name))
    .reduce((a, r) => ({ eps: a.eps + r.episodes, s: a.s + r.duration_s }), { eps: 0, s: 0 })

  const go = async () => {
    setBusy(true)
    try { await onDone([...picked].sort()) }
    catch (e) { setErr(e.message) }
    finally { setBusy(false) }
  }

  return (
    <div className="space-y-4 max-w-4xl mx-auto w-full">
      <ErrorLine>{err}</ErrorLine>
      <p className="text-sm text-muted-foreground">
        Pick the sessions from <code>data/capture/</code> that go into this dataset. Episodes
        are used in place; nothing is copied.
      </p>
      {rows === null ? <p className="text-sm text-muted-foreground">loading…</p>
        : rows.length === 0 ? (
          <Card><CardContent className="pt-6 text-sm text-muted-foreground">
            No capture sessions with episodes yet.
          </CardContent></Card>
        ) : (
          <div className="rounded-lg border border-border overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-muted/50 text-muted-foreground">
                <tr className="text-left">
                  <th className="px-3 py-2 w-8" />
                  <th className="px-3 py-2 font-medium">session</th>
                  <th className="px-3 py-2 font-medium">episodes</th>
                  <th className="px-3 py-2 font-medium">duration</th>
                  <th className="px-3 py-2 font-medium">size</th>
                  <th className="px-3 py-2 font-medium">focus / zoom</th>
                  <th className="px-3 py-2 font-medium">prepared</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.name} onClick={() => toggle(r.name)}
                      className={cn('border-t border-border cursor-pointer hover:bg-muted/40',
                        picked.has(r.name) && 'bg-primary/5')}>
                    <td className="px-3 py-2">
                      <input type="checkbox" readOnly checked={picked.has(r.name)} className="accent-primary" />
                    </td>
                    <td className="px-3 py-2 font-medium">
                      {r.name}
                      {!r.complete && <Badge variant="outline" className="ml-2">no session.json</Badge>}
                    </td>
                    <td className="px-3 py-2 tabular-nums">{r.episodes}</td>
                    <td className="px-3 py-2 tabular-nums">{r.duration_s.toFixed(1)} s</td>
                    <td className="px-3 py-2 tabular-nums">{r.size_mb >= 1000 ? `${(r.size_mb / 1000).toFixed(1)} GB` : `${r.size_mb.toFixed(0)} MB`}</td>
                    <td className="px-3 py-2 tabular-nums">{r.focus ?? '—'} / {r.zoom ?? '—'}</td>
                    <td className="px-3 py-2 tabular-nums text-muted-foreground">
                      {r.prepped === r.episodes
                        ? <span className="text-[#22c55e]">all</span>
                        : `${r.prepped}/${r.episodes}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      <div className="flex items-center gap-3">
        <span className="text-sm text-muted-foreground">
          {picked.size} session{picked.size === 1 ? '' : 's'} · {total.eps} episodes · {total.s.toFixed(1)} s
        </span>
        <span className="flex-1" />
        <Button onClick={go} disabled={picked.size === 0 || busy}>
          {busy && <Loader2 className="size-4 animate-spin" />}
          Continue to Verify
        </Button>
      </div>
    </div>
  )
}

/* ------------------------------------------------------ 2: verify sessions */
function VerifyStage({ project, setProject, onDone }) {
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  const [openReport, setOpenReport] = useState(null)
  const ran = useRef(false)

  const run = useCallback(async () => {
    setBusy(true); setErr(null)
    try { setProject(await api.verifyProject(project.id)) }
    catch (e) { setErr(e.message) }
    finally { setBusy(false) }
  }, [project.id, setProject])

  const missing = project.sessions.filter((s) => !project.verify?.[s])
  useEffect(() => {
    if (missing.length && !ran.current) { ran.current = true; run() }
  }, [missing.length, run])

  const all = project.sessions.flatMap((s) => Object.values(project.verify?.[s]?.episodes || {}))
  const pass = all.filter((e) => e.ok).length

  return (
    <div className="space-y-4 max-w-4xl mx-auto w-full">
      <ErrorLine>{err}</ErrorLine>
      <div className="flex items-center gap-3">
        <p className="text-sm text-muted-foreground flex-1">
          Checks every episode&apos;s files and timestamps, and each session&apos;s focus and zoom
          against the active calibration. Episodes that fail are left out of the dataset.
        </p>
        <Button size="sm" variant="outline" onClick={run} disabled={busy}>
          {busy ? <Loader2 className="size-4 animate-spin" /> : <RotateCcw className="size-4" />}
          Re-run
        </Button>
      </div>

      {busy && missing.length > 0 && (
        <p className="text-sm text-muted-foreground flex items-center gap-2">
          <Loader2 className="size-4 animate-spin" /> verifying {project.sessions.length} session(s)…
        </p>
      )}

      {project.sessions.map((s) => {
        const r = project.verify?.[s]
        if (!r) return null
        const eps = Object.entries(r.episodes || {}).sort(([a], [b]) => a.localeCompare(b))
        return (
          <Card key={s}><CardContent className="pt-6 space-y-3">
            <div className="flex items-center gap-2">
              {r.ok ? <CheckCircle2 className="size-5 text-[#22c55e]" />
                : <XCircle className="size-5 text-destructive" />}
              <span className="font-medium">{s}</span>
              <span className="text-xs text-muted-foreground">
                {eps.filter(([, e]) => e.ok).length}/{eps.length} episodes pass
              </span>
            </div>
            {r.session_fails.map((f) => (
              <p key={f} className="text-sm text-destructive flex gap-2">
                <XCircle className="size-4 mt-0.5 shrink-0" /> {f} — every episode in this session is left out
              </p>
            ))}
            {r.warnings.length > 0 && (
              <p className="text-xs text-muted-foreground flex gap-2">
                <AlertTriangle className="size-3.5 mt-0.5 shrink-0 text-[#f59e0b]" />
                {r.warnings.join(' · ')}
              </p>
            )}
            <div className="flex flex-wrap gap-1.5">
              {eps.map(([ep, e]) => (
                <span key={ep} title={[...e.fails, ...e.warnings].join('\n') || 'all checks passed'}
                      className={cn('inline-flex items-center gap-1 rounded border px-2 py-0.5 text-xs',
                        e.ok ? 'border-[#22c55e]/40 text-[#22c55e]' : 'border-destructive/50 text-destructive')}>
                  {e.ok ? <Check className="size-3" /> : <XCircle className="size-3" />}
                  {ep}
                  {!e.ok && e.fails.length > 0 && !r.session_fails.includes(e.fails[0]) &&
                    <span className="text-muted-foreground">· {e.fails[0]}</span>}
                </span>
              ))}
            </div>
            <button type="button" onClick={() => setOpenReport(openReport === s ? null : s)}
                    className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1">
              {openReport === s ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
              full report
            </button>
            {openReport === s && (
              <pre className="text-[11px] leading-snug bg-muted/40 rounded p-3 max-h-80 overflow-auto">
                {r.lines.join('\n')}
              </pre>
            )}
          </CardContent></Card>
        )
      })}

      <div className="flex items-center gap-3">
        <span className="text-sm text-muted-foreground">
          {pass}/{all.length} episodes pass verify
        </span>
        <span className="flex-1" />
        <Button onClick={onDone} disabled={busy || missing.length > 0 || pass === 0}>
          Continue to Edit
        </Button>
      </div>
    </div>
  )
}

/* ------------------------------------------------ 4: export training dataset */
function ExportStage({ project, reload }) {
  const [rows, setRows] = useState(null)
  const [job, setJob] = useState(project.export_job || null)
  const [log, setLog] = useState([])
  const [status, setStatus] = useState(project.export_job ? 'running' : null)
  const [err, setErr] = useState(null)
  const [copied, setCopied] = useState(false)
  const logBox = useRef(null)
  const out = `data/dataset/${project.id}_dataset.zarr.zip`

  // what export will do with each included episode
  useEffect(() => {
    const eps = project.sessions.flatMap((s) => Object.entries(project.verify?.[s]?.episodes || {})
      .filter(([, v]) => v.ok).map(([ep]) => `${s}/${ep}`))
      .filter((k) => !project.episodes?.[k]?.excluded)
    let dead = false
    Promise.all(eps.map((k) => api.plan(k, project.episodes?.[k]?.trim)
      .then((p) => ({ key: k, ...p }))
      .catch((e) => ({ key: k, pending: e.status === 409, error: e.status === 409 ? null : e.message }))))
      .then((r) => { if (!dead) setRows(r) })
    return () => { dead = true }
  }, [project])

  useEffect(() => {
    if (!job) return undefined
    setLog([])
    return sse(`/api/jobs/${job}/log`, (m) => {
      if (m.line !== undefined) setLog((l) => [...l, m.line])
      if (m.done) { setStatus(m.done); reload() }
    })
  }, [job, reload])
  useEffect(() => { logBox.current?.scrollTo(0, logBox.current.scrollHeight) }, [log])

  const start = async () => {
    if (project.export && !confirm(`Overwrite ${out}?`)) return
    setErr(null)
    try {
      const r = await api.exportProject(project.id)
      setStatus('running')
      setJob(r.job_id)
    } catch (e) { setErr(e.message) }
  }

  const kept = (rows || []).filter((r) => r.reason === null)
  const drops = (rows || []).filter((r) => r.reason)
  const pending = (rows || []).filter((r) => r.pending)
  const secs = kept.reduce((a, r) => a + r.kept_s, 0)
  const running = status === 'running'
  const ex = project.export

  return (
    <div className="space-y-4 max-w-4xl mx-auto w-full">
      <ErrorLine>{err}</ErrorLine>
      <Card><CardContent className="pt-6 space-y-3">
        <div className="flex items-center gap-2">
          <Database className="size-5 text-primary" />
          <span className="font-medium">{out}</span>
        </div>
        {rows === null ? <p className="text-sm text-muted-foreground">checking episodes…</p> : (
          <dl className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
            <div><dt className="text-xs text-muted-foreground">kept</dt>
                 <dd className="tabular-nums">{kept.length} episodes · {secs.toFixed(1)} s</dd></div>
            <div><dt className="text-xs text-muted-foreground">dropped by export</dt>
                 <dd className="tabular-nums">{drops.length}</dd></div>
            <div><dt className="text-xs text-muted-foreground">still to prepare</dt>
                 <dd className="tabular-nums">{pending.length}{pending.length > 0 && ' (export does it)'}</dd></div>
            <div><dt className="text-xs text-muted-foreground">grid</dt>
                 <dd className="tabular-nums">60 Hz · wrist 224×224</dd></div>
          </dl>
        )}
        {drops.length > 0 && (
          <ul className="text-xs text-[#f59e0b] space-y-0.5">
            {drops.map((r) => <li key={r.key}>{r.key}: {r.reason}</li>)}
          </ul>
        )}
        <div className="flex items-center gap-2">
          <Button onClick={start} disabled={running || rows === null}>
            {running && <Loader2 className="size-4 animate-spin" />}
            {running ? 'Exporting…' : ex ? 'Export again' : 'Export dataset'}
          </Button>
          <span className="text-xs text-muted-foreground">
            Preparation pauses while this runs.
          </span>
        </div>
      </CardContent></Card>

      {ex && !running && (
        <Card><CardContent className="pt-6 space-y-2">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="size-5 text-[#22c55e]" />
            <span className="font-medium">Exported {ex.at?.replace('T', ' ')}</span>
          </div>
          <p className="text-sm tabular-nums">
            {ex.episodes} episodes · {ex.steps} steps ({ex.duration_s} s at 60 Hz) · {ex.size_mb} MB
          </p>
          {ex.dropped?.length > 0 && (
            <ul className="text-xs text-[#f59e0b]">
              {ex.dropped.map((d) => <li key={d.episode}>dropped {d.episode}: {d.reason}</li>)}
            </ul>
          )}
          <div className="flex items-center gap-2 text-xs">
            <code className="bg-muted/50 rounded px-2 py-1 truncate">{ex.path}</code>
            <button type="button" title="Copy path" onClick={() => {
              navigator.clipboard?.writeText(ex.path); setCopied(true); setTimeout(() => setCopied(false), 1500)
            }}>
              {copied ? <Check className="size-3.5 text-[#22c55e]" /> : <Copy className="size-3.5" />}
            </button>
          </div>
        </CardContent></Card>
      )}

      {(running || log.length > 0) && (
        <div>
          <p className={cn('text-xs mb-1', status === 'failed' ? 'text-destructive' : 'text-muted-foreground')}>
            {status === 'failed' ? 'export failed' : status === 'done' ? 'export log' : 'exporting…'}
          </p>
          <pre ref={logBox} className="text-[11px] leading-snug bg-muted/40 rounded p-3 h-72 overflow-auto">
            {log.join('\n')}
          </pre>
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------- page */
export default function EditProject() {
  const { pid, stage = 'select' } = useParams()
  const nav = useNavigate()
  const isNew = pid === 'new'
  const [project, setProject] = useState(isNew ? null : undefined)
  const [active, setActive] = useState(undefined)
  const [err, setErr] = useState(null)

  const reload = useCallback(async () => {
    if (isNew) {
      api.active().then(setActive).catch((e) => setErr(e.message))
      return
    }
    try { setProject(await api.project(pid)); setErr(null) }
    catch (e) { setErr(e.message); setProject((p) => p ?? null) }
  }, [pid, isNew])
  useEffect(() => { reload() }, [reload])

  if (project === undefined) {
    return <Page title="Edit dataset" back="/edit">
      <p className="text-sm text-muted-foreground text-center">loading…</p>
    </Page>
  }
  if (!isNew && project === null) {
    return <Page title="Edit dataset" back="/edit">
      <div className="max-w-md mx-auto space-y-3 text-center">
        <ErrorLine>{err || 'project not found'}</ErrorLine>
        <Button variant="outline" onClick={reload}><RotateCcw className="size-4" /> Retry</Button>
      </div>
    </Page>
  }

  const verified = project && project.sessions.every((s) => project.verify?.[s])
  const reached = (id) => {
    if (id === 'select') return true
    if (!project) return false
    if (id === 'verify') return true
    return verified && idx(id) <= idx(project.stage)
  }
  if (!reached(stage)) {
    const earliest = STAGES.filter((s) => reached(s.id)).pop()
    return <Navigate to={`/edit/${isNew ? 'new' : pid}/${earliest.id}`} replace />
  }

  const lock = isNew ? (active === null ? { status: 'none', active: null } : null)
    : project.intrinsics_state?.status !== 'ok' ? project.intrinsics_state : null
  const unlocked = (r) => {
    if (r?.id) { setProject(r); nav(`/edit/${r.id}/verify`) }   // rebound
    else reload()                                               // activated
  }

  const advance = async (to) => {
    setProject(await api.saveProject(project.id, { stage: to }))
    nav(`/edit/${project.id}/${to}`)
  }

  return (
    <Page title={isNew ? 'New dataset' : `Dataset ${pid}`} back="/edit" fill wide>
      <div className="pb-4 mb-4 border-b border-border shrink-0">
        <Stepper stages={STAGES} current={stage} reached={reached}
                 onPick={(id) => nav(`/edit/${isNew ? 'new' : pid}/${id}`)} />
      </div>
      {err && <div className="mb-3"><ErrorLine>{err}</ErrorLine></div>}
      <div className={cn('flex-1 min-h-0 flex flex-col', (stage !== 'edit' || lock) && 'overflow-y-auto')}>
        {lock && <IntrinsicLock state={lock} pid={pid} exported={!!project?.export} onChange={unlocked} />}
        {!lock && stage === 'select' && (
          <SelectStage project={project} onDone={async (sessions) => {
            const p = isNew ? await api.createProject(sessions)
              : await api.saveProject(project.id, { sessions })
            setProject(p)
            nav(`/edit/${p.id}/verify`)
          }} />
        )}
        {!lock && stage === 'verify' && (
          <VerifyStage project={project} setProject={setProject}
                       onDone={() => advance('edit').catch((e) => setErr(e.message))} />
        )}
        {!lock && stage === 'edit' && (
          <EpisodeTrimmer key={project.id} project={project} onUnbound={reload}
                          onEdits={(episodes) => setProject((p) => ({ ...p, episodes }))}
                          onContinue={() => advance('export').catch((e) => setErr(e.message))} />
        )}
        {!lock && stage === 'export' && <ExportStage project={project} reload={reload} />}
      </div>
    </Page>
  )
}
