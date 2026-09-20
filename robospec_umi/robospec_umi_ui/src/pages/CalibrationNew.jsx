import { useCallback, useEffect, useRef, useState } from 'react'
import { Navigate, useNavigate, useParams, Link } from 'react-router-dom'
import { Eye, Camera, Calculator, Check, X, RotateCcw, Play, Pause, AlertTriangle, Loader2 } from 'lucide-react'
import Page from '@/components/Page'
import CameraStream from '@/components/CameraStream'
import CoverageMeters from '@/components/CoverageMeters'
import CameraBusy from '@/components/CameraBusy'
import PreviewLock from '@/components/PreviewLock'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { api, sse, ApiError, CALIB_PREVIEW } from '@/lib/api'
import { cn } from '@/lib/utils'

const STAGES = [
  { id: 'lock',    label: 'Preview & Lock', icon: Eye },
  { id: 'capture', label: 'Capture',        icon: Camera },
  { id: 'solve',   label: 'Solve',          icon: Calculator },
]
const MIN_FRAMES = 8

function Stepper({ current, reached }) {
  const at = STAGES.findIndex((s) => s.id === current)
  return (
    <div className="flex items-center gap-1">
      {STAGES.map((s, i) => {
        const Icon = s.icon
        const done = i < at
        const now = i === at
        return (
          <div key={s.id} className="flex items-center gap-1">
            <div className={cn(
              'flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm transition-colors',
              now && 'bg-primary text-primary-foreground font-medium',
              done && 'text-[#22c55e]',
              !now && !done && 'text-muted-foreground',
              !reached(s.id) && 'opacity-40',
            )}>
              {done ? <Check className="size-4" /> : <Icon className="size-4" />}
              {s.label}
            </div>
            {i < STAGES.length - 1 && <div className="w-5 h-px bg-border" />}
          </div>
        )
      })}
    </div>
  )
}

/* ------------------------------------------------------ stage 2: capture */
function CaptureStage({ onSolve, onProgress }) {
  const [state, setState] = useState(null)
  const [runDir, setRunDir] = useState(null)
  const [busyOwner, setBusyOwner] = useState(null)
  const [err, setErr] = useState(null)
  const [started, setStarted] = useState(false)
  const unsubRef = useRef(null)

  const start = useCallback(async () => {
    setErr(null); setBusyOwner(null)
    try {
      const r = await api.startSession()
      setRunDir(r.run_dir); setStarted(true)
      unsubRef.current = sse('/api/calibration/state', (s) => {
        setState(s); onProgress?.(s.saved)
      })
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && e.owner) setBusyOwner(e.owner)
      else setErr(e.message)
    }
  }, [onProgress])

  useEffect(() => { start(); return () => { unsubRef.current?.() } }, [start])

  const reset = async () => {
    unsubRef.current?.()
    await api.resetSession(false)
    setState(null); setRunDir(null); setStarted(false); onProgress?.(0)
    start()
  }

  if (busyOwner) return <CameraBusy owner={busyOwner} onReleased={start} />
  if (err) return <p className="text-sm text-destructive">{err}</p>
  if (!started) return <p className="text-sm text-muted-foreground">Opening camera…</p>

  const auto = state?.auto
  const saved = state?.saved ?? 0

  return (
    <div className="flex-1 min-h-0 grid lg:grid-cols-[1fr_20rem] gap-5">
      <div className="flex flex-col gap-3 min-h-0">
        <CameraStream src="/api/calibration/preview" className="flex-1 min-h-0" />
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={() => api.setAuto(!auto)} variant={auto ? 'secondary' : 'default'}>
            {auto ? <><Pause className="size-4" /> Pause capture</>
                  : <><Play className="size-4" /> Start calibration</>}
          </Button>
          <Button variant="outline" onClick={() => api.keep()}>Keep frame</Button>
          <Button variant="ghost" onClick={() => api.undo()}>Undo</Button>
          <div className="flex-1" />
          <Button variant="destructive" onClick={reset}>
            <RotateCcw className="size-4" /> Reset
          </Button>
          <Button disabled={saved < MIN_FRAMES} onClick={() => onSolve(runDir)}>
            Solve ({saved})
          </Button>
        </div>
        {runDir && (
          <p className="text-[11px] text-muted-foreground font-mono break-all">
            frames → {runDir} · Reset deletes this directory and everything in it
          </p>
        )}
      </div>

      {/* mt-auto pushes the cards to the bottom when there is room and does
          nothing when there is not -- unlike justify-end, which would make the
          overflow unscrollable once these grow. */}
      <div className="flex flex-col gap-4 min-h-0 overflow-auto">
        <Card className="mt-auto"><CardContent className="pt-6 space-y-3">
          <div className="flex items-baseline justify-between">
            <span className="text-2xl font-heading tabular-nums">{saved}</span>
            <span className="text-xs text-muted-foreground">frames saved</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            <Badge variant={state?.enough ? 'default' : 'outline'}>{state?.corners ?? 0} corners</Badge>
            <Badge variant={state?.still ? 'default' : 'destructive'}>
              {state?.still ? 'still' : 'moving'}
            </Badge>
            <Badge variant="outline">{state?.scale_bin}</Badge>
            <Badge variant="outline">{state?.tilt_bin}</Badge>
          </div>
          {state?.exposure && (
            <p className={cn('text-xs', state.exposure.verdict === 'exposure ok'
              ? 'text-muted-foreground' : 'text-[#f59e0b]')}>
              {state.exposure.verdict} · mean {state.exposure.mean}
            </p>
          )}
        </CardContent></Card>

        <Card><CardContent className="pt-6">
          <CoverageMeters coverage={state?.coverage} />
        </CardContent></Card>
      </div>
    </div>
  )
}

/* -------------------------------------------------------- stage 3: solve */
const N = (v, d = 4) => (v === null || v === undefined || Number.isNaN(v)
  ? '\u2014' : Number(v).toFixed(d))

function Row({ label, value, sub }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5 border-b border-border/40 last:border-0">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-sm tabular-nums text-right">
        {value}
        {sub && <span className="ml-1.5 text-[11px] text-muted-foreground">{sub}</span>}
      </span>
    </div>
  )
}

function IntrinsicsTable({ j }) {
  if (!j) return null
  const k = j.k || []
  const sd = j.std_dev || {}
  const fov = j.fov_deg || {}
  const lc = j.locked_controls || {}
  const aspect = k[4] && k[0] ? k[4] / k[0] : null
  return (
    <div className="grid md:grid-cols-3 gap-x-8 gap-y-1">
      <div>
        <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground mb-1">focal length</p>
        <Row label="fx" value={N(k[0], 2)} sub={sd.fx ? `± ${N(sd.fx, 2)}` : null} />
        <Row label="fy" value={N(k[4], 2)} sub={sd.fy ? `± ${N(sd.fy, 2)}` : null} />
        <Row label="aspect fy/fx" value={N(aspect, 5)}
             sub={aspect && Math.abs(aspect - 1) > 0.01 ? 'off square' : null} />
      </div>
      <div>
        <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground mb-1">principal point</p>
        <Row label="cx" value={N(k[2], 2)} sub={sd.cx ? `± ${N(sd.cx, 2)}` : null} />
        <Row label="cy" value={N(k[5], 2)} sub={sd.cy ? `± ${N(sd.cy, 2)}` : null} />
        <Row label="image" value={`${j.image_width} × ${j.image_height}`} />
      </div>
      <div>
        <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground mb-1">quality</p>
        <Row label="reprojection" value={`${N(j.final_reproj_error)} px`} />
        <Row label="held out" value={`${N(j.holdout_reproj_error)} px`} />
        <Row label="images used" value={j.nr_calib_images ?? '\u2014'} />
      </div>
      <div>
        <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground mb-1">distortion</p>
        <Row label="model" value={j.distortion_model ?? '\u2014'} />
        <div className="py-1.5 text-[11px] font-mono text-muted-foreground break-all leading-relaxed">
          {(j.d || []).map((v) => Number(v).toFixed(5)).join('  ')}
        </div>
      </div>
      <div>
        <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground mb-1">geometry</p>
        <Row label="field of view" value={`${N(fov.horizontal, 1)}° × ${N(fov.vertical, 1)}°`} />
        <Row label="focus / zoom" value={`${lc.focus_absolute ?? '\u2014'} / ${lc.zoom_absolute ?? '\u2014'}`} />
      </div>
      <div>
        <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground mb-1">provenance</p>
        <Row label="from run" value={j.source_run ?? '\u2014'} />
        <Row label="solved" value={j.solved_at ?? '\u2014'} />
      </div>
    </div>
  )
}

function SolveStage({ runDir }) {
  const [lines, setLines] = useState([])
  const [status, setStatus] = useState('starting')
  const [intr, setIntr] = useState(null)
  const [err, setErr] = useState(null)
  const logRef = useRef(null)
  const nav = useNavigate()
  const runName = runDir ? runDir.split('/').filter(Boolean).pop() : null

  useEffect(() => {
    let unsub
    ;(async () => {
      try {
        const { job_id } = await api.solve(runName)
        setStatus('running')
        unsub = sse(`/api/jobs/${job_id}/log`, (m) => {
          if (m.line !== undefined) setLines((L) => [...L, m.line])
          if (m.done) setStatus(m.done)
        })
      } catch (e) { setErr(e.message); setStatus('failed') }
    })()
    return () => unsub?.()
  }, [runName])

  // The solver writes the JSON; read it back rather than re-deriving anything
  // from the log, which would mean parsing prose.
  useEffect(() => {
    if (status !== 'done' || !runName) return
    fetch(api.runFile(runName, 'scene_intrinsics.json'))
      .then((r) => (r.ok ? r.json() : Promise.reject(
        new Error(`could not read scene_intrinsics.json (HTTP ${r.status})`))))
      .then(setIntr)
      // Solve said it succeeded but the result is unreadable -- that is worth
      // saying out loud, not hiding behind a table that silently never appears.
      .catch((e) => setErr(e.message))
  }, [status, runName])

  useEffect(() => {
    if (status === 'failed') logRef.current?.scrollTo(0, logRef.current.scrollHeight)
  }, [status, lines])

  const latest = lines.length ? lines[lines.length - 1] : 'starting the solver…'

  return (
    <div className="space-y-5 w-full">
      {status === 'running' && (
        <Card><CardContent className="pt-6 flex items-center gap-3">
          <Loader2 className="size-4 animate-spin text-primary shrink-0" />
          <div className="min-w-0">
            <p className="text-sm">Solving…</p>
            {/* The newest log line as a status: enough to show it is alive
                through a minute of blocking work, without a terminal. */}
            <p className="text-[11px] text-muted-foreground font-mono truncate">{latest}</p>
          </div>
        </CardContent></Card>
      )}

      {status === 'done' && (
        <>
          <div className="space-y-2">
            <img src={api.runFile(runName, 'undistort_preview.png')} alt="raw | undistorted"
                 className="w-full rounded-lg border border-border" />
            <p className="text-[11px] text-muted-foreground">
              Left raw, right undistorted. Check the frame edges — straight edges
              in the scene should be straight on the right. That is the part
              reprojection error cannot tell you, because it only scores pixels
              that had observations.
            </p>
          </div>

          <Card><CardContent className="pt-6">
            <IntrinsicsTable j={intr} />
          </CardContent></Card>

          <div className="flex gap-2">
            <Button onClick={async () => { await api.activate(runName); nav('/calibration') }}>
              Activate
            </Button>
            <Link to={`/calibration/test?run=${encodeURIComponent(runName)}`}>
              <Button variant="outline">Test it</Button>
            </Link>
            <Link to="/calibration"><Button variant="ghost">Done</Button></Link>
          </div>
        </>
      )}

      {status === 'failed' && (
        <Card><CardContent className="pt-6 space-y-3">
          <div className="flex items-center gap-2 text-destructive">
            <AlertTriangle className="size-4" />
            <span className="text-sm font-medium">Solve failed</span>
          </div>
          {err && <p className="text-sm text-destructive">{err}</p>}
          {/* Shown only on failure, because the log is the only place the
              reason appears. */}
          <div ref={logRef}
               className="h-72 overflow-auto rounded-lg border border-border bg-muted/30 p-3
                          font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
            {lines.join('\n') || 'no output'}
          </div>
          <Link to="/calibration"><Button size="sm" variant="outline">Back</Button></Link>
        </CardContent></Card>
      )}
    </div>
  )
}

/* -------------------------------------------------------------- the page */
export default function CalibrationNew() {
  const { stage = 'lock' } = useParams()
  const nav = useNavigate()
  const [progress, setProgress] = useState(null)   // null until the server answers
  const [runDir, setRunDir] = useState(null)

  // Ask the SERVER what has happened rather than trusting component state, so a
  // refresh mid-capture lands back on Capture with the session still running.
  useEffect(() => {
    api.sessionInfo()
      .then((s) => { setProgress(s); if (s.run_dir) setRunDir(s.run_dir) })
      .catch(() => setProgress({ active: false, saved: 0, locked: false }))
  }, [])

  if (progress === null) {
    return <Page title="New calibration" back="/calibration">
      <p className="text-sm text-muted-foreground text-center">checking camera…</p>
    </Page>
  }

  const reached = (id) => {
    if (id === 'lock') return true
    if (id === 'capture') return progress.locked || progress.active
    if (id === 'solve') return progress.active && progress.saved >= MIN_FRAMES
    return false
  }
  // Guard rather than hide: a stage you have not earned redirects to the
  // earliest one you have, so a typed URL cannot skip ahead.
  if (!reached(stage)) {
    const earliest = STAGES.filter((s) => reached(s.id)).pop() || STAGES[0]
    return <Navigate to={`/calibration/new/${earliest.id}`} replace />
  }

  const go = (s) => nav(`/calibration/new/${s}`)
  const fill = stage !== 'solve'

  return (
    <Page title="New calibration" back="/calibration" fill={fill} wide>
      <div className="pb-4 mb-4 border-b border-border shrink-0">
        <Stepper current={stage} reached={reached} />
      </div>
      {stage === 'lock' && (
        <PreviewLock endpoints={CALIB_PREVIEW} onDone={() => {
          setProgress((p) => ({ ...p, locked: true })); go('capture')
        }} />
      )}
      {stage === 'capture' && (
        <CaptureStage
          onProgress={(saved) => setProgress((p) => ({ ...p, active: true, saved }))}
          onSolve={(d) => { setRunDir(d); go('solve') }} />
      )}
      {stage === 'solve' && <SolveStage runDir={runDir} />}
    </Page>
  )
}
