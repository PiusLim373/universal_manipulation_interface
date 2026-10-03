import { useCallback, useEffect, useRef, useState } from 'react'
import { Navigate, useNavigate, useParams, Link } from 'react-router-dom'
import { Eye, Video, Check, Circle, Square, Loader2, AlertTriangle, Copy } from 'lucide-react'
import Page from '@/components/Page'
import CameraStream from '@/components/CameraStream'
import GripperReadout from '@/components/GripperReadout'
import CameraBusy from '@/components/CameraBusy'
import PreviewLock from '@/components/PreviewLock'
import Stepper from '@/components/Stepper'
import useHotkeys from '@/hooks/useHotkeys'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { api, sse, ApiError, CAPTURE_PREVIEW } from '@/lib/api'
import { cn } from '@/lib/utils'

const STAGES = [
  { id: 'lock',   label: 'Preview & Lock', icon: Eye },
  { id: 'record', label: 'Capture Dataset', icon: Video },
]


/* ----------------------------------------------- the RECORD / IDLE / WAITING
   WAITING is not decoration. Arming blocks for LEAD (0.2 s) and stopping for
   LEAD + 0.4 s, and without a distinct state for that window the operator
   presses again and the second press lands on a 409. */
// A live session always carries `ready`. Two real payloads do not: the
// inactive-session reply from GET /api/capture/session, and the {"ended":true}
// terminator the SSE writes when the session goes away. Both used to fall into
// WAITING and dereference s.warmup_left, which is not there.
export const isLive = (s) => !!s && s.ready !== undefined

function Status({ s }) {
  if (!s) {
    return <div className="flex items-center gap-2 text-muted-foreground text-sm">
      <Loader2 className="size-4 animate-spin" /> connecting…
    </div>
  }
  if (!isLive(s)) {
    return (
      <div className="flex items-center gap-2 text-muted-foreground">
        <Square className="size-4 shrink-0" />
        <span className="text-lg font-heading">SESSION ENDED</span>
      </div>
    )
  }
  if (!s.ready || s.transition) {
    const ep = String(s.episode_index).padStart(3, '0')
    const what = s.transition === 'arming' ? `arming ep${ep}…`
      : s.transition === 'stopping' ? `saving ep${ep}…`
      : `cameras settling — ${(s.warmup_left ?? 0).toFixed(1)}s`
    return (
      <div className="flex items-center gap-2 text-[#f59e0b]">
        <Loader2 className="size-4 animate-spin shrink-0" />
        <span className="text-lg font-heading">WAITING</span>
        <span className="text-sm text-muted-foreground">{what}</span>
      </div>
    )
  }
  if (s.recording) {
    return (
      <div className="flex items-center gap-2 text-destructive">
        <Circle className="size-4 fill-current shrink-0" />
        <span className="text-lg font-heading">REC</span>
        <span className="text-sm tabular-nums">
          ep{String(s.episode_index).padStart(3, '0')} · {s.elapsed_s.toFixed(1)}s
        </span>
      </div>
    )
  }
  return (
    <div className="flex items-center gap-2 text-muted-foreground">
      <Circle className="size-4 shrink-0" />
      <span className="text-lg font-heading">IDLE</span>
      <span className="text-sm">
        press SPACE to record ep{String(s.episode_index).padStart(3, '0')}
      </span>
    </div>
  )
}

function RecordStage({ onFinished }) {
  const [state, setState] = useState(null)
  const [busyOwner, setBusyOwner] = useState(null)
  const [err, setErr] = useState(null)
  const [pending, setPending] = useState(false)
  const [copied, setCopied] = useState(false)
  const [loadErr, setLoadErr] = useState(null)
  const [reloads, setReloads] = useState(0)
  const unsubRef = useRef(null)
  const epList = useRef(null)
  const nav = useNavigate()

  // Never swallow this. When it failed silently the page sat on "connecting…"
  // for ever with nothing to go on -- the symptom that sent us hunting through
  // the whole stack for a cause the browser already knew.
  useEffect(() => {
    let alive = true
    setLoadErr(null)
    api.captureInfo()
      .then((s) => { if (alive) setState(s) })
      .catch((e) => { if (alive) setLoadErr(e.message) })
    unsubRef.current = sse('/api/capture/state', setState)
    return () => { alive = false; unsubRef.current?.() }
  }, [reloads])

  const toggle = useCallback(async () => {
    if (pending) return
    const s = state
    if (!s || !s.ready || s.transition) return
    setPending(true); setErr(null)
    try {
      const r = await api.episode(s.recording ? 'stop' : 'start')
      setState((prev) => ({ ...prev, ...r }))
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && e.owner) setBusyOwner(e.owner)
      else setErr(e.message)
    } finally { setPending(false) }
  }, [state, pending])

  useHotkeys({ ' ': toggle, Enter: toggle })

  // keep the newest episode in view as the list grows
  const nEps = state?.episodes?.length ?? 0
  useEffect(() => {
    const el = epList.current
    if (el) el.scrollTop = el.scrollHeight
  }, [nEps])

  const finish = async () => {
    setPending(true)
    try { onFinished(await api.finishCapture()) } catch (e) { setErr(e.message) }
    finally { setPending(false) }
  }

  if (busyOwner) return <CameraBusy owner={busyOwner} onReleased={() => nav(0)} />

  if (loadErr) {
    return (
      <Card className="max-w-lg"><CardContent className="pt-6 space-y-3">
        <div className="flex items-start gap-2 text-destructive">
          <AlertTriangle className="size-5 mt-0.5 shrink-0" />
          <div className="space-y-1">
            <p className="font-medium">Could not reach the capture session</p>
            <p className="text-sm text-muted-foreground">{loadErr}</p>
          </div>
        </div>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => setReloads((n) => n + 1)}>Retry</Button>
          <Link to="/capture/new/lock">
            <Button size="sm" variant="outline">Back to Preview &amp; Lock</Button>
          </Link>
        </div>
      </CardContent></Card>
    )
  }

  // The session went away underneath us -- stopped elsewhere, taken over, or the
  // server restarted. Say so and offer the way back instead of rendering a
  // recording UI with nothing behind it.
  if (state && !isLive(state)) {
    return (
      <Card className="max-w-lg"><CardContent className="pt-6 space-y-3">
        <p className="font-medium">The capture session has ended</p>
        <p className="text-sm text-muted-foreground">
          {state.n_episodes
            ? `${state.n_episodes} episode(s) were saved to ${state.session_dir}.`
            : 'Nothing was recorded.'}
        </p>
        <div className="flex gap-2">
          <Link to="/capture/new/lock">
            <Button size="sm">Start a new session</Button>
          </Link>
          {state.n_episodes > 0 && (
            <Link to="/edit"><Button size="sm" variant="outline">Go to Edit</Button></Link>
          )}
        </div>
      </CardContent></Card>
    )
  }

  const s = state
  const eps = s?.episodes || []
  const camErr = Object.entries(s?.cameras || {})
    .filter(([, v]) => v.error).map(([k, v]) => `${k}: ${v.error}`)

  return (
    <div className="flex-1 min-h-0 grid lg:grid-cols-[1fr_22rem] gap-5">
      <div className={cn('flex flex-col gap-3 min-h-0 rounded-lg',
                         s?.recording && 'ring-2 ring-destructive')}>
        <div className="flex-1 min-h-0 grid grid-rows-2 gap-3">
          {(s?.cams || ['scene', 'wrist']).map((cam) => (
            <div key={cam} className="flex flex-col gap-1.5 min-h-0">
              <div className="flex items-center gap-2 shrink-0 text-[11px]">
                <span className="uppercase tracking-wide text-muted-foreground">{cam}</span>
                <span className="tabular-nums text-muted-foreground">
                  {s?.cameras?.[cam]?.fps?.toFixed(1) ?? '—'} fps
                </span>
                {s?.recording && (
                  <Badge variant="destructive">
                    {s.cameras?.[cam]?.written ?? 0} frames
                  </Badge>
                )}
              </div>
              <CameraStream src={`/api/capture/stream/${cam}`} className="flex-1 min-h-0" />
            </div>
          ))}
        </div>
        <GripperReadout g={s?.gripper} recording={s?.recording} />
        {err && <p className="text-sm text-destructive shrink-0">{err}</p>}
        {camErr.map((m) => (
          <p key={m} className="text-sm text-destructive shrink-0 flex items-center gap-1.5">
            <AlertTriangle className="size-4" /> {m}
          </p>
        ))}
      </div>

      <div className="flex flex-col gap-4 min-h-0 overflow-auto">
        <Card className="mt-auto shrink-0"><CardContent className="pt-6 space-y-3">
          <Status s={s} />
          <Button className="w-full" onClick={(e) => { e.currentTarget.blur(); toggle() }}
                  variant={s?.recording ? 'destructive' : 'default'}
                  disabled={!s || !s.ready || !!s.transition || pending}>
            {s?.recording ? <><Square className="size-4" /> Stop episode</>
                          : <><Circle className="size-4" /> Start episode</>}
          </Button>
          <p className="text-[10px] text-muted-foreground text-center">
            SPACE or ENTER — start / stop an episode
          </p>
        </CardContent></Card>

        <Card className="shrink-0"><CardContent className="pt-6 space-y-2">
          <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
            saving to
          </p>
          <div className="flex items-start gap-2">
            <p className="text-[11px] font-mono break-all flex-1">{s?.session_dir}</p>
            <Button size="icon-xs" variant="ghost" title="copy path"
                    onClick={(e) => {
                      e.currentTarget.blur()
                      navigator.clipboard?.writeText(s?.session_dir || '')
                      setCopied(true); setTimeout(() => setCopied(false), 1200)
                    }}>
              {copied ? <Check className="size-3.5 text-[#22c55e]" /> : <Copy className="size-3.5" />}
            </Button>
          </div>
        </CardContent></Card>

        {/* the only card that shrinks: its list scrolls, so Finish stays on screen */}
        <Card className="min-h-0"><CardContent className="pt-6 flex flex-col gap-2 min-h-0">
          <div className="flex items-baseline justify-between shrink-0">
            <span className="text-2xl font-heading tabular-nums">{eps.length}</span>
            <span className="text-xs text-muted-foreground">episodes recorded</span>
          </div>
          {eps.length === 0 && (
            <p className="text-xs text-muted-foreground">nothing recorded yet</p>
          )}
          <div ref={epList} className="min-h-0 overflow-y-auto space-y-2 -mr-2 pr-2">
          {eps.map((e) => (
            <div key={e.index} className="text-[11px] border-t border-border/40 pt-1.5">
              <div className="flex items-baseline justify-between">
                <span className="font-mono">{e.dir}</span>
                <span className="tabular-nums text-muted-foreground">
                  {e.duration_s?.toFixed(1)}s
                </span>
              </div>
              <div className="text-muted-foreground tabular-nums">
                {['scene', 'wrist'].filter((k) => e[k]).map((k) => (
                  <span key={k} className="mr-3">
                    {k} {e[k].frames} @ {e[k].fps.toFixed(1)}
                    {e[k].queue_drops > 0 && (
                      <span className="text-[#f59e0b]"> ·{e[k].queue_drops} dropped</span>
                    )}
                  </span>
                ))}
                {e.gripper && <span>gripper {e.gripper.samples} @ {e.gripper.hz.toFixed(1)}</span>}
              </div>
            </div>
          ))}
          </div>
        </CardContent></Card>

        <Card className="shrink-0"><CardContent className="pt-6 space-y-3">
          <p className="text-xs text-muted-foreground">
            Finishing writes session.json and releases both cameras. Reviewing,
            verifying and compiling the dataset happen in Edit.
          </p>
          <Button className="w-full" onClick={finish}
                  disabled={!s || s.recording || pending || eps.length === 0}>
            Finish &amp; go to Edit
          </Button>
        </CardContent></Card>
      </div>
    </div>
  )
}

export default function CaptureNew() {
  const { stage = 'lock' } = useParams()
  const nav = useNavigate()
  const [progress, setProgress] = useState(null)
  const [bootErr, setBootErr] = useState(null)

  // Ask the SERVER what has happened rather than trusting component state, so a
  // refresh mid-episode lands back on Capture with the session still recording.
  useEffect(() => {
    api.captureInfo().then(setProgress)
      .catch((e) => { setBootErr(e.message); setProgress({ active: false }) })
  }, [])

  if (progress === null) {
    return <Page title="New capture" back="/">
      <p className="text-sm text-muted-foreground text-center">checking cameras…</p>
    </Page>
  }

  // A session EXISTING is what makes the recording stage reachable. Readiness is
  // a within-stage concern that RecordStage already renders as WAITING -- gating
  // on it here bounced you back to Preview & Lock if you locked inside the 3 s
  // warm-up, which looked exactly like the Continue button not working.
  const reached = (id) => id === 'lock' || !!progress.active
  if (!reached(stage)) {
    const earliest = STAGES.filter((s) => reached(s.id)).pop() || STAGES[0]
    return <Navigate to={`/capture/new/${earliest.id}`} replace />
  }

  return (
    <Page title="New capture" back="/" fill wide>
      <div className="pb-4 mb-4 border-b border-border shrink-0">
        <Stepper stages={STAGES} current={stage} reached={reached} />
      </div>
      {bootErr && (
        <p className="text-sm text-destructive shrink-0 mb-3">
          could not read the capture session: {bootErr}
        </p>
      )}
      {stage === 'lock' && (
        // AWAIT the refresh before navigating. Firing it and calling nav() in
        // the same tick means the guard above re-runs against the stale
        // progress, sees active:false, and redirects straight back here.
        <PreviewLock endpoints={CAPTURE_PREVIEW} aside={(st) => <GripperReadout g={st?.gripper} />}
                     onDone={async () => {
          try {
            setProgress(await api.captureInfo())
          } catch (e) {
            // PreviewLock only reaches here after locking a LIVE session, so the
            // session exists -- it is the status call that failed. Carry on to
            // the record stage and let it surface the failure with a Retry.
            // Letting the guard bounce us back instead just looks like Continue
            // not working, which is the bug this whole change is about.
            setBootErr(e.message)
            setProgress((p) => ({ ...p, active: true }))
          }
          nav('/capture/new/record')
        }} />
      )}
      {stage === 'record' && (
        <RecordStage onFinished={() => nav('/edit')} />
      )}
    </Page>
  )
}
