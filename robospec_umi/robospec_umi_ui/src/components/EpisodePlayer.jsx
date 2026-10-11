import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Loader2, Play, Pause, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight,
  AlertTriangle, RotateCcw, X, Clock,
} from 'lucide-react'
import TrimBar from '@/components/TrimBar'
import { STATUS_LEGEND } from '@/lib/editStatus'
import TcpGraph from '@/components/TcpGraph'
import GripperGraph from '@/components/GripperGraph'
import ImuGraph from '@/components/ImuGraph'
import useHotkeys from '@/hooks/useHotkeys'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

const SPEEDS = [0.25, 0.5, 1]
const f2 = (v) => (v == null ? '—' : v.toFixed(2))

// Largest i with t[i] <= v.
function frameAt(t, v) {
  let lo = 0, hi = t.length - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (t[mid] <= v + 1e-4) lo = mid
    else hi = mid - 1
  }
  return lo
}

/* ------------------------------------------------------ not-ready placeholder */
export function Waiting({ label, prep, paused, err, onRetry,
  note = 'ChArUco detection, pose track and preview videos, saved beside the recording so they are only built once.' }) {
  let body
  if (err) {
    body = <><AlertTriangle className="size-6 text-destructive" />
      <p className="text-sm text-destructive max-w-md text-center break-words">{err}</p>
      {onRetry && <Button size="sm" variant="outline" onClick={onRetry}><RotateCcw className="size-4" /> Retry</Button>}</>
  } else if (prep?.status === 'failed') {
    body = <><AlertTriangle className="size-6 text-destructive" />
      <p className="text-sm">Preparing {label} failed</p>
      <p className="text-xs text-muted-foreground max-w-md text-center break-words">{prep.err}</p>
      {onRetry && <Button size="sm" variant="outline" onClick={onRetry}><RotateCcw className="size-4" /> Retry</Button>}</>
  } else if (paused && prep?.status !== 'running') {
    body = <><Clock className="size-6 text-muted-foreground" />
      <p className="text-sm text-muted-foreground">
        Preparation paused while {paused === 'recording' ? 'a capture is recording' : 'an export runs'}
      </p></>
  } else {
    const running = prep?.status === 'running'
    body = <><Loader2 className="size-6 animate-spin text-primary" />
      <p className="text-sm">{running ? `Preparing ${label}` : 'Queued for preparation'}</p>
      {running && <>
        <div className="w-64 h-1.5 rounded-full bg-muted overflow-hidden">
          <div className="h-full bg-primary transition-all" style={{ width: `${prep.pct}%` }} />
        </div>
        <p className="text-xs text-muted-foreground tabular-nums">{prep.stage} · {prep.pct}%</p>
      </>}
      <p className="text-xs text-muted-foreground max-w-sm text-center">{note}</p></>
  }
  return <div className="flex-1 flex flex-col items-center justify-center gap-3 bg-black/40 rounded-md">{body}</div>
}

/* ------------------------------------------------------------------- player
   Annotated scene + wrist in sync, the graphs, the status bar with what export
   keeps, and the transport. With onTrim it also marks the trim. Remount it (key)
   per episode: playback state is its own, bar the speed when a parent holds it. */
export default function EpisodePlayer({ tl, plan, src, trim = null, onTrim, locked = false,
                                        rate: heldRate, onRate }) {
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [ownRate, setOwnRate] = useState(1)
  const rate = heldRate ?? ownRate
  const setRate = onRate ?? setOwnRate
  const sceneRef = useRef(null)
  const wristRef = useRef(null)

  const off = tl.wrist_offset_s ?? 0
  const wdur = tl.wrist_duration_s ?? 0
  const syncWrist = useCallback((t, hard, play) => {
    const w = wristRef.current
    if (!w) return
    const target = Math.min(Math.max(t - off, 0), wdur)
    if (hard || Math.abs(w.currentTime - target) > 0.05) w.currentTime = target
    const inside = t >= off && t - off < wdur
    if (play && inside && w.paused) w.play().catch(() => {})
    if ((!play || !inside) && !w.paused) w.pause()
  }, [off, wdur])

  const pause = useCallback(() => {
    const v = sceneRef.current
    v?.pause()
    setPlaying(false)
    if (v) { setTime(v.currentTime); syncWrist(v.currentTime, true, false) }
  }, [syncWrist])

  const seek = useCallback((t) => {
    const v = sceneRef.current
    if (!v) return
    t = Math.min(Math.max(t, 0), tl.duration_s)
    v.currentTime = t
    setTime(t)
    syncWrist(t, true, playing)
  }, [tl, playing, syncWrist])

  const play = useCallback(() => {
    const v = sceneRef.current
    if (!v) return
    let t = v.currentTime
    if (trim && ((trim[0] != null && t < trim[0]) || (trim[1] != null && t >= trim[1] - 0.01))) {
      t = trim[0] ?? 0
      v.currentTime = t
    } else if (t >= tl.duration_s - 0.01) {
      t = 0
      v.currentTime = 0
    }
    v.play().catch(() => {})
    setPlaying(true)
    syncWrist(t, true, true)
  }, [tl, trim, syncWrist])

  useEffect(() => {
    if (!playing) return undefined
    let id
    const loop = () => {
      const v = sceneRef.current
      if (!v) return
      const t = v.currentTime
      if (trim?.[1] != null && t >= trim[1]) {
        v.pause()
        v.currentTime = trim[1]
        setPlaying(false)
        setTime(trim[1])
        syncWrist(trim[1], true, false)
        return
      }
      setTime(t)
      syncWrist(t, false, true)
      id = requestAnimationFrame(loop)
    }
    id = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(id)
  }, [playing, trim, syncWrist])

  useEffect(() => {
    for (const v of [sceneRef.current, wristRef.current]) if (v) v.playbackRate = rate
  }, [rate])

  // release the media connections on unmount, as CameraStream does its stream
  useEffect(() => {
    const vids = [sceneRef.current, wristRef.current]
    return () => vids.forEach((v) => { if (v) { v.removeAttribute('src'); v.load() } })
  }, [])

  const fi = frameAt(tl.t, time)
  const step = (d) => {
    if (playing) pause()
    const i = Math.min(Math.max(fi + d, 0), tl.t.length - 1)
    seek(tl.t[i] + 0.0005)
  }

  /* -------- marks */
  const canTrim = !!onTrim && !locked
  const here = () => Math.round(tl.t[fi] * 1e4) / 1e4
  const markIn = () => {
    if (!canTrim) return
    const v = here()
    onTrim([v, trim?.[1] != null && trim[1] > v ? trim[1] : null])
  }
  const markOut = () => {
    if (!canTrim) return
    const v = here()
    onTrim([trim?.[0] != null && trim[0] < v ? trim[0] : null, v])
  }
  const clearMark = (k) => {
    const t = [...(trim || [null, null])]
    t[k] = null
    onTrim(t[0] == null && t[1] == null ? null : t)
  }

  useHotkeys({
    ' ': () => (playing ? pause() : play()),
    ArrowLeft: (e) => step(e.shiftKey ? -10 : -1),
    ArrowRight: (e) => step(e.shiftKey ? 10 : 1),
    ...(onTrim && { '[': markIn, ']': markOut }),
    1: () => setRate(0.25),
    2: () => setRate(0.5),
    3: () => setRate(1),
  }, true, ['ArrowLeft', 'ArrowRight'])

  return (
    <>
      <div className="flex-1 min-h-0 grid grid-cols-[3fr_2fr] grid-rows-[2fr_1fr] gap-2"
           onWheel={(e) => step(e.deltaY > 0 ? 1 : -1)}>
        <div className="relative min-h-0 bg-black rounded-md overflow-hidden">
          <span className="absolute top-1.5 right-2 z-10 text-[10px] text-white/70 bg-black/50 px-1.5 rounded">
            scene · annotated
          </span>
          <video ref={sceneRef} muted playsInline preload="auto"
                 src={src('scene_annotated.mp4')}
                 className="w-full h-full object-contain"
                 onEnded={() => setPlaying(false)} />
        </div>
        <div className="relative min-h-0 bg-black rounded-md overflow-hidden">
          <span className="absolute top-1.5 right-2 z-10 text-[10px] text-white/70 bg-black/50 px-1.5 rounded">
            wrist
          </span>
          <video ref={wristRef} muted playsInline preload="auto"
                 src={src('wrist.mp4')}
                 className="w-full h-full object-contain" />
          {/* the training crop; 'meet' scales it exactly like object-contain */}
          {tl.wrist_crop && (
            <svg className="absolute inset-0 w-full h-full pointer-events-none"
                 viewBox={`0 0 ${tl.wrist_crop.w} ${tl.wrist_crop.h}`}
                 preserveAspectRatio="xMidYMid meet">
              <rect x={tl.wrist_crop.x0} y={0} width={tl.wrist_crop.size} height={tl.wrist_crop.h}
                    fill="none" stroke="#facc15" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
            </svg>
          )}
        </div>
        <TcpGraph tcp={tl.tcp} duration={tl.duration_s} time={time} trim={trim} />
        <div className="min-h-0 flex flex-col gap-1">
          <GripperGraph className="flex-1" gripper={tl.gripper} duration={tl.duration_s}
                        time={time} trim={trim} />
          {tl.imu && <ImuGraph className="flex-1" imu={tl.imu} duration={tl.duration_s}
                               time={time} trim={trim} />}
        </div>
      </div>

      <div className="space-y-1.5">
        <TrimBar t={tl.t} status={tl.status} duration={tl.duration_s}
                 spans={plan?.spans} trim={trim} time={time} onSeek={seek} />
        <div className="flex items-center gap-3 text-[11px] text-muted-foreground flex-wrap">
          {STATUS_LEGEND.map(([c, label]) => (
            <span key={label} className="flex items-center gap-1">
              <span className="size-2 rounded-sm" style={{ background: c }} />{label}
            </span>
          ))}
          <span className="flex items-center gap-1">
            <span className="w-3 h-2 rounded-sm bg-primary" />exported
          </span>
          <span className="flex-1" />
          {plan?.error ? <span className="text-destructive">{plan.error}</span>
            : !plan ? <span>checking…</span>
            : plan.reason
              ? <span className="text-[#f59e0b] flex items-center gap-1">
                  <AlertTriangle className="size-3.5" />
                  export drops this episode: {plan.reason}
                  {onTrim && !trim && ' (trim off the bad stretch to keep the rest)'}
                </span>
              : <span className="text-foreground">
                  export keeps {plan.kept_s.toFixed(2)} s in {plan.spans.length} segment
                  {plan.spans.length === 1 ? '' : 's'} · usable {(plan.usable * 100).toFixed(0)}%
                </span>}
        </div>
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        <Button size="icon-sm" variant="outline" onClick={() => step(-10)} title="Back 10 frames (Shift+←)">
          <ChevronsLeft className="size-4" /></Button>
        <Button size="icon-sm" variant="outline" onClick={() => step(-1)} title="Previous frame (←)">
          <ChevronLeft className="size-4" /></Button>
        <Button size="icon-sm" onClick={() => (playing ? pause() : play())} title="Play / pause (Space)">
          {playing ? <Pause className="size-4" /> : <Play className="size-4" />}</Button>
        <Button size="icon-sm" variant="outline" onClick={() => step(1)} title="Next frame (→)">
          <ChevronRight className="size-4" /></Button>
        <Button size="icon-sm" variant="outline" onClick={() => step(10)} title="Forward 10 frames (Shift+→)">
          <ChevronsRight className="size-4" /></Button>
        <span className="font-mono text-xs tabular-nums ml-1">
          <span className="text-foreground">{f2(time)}</span>
          <span className="text-muted-foreground"> / {f2(tl.duration_s)} s · frame {fi}/{tl.t.length - 1}</span>
        </span>
        <div className="flex items-center rounded-md border border-border overflow-hidden ml-2">
          {SPEEDS.map((s, i) => (
            <button key={s} type="button" onClick={() => setRate(s)} title={`${s}× (${i + 1})`}
                    className={cn('px-2 h-7 text-xs tabular-nums',
                      rate === s ? 'bg-primary text-primary-foreground' : 'hover:bg-muted')}>
              {s}×
            </button>
          ))}
        </div>

        {onTrim && <>
          <span className="flex-1" />
          <div className="flex items-center gap-1">
            <Button size="sm" variant="outline" onClick={markIn} disabled={locked}
                    className="text-[#22c55e]" title="Trim start at this frame ([)">[ In</Button>
            {trim?.[0] != null && <>
              <button type="button" className="text-xs font-mono tabular-nums hover:underline"
                      onClick={() => seek(trim[0] + 0.0005)}>{f2(trim[0])}</button>
              <button type="button" onClick={() => clearMark(0)} title="Clear"
                      className="text-muted-foreground hover:text-destructive"><X className="size-3.5" /></button>
            </>}
          </div>
          <div className="flex items-center gap-1">
            <Button size="sm" variant="outline" onClick={markOut} disabled={locked}
                    className="text-[#ef4444]" title="Trim end at this frame (])">Out ]</Button>
            {trim?.[1] != null && <>
              <button type="button" className="text-xs font-mono tabular-nums hover:underline"
                      onClick={() => seek(trim[1] - 0.0005)}>{f2(trim[1])}</button>
              <button type="button" onClick={() => clearMark(1)} title="Clear"
                      className="text-muted-foreground hover:text-destructive"><X className="size-3.5" /></button>
            </>}
          </div>
        </>}
      </div>
    </>
  )
}
