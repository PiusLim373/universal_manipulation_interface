import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Loader2, Lock, Scissors, Play, Pause, ChevronLeft, ChevronRight, ChevronsLeft,
  ChevronsRight, ChevronDown, AlertTriangle, RotateCcw, X, Clock, Ban,
} from 'lucide-react'
import TrimBar from '@/components/TrimBar'
import { STATUS_LEGEND } from '@/lib/editStatus'
import TcpGraph from '@/components/TcpGraph'
import GripperGraph from '@/components/GripperGraph'
import useHotkeys from '@/hooks/useHotkeys'
import { Button } from '@/components/ui/button'
import { api, sse } from '@/lib/api'
import { cn } from '@/lib/utils'

const SPEEDS = [0.25, 0.5, 1]
const EMPTY = { excluded: false, trim: null }
const f2 = (v) => (v == null ? '—' : v.toFixed(2))
const ago = (ms) => {
  const s = Math.round((Date.now() - ms) / 1000)
  return s < 5 ? 'just now' : s < 60 ? `${s}s ago` : new Date(ms).toLocaleTimeString()
}

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

/* ------------------------------------------------------------ episode list */
const EpisodeRow = memo(function EpisodeRow({ e, edit, prep, selected, onPick }) {
  const ref = useRef(null)
  useEffect(() => { if (selected) ref.current?.scrollIntoView({ block: 'nearest' }) }, [selected])
  const trimmed = !!edit.trim
  const drop = prep?.status === 'done' && !trimmed ? prep.meta?.untrimmed?.reason : null
  return (
    <button ref={ref} type="button" onClick={() => onPick(e.key)}
            className={cn(
              'w-full text-left px-3 py-2 border-b border-border/50 transition-colors',
              selected ? 'bg-primary/10 border-l-2 border-l-primary' : 'hover:bg-muted/40',
            )}>
      <div className="flex items-center gap-2">
        <span className={cn('text-sm font-medium tabular-nums',
          (edit.excluded || e.locked) && 'line-through text-muted-foreground')}>{e.ep}</span>
        {prep?.meta?.duration_s != null && (
          <span className="text-[11px] text-muted-foreground tabular-nums">
            {prep.meta.duration_s.toFixed(1)} s
          </span>
        )}
        <span className="flex-1" />
        {trimmed && <Scissors className="size-3.5 text-primary" />}
        {e.locked ? <Lock className="size-3.5 text-muted-foreground" />
          : edit.excluded ? <Ban className="size-3.5 text-muted-foreground" />
          : prep?.status === 'running'
            ? <span className="flex items-center gap-1 text-[10px] text-muted-foreground tabular-nums">
                <Loader2 className="size-3 animate-spin" />{prep.pct}%
              </span>
          : prep?.status === 'queued' ? <Clock className="size-3.5 text-muted-foreground" />
          : prep?.status === 'failed' ? <AlertTriangle className="size-3.5 text-destructive" />
          : null}
      </div>
      <div className="text-[11px] leading-snug mt-0.5 truncate">
        {e.locked ? <span className="text-destructive">verify: {e.reasons[0] || 'failed'}</span>
          : edit.excluded ? <span className="text-muted-foreground">excluded</span>
          : trimmed ? <span className="text-primary">trim {f2(edit.trim[0])} – {f2(edit.trim[1])} s</span>
          : drop ? <span className="text-[#f59e0b]">export drops it: {drop}</span>
          : <span className="text-muted-foreground/60">
              {prep?.status === 'done' ? 'full length' : prep?.status || '—'}
            </span>}
      </div>
    </button>
  )
})

/* ------------------------------------------------------ not-ready placeholder */
function Waiting({ ep, prep, paused, err, onRetry }) {
  let body
  if (err) {
    body = <><AlertTriangle className="size-6 text-destructive" />
      <p className="text-sm text-destructive max-w-md text-center break-words">{err}</p>
      <Button size="sm" variant="outline" onClick={onRetry}><RotateCcw className="size-4" /> Retry</Button></>
  } else if (prep?.status === 'failed') {
    body = <><AlertTriangle className="size-6 text-destructive" />
      <p className="text-sm">Preparing {ep.key} failed</p>
      <p className="text-xs text-muted-foreground max-w-md text-center break-words">{prep.err}</p>
      <Button size="sm" variant="outline" onClick={onRetry}><RotateCcw className="size-4" /> Retry</Button></>
  } else if (paused && prep?.status !== 'running') {
    body = <><Clock className="size-6 text-muted-foreground" />
      <p className="text-sm text-muted-foreground">
        Preparation paused while {paused === 'recording' ? 'a capture is recording' : 'an export runs'}
      </p></>
  } else {
    const running = prep?.status === 'running'
    body = <><Loader2 className="size-6 animate-spin text-primary" />
      <p className="text-sm">{running ? 'Preparing this episode' : 'Queued for preparation'}</p>
      {running && <>
        <div className="w-64 h-1.5 rounded-full bg-muted overflow-hidden">
          <div className="h-full bg-primary transition-all" style={{ width: `${prep.pct}%` }} />
        </div>
        <p className="text-xs text-muted-foreground tabular-nums">{prep.stage} · {prep.pct}%</p>
      </>}
      <p className="text-xs text-muted-foreground max-w-sm text-center">
        ChArUco detection, pose track and preview videos, saved beside the recording so
        they are only built once.
      </p></>
  }
  return <div className="flex-1 flex flex-col items-center justify-center gap-3 bg-black/40 rounded-md">{body}</div>
}

/* ------------------------------------------------------------------ trimmer */
export default function EpisodeTrimmer({ project, onContinue, onEdits, onUnbound }) {
  const pid = project.id
  const episodes = useMemo(() => project.sessions.flatMap((sess) => (
    Object.entries(project.verify?.[sess]?.episodes || {}).sort(([a], [b]) => a.localeCompare(b))
      .map(([ep, r]) => ({ key: `${sess}/${ep}`, sess, ep, locked: !r.ok, reasons: r.fails || [] }))
  )), [project])
  const open = useMemo(() => episodes.filter((e) => !e.locked), [episodes])

  const [edits, setEdits] = useState(() => project.episodes || {})
  const [current, setCurrent] = useState(() => (open[0] || episodes[0])?.key)
  const [prep, setPrep] = useState({ paused: null, episodes: {} })
  const [prepErr, setPrepErr] = useState(null)
  const [tl, setTl] = useState(null)
  const [tlErr, setTlErr] = useState(null)
  const [plan, setPlan] = useState(null)
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [rate, setRate] = useState(1)
  const [save, setSave] = useState({ state: 'idle', at: null, err: null })
  const [collapsed, setCollapsed] = useState(() => new Set())
  const toggleSession = (sess) => setCollapsed((c) => {
    const n = new Set(c)
    if (n.has(sess)) n.delete(sess)
    else n.add(sess)
    return n
  })
  const [, tick] = useState(0)
  const sceneRef = useRef(null)
  const wristRef = useRef(null)

  const ep = episodes.find((e) => e.key === current)
  const cur = prep.episodes[current]
  const ready = cur?.status === 'done'
  const prepAt = cur?.meta?.prepared_at
  const edit = edits[current] || EMPTY
  const trim = edit.trim
  const trimKey = JSON.stringify(trim)

  /* -------- prep state + focus */
  const onUnboundRef = useRef(onUnbound)
  onUnboundRef.current = onUnbound
  useEffect(() => sse(`/api/edit/projects/${pid}/prep/state`, (m) => {
    setPrep(m)
    if (m.intrinsics && m.intrinsics !== 'ok') onUnboundRef.current?.()
  }), [pid])
  const focus = useCallback(() => {
    if (!current) return
    api.prep(pid, current).then(() => setPrepErr(null)).catch((e) => setPrepErr(e.message))
  }, [pid, current])
  useEffect(focus, [focus])

  /* -------- timeline of the current episode */
  const loadTl = useCallback(() => {
    setTl(null); setTlErr(null); setPlan(null); setTime(0); setPlaying(false)
    if (!current || !ready) return undefined
    let dead = false
    api.timeline(current).then((d) => { if (!dead) setTl(d) })
      .catch((e) => { if (!dead) setTlErr(e.message) })
    return () => { dead = true }
  }, [current, ready, prepAt]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(loadTl, [loadTl])

  /* -------- live export preview for the trim */
  useEffect(() => {
    if (!tl) return undefined
    let dead = false
    const id = setTimeout(() => {
      api.plan(current, JSON.parse(trimKey)).then((p) => { if (!dead) setPlan(p) })
        .catch((e) => { if (!dead) setPlan({ error: e.message }) })
    }, 150)
    return () => { dead = true; clearTimeout(id) }
  }, [tl, current, trimKey])

  /* -------- edits + debounced autosave */
  const pending = useRef({})
  const timer = useRef(null)
  const flush = useCallback(async () => {
    clearTimeout(timer.current)
    const patch = pending.current
    pending.current = {}
    if (!Object.keys(patch).length) return true
    setSave((s) => ({ ...s, state: 'saving' }))
    try {
      await api.saveProject(pid, { episodes: patch })
      setSave({ state: 'saved', at: Date.now(), err: null })
      return true
    } catch (e) {
      pending.current = { ...patch, ...pending.current }
      setSave({ state: 'error', at: null, err: e.message })
      return false
    }
  }, [pid])
  const update = useCallback((key, fn) => {
    setEdits((prev) => {
      const next = { ...prev, [key]: fn(prev[key] || EMPTY) }
      pending.current[key] = next[key]
      return next
    })
    clearTimeout(timer.current)
    timer.current = setTimeout(flush, 500)
  }, [flush])
  useEffect(() => () => { flush() }, [flush])
  const onEditsRef = useRef(onEdits)
  onEditsRef.current = onEdits
  useEffect(() => { onEditsRef.current?.(edits) }, [edits])
  useEffect(() => {
    if (save.state !== 'saved') return undefined
    const id = setInterval(() => tick((n) => n + 1), 15000)
    return () => clearInterval(id)
  }, [save.state])

  /* -------- playback */
  const off = tl?.wrist_offset_s ?? 0
  const wdur = tl?.wrist_duration_s ?? 0
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
    if (!v || !tl) return
    t = Math.min(Math.max(t, 0), tl.duration_s)
    v.currentTime = t
    setTime(t)
    syncWrist(t, true, playing)
  }, [tl, playing, syncWrist])

  const play = useCallback(() => {
    const v = sceneRef.current
    if (!v || !tl) return
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
  }, [rate, tl])

  const fi = tl ? frameAt(tl.t, time) : 0
  const step = (d) => {
    if (!tl) return
    if (playing) pause()
    const i = Math.min(Math.max(fi + d, 0), tl.t.length - 1)
    seek(tl.t[i] + 0.0005)
  }

  /* -------- marks */
  const here = () => (tl ? Math.round(tl.t[fi] * 1e4) / 1e4 : 0)
  const markIn = () => {
    if (!tl || ep?.locked) return
    const v = here()
    update(current, (e) => ({ ...e, trim: [v, e.trim?.[1] != null && e.trim[1] > v ? e.trim[1] : null] }))
  }
  const markOut = () => {
    if (!tl || ep?.locked) return
    const v = here()
    update(current, (e) => ({ ...e, trim: [e.trim?.[0] != null && e.trim[0] < v ? e.trim[0] : null, v] }))
  }
  const clearMark = (k) => update(current, (e) => {
    const t = [...(e.trim || [null, null])]
    t[k] = null
    return { ...e, trim: t[0] == null && t[1] == null ? null : t }
  })
  const toggleExclude = () => {
    if (!ep || ep.locked) return
    update(current, (e) => ({ ...e, excluded: !e.excluded }))
  }

  const move = (d) => {
    const i = open.findIndex((e) => e.key === current)
    const nxt = open[Math.min(Math.max((i < 0 ? 0 : i) + d, 0), open.length - 1)]
    if (!nxt) return
    setCurrent(nxt.key)
    setCollapsed((c) => {
      if (!c.has(nxt.sess)) return c
      const n = new Set(c)
      n.delete(nxt.sess)
      return n
    })
  }

  useHotkeys({
    ' ': () => (playing ? pause() : play()),
    ArrowLeft: (e) => step(e.shiftKey ? -10 : -1),
    ArrowRight: (e) => step(e.shiftKey ? 10 : 1),
    '[': markIn,
    ']': markOut,
    x: toggleExclude,
    X: toggleExclude,
    ArrowUp: () => move(-1),
    ArrowDown: () => move(1),
    1: () => setRate(0.25),
    2: () => setRate(0.5),
    3: () => setRate(1),
  }, true, ['ArrowLeft', 'ArrowRight'])

  const included = open.filter((e) => !(edits[e.key] || EMPTY).excluded).length

  return (
    <div className="flex-1 flex min-h-0 gap-4">
      {/* ---------------- left: episodes */}
      <div className="w-64 shrink-0 flex flex-col min-h-0 border border-border rounded-lg overflow-hidden">
        <div className="px-3 py-2.5 border-b border-border text-sm font-medium flex items-center">
          Episodes
          <span className="ml-auto text-xs font-normal text-muted-foreground">
            {included}/{episodes.length} included
          </span>
        </div>
        <div className="flex-1 overflow-y-auto">
          {project.sessions.map((sess) => {
            const eps = episodes.filter((e) => e.sess === sess)
            const inc = eps.filter((e) => !e.locked && !(edits[e.key] || EMPTY).excluded).length
            const shut = collapsed.has(sess)
            return (
              <div key={sess}>
                <button type="button" onClick={() => toggleSession(sess)}
                        className="w-full flex items-center gap-1.5 px-2 py-1.5 text-[11px] font-medium
                                   text-muted-foreground bg-muted/60 hover:bg-muted sticky top-0 z-10">
                  {shut ? <ChevronRight className="size-3.5" /> : <ChevronDown className="size-3.5" />}
                  {sess}
                  <span className="ml-auto font-normal tabular-nums">{inc}/{eps.length}</span>
                </button>
                {!shut && eps.map((e) => (
                  <EpisodeRow key={e.key} e={e} edit={edits[e.key] || EMPTY}
                              prep={prep.episodes[e.key]} selected={e.key === current}
                              onPick={setCurrent} />
                ))}
              </div>
            )
          })}
        </div>
        <div className="p-3 border-t border-border space-y-1.5">
          <Button className="w-full" disabled={included === 0}
                  onClick={async () => { if (await flush()) onContinue() }}>
            Continue to Export
          </Button>
          <p className={cn('text-[11px] text-center',
            save.state === 'error' ? 'text-destructive' : 'text-muted-foreground')}>
            {save.state === 'saving' ? 'saving…'
              : save.state === 'error' ? `not saved: ${save.err}`
              : save.state === 'saved' ? `saved ${ago(save.at)}`
              : `autosaved to data/dataset/${pid}_dataset.json`}
          </p>
        </div>
      </div>

      {/* ---------------- right: player */}
      <div className="flex-1 min-w-0 flex flex-col min-h-0 gap-3">
        <div className="flex items-center gap-2 text-sm">
          <span className="font-medium">{current || '—'}</span>
          {ep?.locked && (
            <span className="text-xs text-destructive flex items-center gap-1">
              <Lock className="size-3.5" /> verify failed, cannot be included: {ep.reasons.join('; ')}
            </span>
          )}
          {!ep?.locked && edit.excluded && (
            <span className="text-xs text-muted-foreground">excluded from the dataset</span>
          )}
          {prepErr && <span className="text-xs text-destructive">prep: {prepErr}</span>}
          <span className="flex-1" />
          {ep && !ep.locked && (
            <Button size="sm" variant={edit.excluded ? 'default' : 'outline'} onClick={toggleExclude}
                    title="Toggle (X)">
              {edit.excluded ? 'Include episode' : 'Exclude episode'}
            </Button>
          )}
        </div>

        {!ready || !tl ? (
          <Waiting ep={ep || {}} prep={cur} paused={prep.paused} err={tlErr}
                   onRetry={() => { setTlErr(null); if (ready) loadTl(); else focus() }} />
        ) : (
          <>
            <div className="flex-1 min-h-0 grid grid-cols-[3fr_2fr] grid-rows-[2fr_1fr] gap-2"
                 onWheel={(e) => step(e.deltaY > 0 ? 1 : -1)}>
              <div className="relative min-h-0 bg-black rounded-md overflow-hidden">
                <span className="absolute top-1.5 right-2 z-10 text-[10px] text-white/70 bg-black/50 px-1.5 rounded">
                  scene · annotated
                </span>
                <video key={`s-${current}`} ref={sceneRef} muted playsInline preload="auto"
                       src={api.media(current, 'scene_annotated.mp4', prepAt)}
                       className="w-full h-full object-contain"
                       onEnded={() => setPlaying(false)} />
              </div>
              <div className="relative min-h-0 bg-black rounded-md overflow-hidden">
                <span className="absolute top-1.5 right-2 z-10 text-[10px] text-white/70 bg-black/50 px-1.5 rounded">
                  wrist
                </span>
                <video key={`w-${current}`} ref={wristRef} muted playsInline preload="auto"
                       src={api.media(current, 'wrist.mp4', prepAt)}
                       className="w-full h-full object-contain" />
              </div>
              <TcpGraph tcp={tl.tcp} duration={tl.duration_s} time={time} trim={trim} />
              <GripperGraph gripper={tl.gripper} duration={tl.duration_s} time={time} trim={trim} />
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
                        {!trim && ' (trim off the bad stretch to keep the rest)'}
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

              <span className="flex-1" />

              <div className="flex items-center gap-1">
                <Button size="sm" variant="outline" onClick={markIn} disabled={ep?.locked}
                        className="text-[#22c55e]" title="Trim start at this frame ([)">[ In</Button>
                {trim?.[0] != null && <>
                  <button type="button" className="text-xs font-mono tabular-nums hover:underline"
                          onClick={() => seek(trim[0] + 0.0005)}>{f2(trim[0])}</button>
                  <button type="button" onClick={() => clearMark(0)} title="Clear"
                          className="text-muted-foreground hover:text-destructive"><X className="size-3.5" /></button>
                </>}
              </div>
              <div className="flex items-center gap-1">
                <Button size="sm" variant="outline" onClick={markOut} disabled={ep?.locked}
                        className="text-[#ef4444]" title="Trim end at this frame (])">Out ]</Button>
                {trim?.[1] != null && <>
                  <button type="button" className="text-xs font-mono tabular-nums hover:underline"
                          onClick={() => seek(trim[1] - 0.0005)}>{f2(trim[1])}</button>
                  <button type="button" onClick={() => clearMark(1)} title="Clear"
                          className="text-muted-foreground hover:text-destructive"><X className="size-3.5" /></button>
                </>}
              </div>
            </div>
            <p className="text-[11px] text-muted-foreground">
              Space play · ←/→ frame (Shift ×10, or scroll) · [ / ] trim in/out · X exclude ·
              ↑/↓ episode · 1/2/3 speed
            </p>
          </>
        )}
      </div>
    </div>
  )
}
