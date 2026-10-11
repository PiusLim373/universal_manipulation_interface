import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Loader2, Lock, Scissors, ChevronRight, ChevronDown, AlertTriangle, Clock, Ban,
} from 'lucide-react'
import EpisodePlayer, { Waiting } from '@/components/EpisodePlayer'
import useHotkeys from '@/hooks/useHotkeys'
import { Button } from '@/components/ui/button'
import { api, sse } from '@/lib/api'
import { cn } from '@/lib/utils'

const EMPTY = { excluded: false, trim: null }
const f2 = (v) => (v == null ? '—' : v.toFixed(2))
const ago = (ms) => {
  const s = Math.round((Date.now() - ms) / 1000)
  return s < 5 ? 'just now' : s < 60 ? `${s}s ago` : new Date(ms).toLocaleTimeString()
}

/* ------------------------------------------------------------ episode list */
// fused: the gyro steadied the pose track. Fell back: there is an IMU, but the
// track is vision only -- the title says why.
function ImuTag({ imu }) {
  if (!imu || (!imu.fused && imu.reason === 'no imu recording')) return null
  return imu.fused
    ? <span className="text-[10px] px-1 rounded border border-[#38bdf8]/50 text-[#38bdf8]"
            title={`IMU-fused track; ${imu.vision_rejected + imu.pose_rejected} vision frames rejected`}>
        imu
      </span>
    : <span className="text-[10px] px-1 rounded border border-[#f59e0b]/50 text-[#f59e0b]"
            title={imu.reason}>
        vision only
      </span>
}

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
        <ImuTag imu={prep?.status === 'done' ? prep.meta?.imu : null} />
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
  const [rate, setRate] = useState(1)       // kept from episode to episode
  const [save, setSave] = useState({ state: 'idle', at: null, err: null })
  const [collapsed, setCollapsed] = useState(() => new Set())
  const toggleSession = (sess) => setCollapsed((c) => {
    const n = new Set(c)
    if (n.has(sess)) n.delete(sess)
    else n.add(sess)
    return n
  })
  const [, tick] = useState(0)

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
    setTl(null); setTlErr(null); setPlan(null)
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

  // playback and trim keys are the player's
  useHotkeys({
    x: toggleExclude,
    X: toggleExclude,
    ArrowUp: () => move(-1),
    ArrowDown: () => move(1),
  })

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
          <Waiting label={current} prep={cur} paused={prep.paused} err={tlErr}
                   onRetry={() => { setTlErr(null); if (ready) loadTl(); else focus() }} />
        ) : (
          <>
            <EpisodePlayer key={`${current}@${prepAt}`} tl={tl} plan={plan}
                           src={(f) => api.media(current, f, prepAt)} trim={trim} locked={ep?.locked}
                           onTrim={(t) => update(current, (e) => ({ ...e, trim: t }))}
                           rate={rate} onRate={setRate} />
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
