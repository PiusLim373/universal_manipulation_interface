import { useCallback, useEffect, useRef, useState } from 'react'
import { Check, X, AlertTriangle } from 'lucide-react'
import CameraStream from '@/components/CameraStream'
import CameraBusy from '@/components/CameraBusy'
import ControlPanel from '@/components/ControlPanel'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { ApiError, sse } from '@/lib/api'
import { cn } from '@/lib/utils'

/**
 * "Preview and Lock" — shared by the calibration wizard and the capture flow.
 *
 * Everything camera-specific arrives in the `endpoints` bundle, because the two
 * flows differ in exactly one way that matters: calibration's tune session is
 * stage-local and must be torn down on the way out, while capture's session
 * SPANS both stages and must not be. That asymmetry is `stopOnUnmount`.
 *
 * `endpoints` and `onDone` are held in refs and never enter a dep array.
 * StrictMode double-invokes the mount effect; with an unstable prop that would
 * mean start -> stop -> start on a session owning two cameras and a 3 s warm-up.
 */
export default function PreviewLock({ endpoints, onDone, aside }) {
  const ep = useRef(endpoints); ep.current = endpoints
  const done = useRef(onDone); done.current = onDone

  const cams = endpoints.cams
  const [sel, setSel] = useState(cams[0])
  const [controls, setControls] = useState({})
  const [state, setState] = useState(null)
  const [mask, setMask] = useState(false)
  const [lockResult, setLockResult] = useState(null)
  const [busyOwner, setBusyOwner] = useState(null)
  const [err, setErr] = useState(null)
  const [live, setLive] = useState(false)
  const [leaving, setLeaving] = useState(false)
  const unsubRef = useRef(null)

  const refreshControls = useCallback(async (cam) => {
    try {
      const c = await ep.current.getControls(cam)
      setControls((prev) => ({ ...prev, [cam]: c }))
    } catch (e) { setErr(e.message) }
  }, [])

  const start = useCallback(async () => {
    setErr(null); setBusyOwner(null)
    try {
      await ep.current.start()
      setLive(true)
      unsubRef.current = sse(ep.current.state, setState)
      await Promise.all(ep.current.cams.map(refreshControls))
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && e.owner) setBusyOwner(e.owner)
      else setErr(e.message)
    }
  }, [refreshControls])

  useEffect(() => {
    start()
    return () => {
      unsubRef.current?.()
      ep.current.stopOnUnmount?.().catch(() => {})
    }
  }, [start])

  const setCtrl = useCallback(async (cam, name, value) => {
    // Optimistic, then reconcile: the driver clamps silently, so the readback
    // is the truth and the slider must snap to it rather than to what we asked.
    setControls((c) => (c[cam]?.[name]
      ? { ...c, [cam]: { ...c[cam], [name]: { ...c[cam][name], value } } } : c))
    await ep.current.setControls(cam, { [name]: value })
    await refreshControls(cam)
  }, [refreshControls])

  const toggleMask = async (on) => { setMask(on); await ep.current.setMask(on) }

  const doLock = async () => {
    setErr(null)
    try { setLockResult(await ep.current.lock()) } catch (e) { setErr(e.message) }
  }

  if (busyOwner) {
    return <CameraBusy owner={busyOwner} onReleased={start} />
  }

  const dyn = lockResult?.dynamic_framerate
  const statsOf = endpoints.statsOf || ((s) => s)

  return (
    <div className="flex-1 min-h-0 grid lg:grid-cols-[1fr_20rem] gap-5">
      <div className="flex flex-col gap-3 min-h-0">
        <div className={cn('flex-1 min-h-0 grid gap-3',
                           cams.length > 1 && 'grid-rows-2')}>
          {cams.map((cam) => {
            const st = statsOf(state, cam)
            return (
              <div key={cam} className="flex flex-col gap-1.5 min-h-0">
                <div className="flex items-center gap-2 shrink-0">
                  <button type="button" onClick={() => setSel(cam)}
                          className={cn('text-[11px] uppercase tracking-wide rounded px-1.5 py-0.5',
                                        sel === cam ? 'bg-primary/15 text-foreground'
                                                    : 'text-muted-foreground')}>
                    {cam}
                  </button>
                  {st && (
                    <>
                      <Badge variant={st.verdict === 'exposure ok' ? 'default' : 'destructive'}>
                        {st.verdict}
                      </Badge>
                      <span className="text-xs text-muted-foreground tabular-nums">
                        mean {st.mean} · clipped {st.clipped}% · crushed {st.crushed}%
                      </span>
                    </>
                  )}
                </div>
                {live && (
                  <CameraStream src={endpoints.stream(cam)}
                                className={cn('flex-1 min-h-0',
                                              cams.length > 1 && sel === cam &&
                                              'ring-1 ring-primary/40 rounded-lg')} />
                )}
              </div>
            )
          })}
        </div>
        {live && aside?.(state)}
        {err && <p className="text-sm text-destructive shrink-0">{err}</p>}
        <div className="flex items-center gap-3 flex-wrap shrink-0">
          <div className="flex-1" />
          <span className="text-[11px] text-muted-foreground">
            focus 0 · zoom 100 · autofocus off — fixed, they define the geometry
          </span>
        </div>
      </div>

      {/* mt-auto on the first child, NOT justify-end on the container: with
          content taller than the column, justify-end pins the overflow above
          the scroll origin where it cannot be reached. */}
      <div className="flex flex-col gap-4 min-h-0 overflow-auto">
        <Card className="mt-auto"><CardContent className="pt-6 space-y-3">
          {cams.length > 1 && (
            <div className="flex rounded-md border border-border p-0.5 text-xs">
              {cams.map((cam) => (
                <button key={cam} type="button"
                        onClick={(e) => { e.currentTarget.blur(); setSel(cam) }}
                        className={cn('flex-1 rounded px-2 py-1 capitalize transition-colors',
                                      sel === cam ? 'bg-primary text-primary-foreground'
                                                  : 'text-muted-foreground hover:text-foreground')}>
                  {cam}
                </button>
              ))}
            </div>
          )}
          <ControlPanel cam={sel} controls={controls[sel]}
                        onSet={(n, v) => setCtrl(sel, n, v)}
                        mean={statsOf(state, sel)?.mean}
                        mask={sel === 'scene' ? mask : undefined}
                        onMask={sel === 'scene' ? toggleMask : undefined} />
        </CardContent></Card>

        <Card><CardContent className="pt-6 space-y-3">
          <p className="text-xs text-muted-foreground">
            Locks the <strong>scene camera&apos;s</strong> geometry — autofocus
            off, focus 0, zoom 100 — and checks the frame rate is not being
            traded away. Exposure, gain, gamma and white balance stay yours.
            {cams.includes('wrist') && ' The wrist camera has fixed lenses with '
              + 'no focus or zoom control, so there is nothing on it to lock.'}
          </p>
          <Button onClick={doLock} className="w-full">Lock scene geometry</Button>

          {lockResult && (
            <div className="space-y-2">
              {lockResult.controls.map((r) => (
                <div key={r.control} className="flex items-center justify-between text-xs">
                  <span className="font-mono">{r.control}</span>
                  <span className="flex items-center gap-1 tabular-nums">
                    {r.got}
                    {r.stuck ? <Check className="size-3.5 text-[#22c55e]" />
                             : <X className="size-3.5 text-destructive" />}
                  </span>
                </div>
              ))}
              <div className={cn('rounded-md px-2.5 py-2 text-xs flex items-start gap-1.5',
                                 dyn?.ok ? 'bg-[#22c55e]/10 text-[#22c55e]'
                                         : 'bg-destructive/10 text-destructive')}>
                {dyn?.ok ? <Check className="size-3.5 mt-0.5 shrink-0" />
                         : <AlertTriangle className="size-3.5 mt-0.5 shrink-0" />}
                <span>
                  exposure_dynamic_framerate = {String(dyn?.value)}
                  {dyn?.ok ? ' — frame rate is protected'
                           : ' — must be 0, or long exposure drops the camera to 78/20/10 fps'}
                </span>
              </div>
              {/* AWAIT onDone: the capture wizard refreshes its session status
                  in there, and its stage guard re-runs the moment we navigate. */}
              <Button className="w-full" disabled={!lockResult.ok || leaving}
                      onClick={async () => {
                        setLeaving(true)
                        try {
                          await ep.current.stopOnDone?.()
                          await done.current?.()
                        } finally { setLeaving(false) }
                      }}>
                {leaving ? 'one moment…' : endpoints.cta}
              </Button>
            </div>
          )}
        </CardContent></Card>
      </div>
    </div>
  )
}
