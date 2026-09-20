import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import Page from '@/components/Page'
import CameraStream from '@/components/CameraStream'
import CameraBusy from '@/components/CameraBusy'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { api, sse, ApiError } from '@/lib/api'

/**
 * Validate a calibration against the live camera.
 *
 * Standalone rather than the last step of the wizard: checking a calibration is
 * something you want to do whenever you suspect the camera has drifted, not
 * only in the minute after solving one.
 *
 * ?run=<name> tests that run's archived intrinsics; with no query it tests the
 * ACTIVE calibration, which is what everything downstream actually uses.
 */
export default function CalibrationTest() {
  const [params] = useSearchParams()
  const run = params.get('run')
  const [state, setState] = useState(null)
  const [busyOwner, setBusyOwner] = useState(null)
  const [err, setErr] = useState(null)
  const [live, setLive] = useState(false)
  const unsubRef = useRef(null)

  const start = useCallback(async () => {
    setErr(null); setBusyOwner(null)
    try {
      await api.startTest(run ? { run } : {})
      setLive(true)
      unsubRef.current = sse('/api/calibration/test/state', setState)
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && e.owner) setBusyOwner(e.owner)
      else setErr(e.message)
    }
  }, [run])

  useEffect(() => {
    start()
    return () => { unsubRef.current?.(); api.stopTest().catch(() => {}) }
  }, [start])

  const title = run ? `Test calibration · ${run}` : 'Test active calibration'

  if (busyOwner) {
    return <Page title={title} back="/calibration">
      <CameraBusy owner={busyOwner} onReleased={start} />
    </Page>
  }

  return (
    <Page title={title} back="/calibration" fill wide>
      {err && <p className="text-sm text-destructive mb-3">{err}</p>}
      <div className="flex-1 min-h-0 grid lg:grid-cols-[1fr_20rem] gap-5">
        <div className="flex flex-col gap-3 min-h-0">
          {live && <CameraStream src="/api/calibration/test/preview" className="flex-1 min-h-0" />}
          <p className="text-[11px] text-muted-foreground">
            Walk the board into all four frame corners. Error by radius is the
            check the calibration set could not perform on itself — distortion
            grows outward, and a fit made from central views is extrapolating
            at the edges.
          </p>
        </div>

        <div className="flex flex-col justify-end gap-4 min-h-0 overflow-auto">
          <Card><CardContent className="pt-6 space-y-3">
            <div className="flex flex-wrap gap-1.5">
              <Badge variant={state?.pose ? 'default' : 'outline'}>
                {state?.status ?? 'starting'}
              </Badge>
              {state?.geometry_ok === false && (
                <Badge variant="destructive">focus/zoom drifted</Badge>
              )}
              {state?.ambiguous && <Badge variant="destructive">ambiguous pose</Badge>}
            </div>
            <dl className="grid grid-cols-2 gap-y-2 text-sm">
              <div><dt className="text-xs text-muted-foreground">corners</dt>
                   <dd className="tabular-nums">{state?.corners ?? 0}</dd></div>
              <div><dt className="text-xs text-muted-foreground">reprojection</dt>
                   <dd className="tabular-nums">{state?.reproj_px ?? '—'} px</dd></div>
              <div><dt className="text-xs text-muted-foreground">distance</dt>
                   <dd className="tabular-nums">{state?.distance_mm ?? '—'} mm</dd></div>
              <div><dt className="text-xs text-muted-foreground">2nd branch</dt>
                   <dd className="tabular-nums">{state?.ambiguity ?? '—'}×</dd></div>
            </dl>
            {state?.jitter_mm && (
              <p className="text-xs text-muted-foreground">
                jitter x{state.jitter_mm[0]} y{state.jitter_mm[1]} z{state.jitter_mm[2]} mm
              </p>
            )}
          </CardContent></Card>

          <Card><CardContent className="pt-6 space-y-2">
            <span className="text-xs font-medium">reprojection error by radius</span>
            <p className="text-[11px] text-muted-foreground">
              0% optical centre, 100% frame corner.
            </p>
            {(state?.radial || [null, null, null, null]).map((v, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-16 text-[11px] text-muted-foreground tabular-nums">
                  {Math.round(100 * i / 4)}–{Math.round(100 * (i + 1) / 4)}%
                </span>
                <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                  <div className="h-full rounded-full bg-primary"
                       style={{ width: `${Math.min((v ?? 0) / 2, 1) * 100}%` }} />
                </div>
                <span className="w-12 text-right text-[11px] tabular-nums">
                  {v === null || v === undefined ? '—' : v.toFixed(2)}
                </span>
              </div>
            ))}
          </CardContent></Card>
        </div>
      </div>
    </Page>
  )
}
