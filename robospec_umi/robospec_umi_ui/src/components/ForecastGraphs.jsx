import { useMemo } from 'react'
import LinePlot from '@/components/LinePlot'

const COLORS = ['#ef4444', '#22c55e', '#3b82f6']
const GRIP = '#f59e0b'

// Largest i with t[i] <= v, or -1.
function before(t, v) {
  let lo = 0, hi = t.length - 1, out = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (t[mid] <= v + 1e-6) { out = mid; lo = mid + 1 } else hi = mid - 1
  }
  return out
}

const series = (rows, names) => names.map((name, k) => ({
  name, color: COLORS[k], values: rows.map((r) => (r ? r[k] : null)),
}))

/**
 * The recorded TCP and gripper (solid) with the model's forecasts over them
 * (dashed): every forecast starts on the recorded line, and how far it peels off
 * is the error. The forecast at the playhead is drawn bold.
 *
 * r: the eval result (arrays f_t, f_xyz, f_rpy, f_grip; step_s). tl: the timeline.
 */
export default function ForecastGraphs({ tl, r, time, trim }) {
  const { f_t: ft, f_xyz: xyz, f_rpy: rpy, f_grip: grip } = r.arrays
  const at = (f, k) => ft[f] + k * r.step_s

  const stubs = useMemo(() => {
    const tt = tl.tcp.t
    // rpy is unwrapped on the recorded line; shift each forecast onto the same turn
    const turn = (f) => {
      const rec = tl.tcp.rpy[Math.max(before(tt, ft[f]), 0)]
      return [0, 1, 2].map((a) => (rec && rpy[f][0][a] != null
        ? 360 * Math.round((rec[a] - rpy[f][0][a]) / 360) : 0))
    }
    const pos = [], rot = [], gr = []
    ft.forEach((_, f) => {
      const sh = turn(f)
      for (let a = 0; a < 3; a++) {
        pos.push({ color: COLORS[a], pts: xyz[f].map((p, k) => [at(f, k), p[a]]) })
        rot.push({ color: COLORS[a], pts: rpy[f].map((p, k) => [at(f, k), p[a] == null ? null : p[a] + sh[a]]) })
      }
      gr.push({ color: GRIP, pts: grip[f].map((g, k) => [at(f, k), g]) })
    })
    return { pos, rot, gr }
  }, [tl, r]) // eslint-disable-line react-hooks/exhaustive-deps

  // the forecast made at or before the playhead, while it is still running
  const f = before(ft, time)
  const live = f >= 0 && time - ft[f] < r.step_s * r.horizon
  const pick = (list, n) => (live ? list.slice(f * n, f * n + n) : null)

  const recorded = useMemo(() => ({
    pos: series(tl.tcp.xyz, ['x', 'y', 'z']),
    rot: series(tl.tcp.rpy, ['r', 'p', 'y']),
    gr: tl.gripper ? [{ name: 'width', color: GRIP, values: tl.gripper.width }] : [],
  }), [tl])

  const common = { duration: tl.duration_s, time, trim }
  return (
    <div className="min-h-0 grid grid-cols-3 gap-2">
      <LinePlot label="TCP position" unit="mm" t={tl.tcp.t} series={recorded.pos} minSpan={10}
                stubs={stubs.pos} highlight={pick(stubs.pos, 3)} {...common} />
      <LinePlot label="TCP rotation" unit="deg" t={tl.tcp.t} series={recorded.rot} minSpan={5}
                stubs={stubs.rot} highlight={pick(stubs.rot, 3)} {...common} />
      {tl.gripper
        ? <LinePlot label="gripper" unit="mm" t={tl.gripper.t} series={recorded.gr} range={[0, 120]}
                    stubs={stubs.gr} highlight={pick(stubs.gr, 1)} {...common} />
        : <div className="min-h-0 flex items-center justify-center bg-black rounded-md text-xs text-white/50">
            no gripper recording
          </div>}
    </div>
  )
}

/** The playhead's forecast (dashed) and what the TCP really did next (solid),
 *  projected onto the scene image. Same viewBox scaling as object-contain. */
export function ForecastOverlay({ r, time }) {
  const { f_t: ft, f_i: fi, f_seg: fs, f_px: fpx, g_seg: gs, g_px: gpx } = r.arrays
  const offset = useMemo(() => {
    const o = {}
    gs.forEach((s, k) => { if (o[s] === undefined) o[s] = k })
    return o
  }, [gs])
  const f = before(ft, time)
  if (f < 0 || time - ft[f] >= r.step_s * r.horizon) return null
  const [w, h] = r.scene_size
  const recorded = []
  for (let k = 0; k < r.horizon; k++) {
    const g = offset[fs[f]] + fi[f] + k * r.down_sample
    if (g >= gs.length || gs[g] !== fs[f]) break
    recorded.push(gpx[g])
  }
  const line = (pts) => pts.filter((p) => p && p[0] != null).map((p) => p.join(',')).join(' ')
  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none"
         viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet">
      <polyline points={line(recorded)} fill="none" stroke="#ffffff" strokeWidth={3}
                vectorEffect="non-scaling-stroke" />
      <polyline points={line(fpx[f])} fill="none" stroke="#f59e0b" strokeWidth={3}
                strokeDasharray="8 5" vectorEffect="non-scaling-stroke" />
      {fpx[f][0]?.[0] != null && <circle cx={fpx[f][0][0]} cy={fpx[f][0][1]} r={6} fill="#f59e0b" />}
    </svg>
  )
}
