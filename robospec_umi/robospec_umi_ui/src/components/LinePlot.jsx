import { useMemo } from 'react'

const W = 1000
const H = 300

// Largest i with t[i] <= v, or -1.
function at(t, v) {
  let lo = 0, hi = t.length - 1, out = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (t[mid] <= v + 1e-6) { out = mid; lo = mid + 1 } else hi = mid - 1
  }
  return out
}

const fmt = (v, d) => (v == null ? '—' : v.toFixed(d))

/**
 * Time-series plot on the episode clock: null gaps, trim shading, a playhead,
 * and the value of each series at the playhead.
 *
 * series: [{name, color, values}], values aligned with t (null = no data).
 * range: [lo, hi] to fix the y-axis; otherwise it fits the data, at least minSpan.
 */
export default function LinePlot({ label, unit, t, series, duration, time, trim,
                                   range, minSpan = 1, digits = 1, className = '', right = null }) {
  const { paths, lo, hi } = useMemo(() => {
    let lo = range?.[0], hi = range?.[1]
    if (!range) {
      const all = series.flatMap((s) => s.values.filter((v) => v != null))
      lo = all.length ? Math.min(...all) : 0
      hi = all.length ? Math.max(...all) : minSpan
      const pad = Math.max((hi - lo) * 0.08, (minSpan - (hi - lo)) / 2, 0)
      lo -= pad
      hi += pad
    }
    const px = (v) => (v / duration) * W
    const py = (v) => H - 6 - ((v - lo) / (hi - lo || 1)) * (H - 12)
    const paths = series.map((s) => {
      let d = '', pen = false
      s.values.forEach((v, i) => {
        if (v == null) { pen = false; return }
        d += `${pen ? 'L' : 'M'}${px(t[i]).toFixed(1)} ${py(v).toFixed(1)}`
        pen = true
      })
      return d
    })
    return { paths, lo, hi }
  }, [t, series, duration, range, minSpan])

  const i = at(t, time)
  const x = (v) => (v / duration) * W
  const [tin, tout] = trim || [null, null]
  return (
    <div className={`min-h-0 flex flex-col bg-black rounded-md overflow-hidden ${className}`}>
      <div className="h-5 shrink-0 px-2 flex items-center gap-2 text-[10px]">
        <span className="text-white/70">{label}</span>
        {series.map((s) => (
          <span key={s.name} className="tabular-nums" style={{ color: s.color }}>
            {s.name} {fmt(i >= 0 ? s.values[i] : null, digits)}
          </span>
        ))}
        <span className="text-white/40">{unit}</span>
        <span className="flex-1" />
        {right}
      </div>
      <div className="relative flex-1 min-h-0">
        <span className="absolute top-0 left-1 z-10 text-[9px] leading-none px-1 py-0.5 rounded bg-black/70 text-white/50 tabular-nums">
          {fmt(hi, digits)}
        </span>
        <span className="absolute bottom-0 left-1 z-10 text-[9px] leading-none px-1 py-0.5 rounded bg-black/70 text-white/50 tabular-nums">
          {fmt(lo, digits)}
        </span>
        <svg className="absolute inset-0 w-full h-full" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
          {tin != null && <rect x={0} y={0} width={x(tin)} height={H} fill="#000" opacity={0.6} />}
          {tout != null && <rect x={x(tout)} y={0} width={W - x(tout)} height={H} fill="#000" opacity={0.6} />}
          {paths.map((d, k) => (
            <path key={k} d={d} fill="none" stroke={series[k].color} strokeWidth={1.5}
                  vectorEffect="non-scaling-stroke" />
          ))}
          <line x1={x(time)} x2={x(time)} y1={0} y2={H} stroke="#fff" strokeWidth={1}
                vectorEffect="non-scaling-stroke" />
        </svg>
      </div>
    </div>
  )
}
