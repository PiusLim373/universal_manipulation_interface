import { memo, useMemo, useRef } from 'react'
import { STATUS_COLOR } from '@/lib/editStatus'

const W = 1000

// Consecutive frames with the same status -> one rect.
function runs(t, status, duration) {
  const out = []
  for (let i = 0; i < t.length; i++) {
    const end = i + 1 < t.length ? t[i + 1] : duration
    const last = out[out.length - 1]
    if (last && last.s === status[i]) last.b = end
    else out.push({ a: t[i], b: end, s: status[i] })
  }
  return out
}

const Strip = memo(function Strip({ t, status, duration }) {
  const x = (v) => (v / duration) * W
  return runs(t, status, duration).map((r, i) => (
    <rect key={i} x={x(r.a)} y={0} width={Math.max(x(r.b) - x(r.a), 0.5)} height={10}
          fill={STATUS_COLOR[r.s]} />
  ))
})

/** Seek bar: status per scene frame, the segments export keeps, and the trim. */
export default function TrimBar({ t, status, duration, spans, trim, time, onSeek }) {
  const box = useRef(null)
  const x = (v) => (Math.min(Math.max(v, 0), duration) / duration) * W
  const [tin, tout] = trim || [null, null]

  const seekAt = (e) => {
    const r = box.current.getBoundingClientRect()
    onSeek(Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1) * duration)
  }
  const down = (e) => { box.current.setPointerCapture(e.pointerId); seekAt(e) }
  const move = (e) => { if (e.buttons & 1) seekAt(e) }

  const spanRects = useMemo(() => (spans || []).map(([a, b], i) => (
    <rect key={i} x={x(a)} y={14} width={Math.max(x(b) - x(a), 1)} height={8} rx={2}
          fill="var(--color-primary)" />
  )), [spans, duration]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div ref={box} className="relative h-9 cursor-pointer select-none touch-none"
         onPointerDown={down} onPointerMove={move}>
      <svg className="absolute inset-0 w-full h-full" viewBox={`0 0 ${W} 36`}
           preserveAspectRatio="none">
        <Strip t={t} status={status} duration={duration} />
        {spanRects}
        <rect x={0} y={27} width={W} height={4} rx={2} fill="var(--color-muted)" />
        {tin != null && <rect x={0} y={0} width={x(tin)} height={36} fill="#000" opacity={0.6} />}
        {tout != null && <rect x={x(tout)} y={0} width={W - x(tout)} height={36} fill="#000" opacity={0.6} />}
        {tin != null && <line x1={x(tin)} x2={x(tin)} y1={0} y2={36} stroke="#22c55e"
                              strokeWidth={2} vectorEffect="non-scaling-stroke" />}
        {tout != null && <line x1={x(tout)} x2={x(tout)} y1={0} y2={36} stroke="#ef4444"
                               strokeWidth={2} vectorEffect="non-scaling-stroke" />}
        <line x1={x(time)} x2={x(time)} y1={0} y2={36} stroke="#fff" strokeWidth={2}
              vectorEffect="non-scaling-stroke" />
      </svg>
    </div>
  )
}
