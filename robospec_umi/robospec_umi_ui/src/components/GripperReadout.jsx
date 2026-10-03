import { AlertTriangle } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

const W = 160
const H = 24

/** Last few seconds of opening (mm), as the server decimated it. */
function Spark({ trace, max }) {
  if (!trace?.length) return <svg width={W} height={H} />
  const x = (i) => (trace.length > 1 ? (i / (trace.length - 1)) * W : W)
  const y = (v) => H - 2 - (Math.min(v, max) / max) * (H - 4)
  const d = trace.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join('')
  return (
    <svg width={W} height={H} className="shrink-0">
      <path d={d} fill="none" stroke="#f59e0b" strokeWidth={1.5} />
    </svg>
  )
}

/**
 * Live gripper opening from the capture state stream. The trace arrives already
 * throttled (10 Hz idle, 4 Hz recording), so this renders what it is given.
 */
export default function GripperReadout({ g, recording }) {
  if (!g) return null
  const max = (g.max_width_m ?? 0.115) * 1000
  const mm = g.width_m != null ? g.width_m * 1000 : null
  const stale = g.age_s != null && g.age_s > g.stale_s
  const bad = g.error || g.age_s == null || stale

  return (
    <div className={cn('flex items-center gap-3 shrink-0 rounded-md border px-3 py-1.5 text-xs',
                       g.error ? 'border-destructive/60' : stale ? 'border-[#f59e0b]/60' : 'border-border')}>
      <span className="uppercase tracking-wide text-[11px] text-muted-foreground">gripper</span>
      <span className="font-heading text-base tabular-nums w-20 text-right">
        {mm != null ? `${mm.toFixed(1)} mm` : '—'}
      </span>
      <div className="w-28 h-1.5 rounded bg-muted overflow-hidden shrink-0">
        <div className="h-full bg-[#f59e0b]" style={{ width: `${mm != null ? (100 * mm) / max : 0}%` }} />
      </div>
      <span className="tabular-nums text-muted-foreground w-9">
        {mm != null ? `${Math.round((100 * mm) / max)}%` : ''}
      </span>
      <Spark trace={g.trace} max={max} />
      <span className="flex-1" />
      {bad ? (
        <span className={cn('flex items-center gap-1', g.error ? 'text-destructive' : 'text-[#f59e0b]')}>
          <AlertTriangle className="size-3.5" />
          {g.error || (g.age_s == null ? 'no reading yet' : `no reading for ${g.age_s.toFixed(1)} s`)}
        </span>
      ) : (
        <span className="tabular-nums text-muted-foreground">
          {g.hz.toFixed(1)} Hz · force {g.force_raw} <span className="opacity-60">raw</span>
        </span>
      )}
      {recording && <Badge variant="destructive">{g.written} samples</Badge>}
    </div>
  )
}
