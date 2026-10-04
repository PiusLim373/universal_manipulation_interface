import { AlertTriangle } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

const W = 160
const H = 24
const COLOUR = '#38bdf8'

/** Last few seconds of rotation speed (deg/s), peak per step as the server binned it. */
function Spark({ trace }) {
  if (!trace?.length) return <svg width={W} height={H} />
  const max = Math.max(90, ...trace)
  const x = (i) => (trace.length > 1 ? (i / (trace.length - 1)) * W : W)
  const y = (v) => H - 2 - (v / max) * (H - 4)
  const d = trace.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join('')
  return (
    <svg width={W} height={H} className="shrink-0">
      <path d={d} fill="none" stroke={COLOUR} strokeWidth={1.5} />
    </svg>
  )
}

// padded with figure spaces so the row does not jump as signs and digits change
const xyz = (v, n) => v?.map((a) => a.toFixed(n).padStart(6, ' ')).join(' ') ?? '—'

/**
 * Live wrist IMU from the capture state stream, throttled like the gripper row.
 * Null on a camera without one (D405), which renders nothing.
 */
export default function ImuReadout({ m, recording }) {
  if (!m) return null
  const stale = m.age_s != null && m.age_s > m.stale_s
  const bad = m.error || m.age_s == null || stale

  return (
    <div className={cn('flex items-center gap-3 shrink-0 rounded-md border px-3 py-1.5 text-xs',
                       m.error ? 'border-destructive/60' : stale ? 'border-[#f59e0b]/60' : 'border-border')}>
      <span className="uppercase tracking-wide text-[11px] text-muted-foreground w-[3.25rem]">imu</span>
      <span className="font-heading text-base tabular-nums w-20 text-right" title="rotation speed">
        {m.speed_dps != null ? `${m.speed_dps.toFixed(1)}°/s` : '—'}
      </span>
      <Spark trace={m.trace} />
      <span className="tabular-nums text-muted-foreground whitespace-pre">
        gyro {xyz(m.gyro_dps, 1)} °/s · accel {xyz(m.accel, 2)} m/s²
      </span>
      <span className="flex-1" />
      {bad ? (
        <span className={cn('flex items-center gap-1', m.error ? 'text-destructive' : 'text-[#f59e0b]')}>
          <AlertTriangle className="size-3.5" />
          {m.error || (m.age_s == null ? 'no sample yet' : `no sample for ${m.age_s.toFixed(1)} s`)}
        </span>
      ) : (
        <span className="tabular-nums text-muted-foreground">
          {m.gyro_hz.toFixed(0)} / {m.accel_hz.toFixed(0)} Hz
        </span>
      )}
      {recording && <Badge variant="destructive">{m.written} samples</Badge>}
    </div>
  )
}
