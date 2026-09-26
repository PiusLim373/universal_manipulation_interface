import { useMemo, useState } from 'react'
import LinePlot from '@/components/LinePlot'
import { cn } from '@/lib/utils'

const COLORS = ['#ef4444', '#22c55e', '#3b82f6']

const split = (rows, names) => names.map((name, k) => ({
  name, color: COLORS[k], values: rows.map((r) => (r ? r[k] : null)),
}))

/** Absolute TCP pose in the scene-camera frame: xyz (mm) and rpy (deg). */
export default function TcpGraph({ tcp, duration, time, trim }) {
  const [show, setShow] = useState({ xyz: true, rpy: true })
  const xyz = useMemo(() => split(tcp.xyz, ['x', 'y', 'z']), [tcp])
  const rpy = useMemo(() => split(tcp.rpy, ['r', 'p', 'y']), [tcp])
  // the last visible plot cannot be switched off
  const toggle = (k) => setShow((s) => (s[k] && !s[k === 'xyz' ? 'rpy' : 'xyz'] ? s : { ...s, [k]: !s[k] }))

  const toggles = (
    <span className="flex gap-1">
      {['xyz', 'rpy'].map((k) => (
        <button key={k} type="button" onClick={() => toggle(k)}
                className={cn('text-[10px] px-1.5 leading-4 rounded border transition-colors',
                  show[k] ? 'border-primary bg-primary/20 text-primary'
                    : 'border-white/20 text-white/40 hover:text-white/70')}>
          {k}
        </button>
      ))}
    </span>
  )

  return (
    <div className="min-h-0 flex flex-col gap-1">
      {show.xyz && (
        <LinePlot className="flex-1" label="TCP" unit="mm" t={tcp.t} series={xyz} right={toggles}
                  duration={duration} time={time} trim={trim} minSpan={10} />
      )}
      {show.rpy && (
        <LinePlot className="flex-1" label="TCP" unit="deg" t={tcp.t} series={rpy}
                  right={show.xyz ? null : toggles}
                  duration={duration} time={time} trim={trim} minSpan={5} />
      )}
    </div>
  )
}
