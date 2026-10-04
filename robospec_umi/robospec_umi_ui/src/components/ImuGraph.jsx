import { useMemo, useState } from 'react'
import LinePlot from '@/components/LinePlot'
import { cn } from '@/lib/utils'

const COLORS = ['#ef4444', '#22c55e', '#3b82f6']
const PLOTS = { gyro: { unit: 'deg/s', minSpan: 20 }, accel: { unit: 'm/s²', minSpan: 2 } }

const split = (rows) => ['x', 'y', 'z'].map((name, k) => ({
  name, color: COLORS[k], values: rows.map((r) => (r ? r[k] : null)),
}))

/** Wrist IMU on the episode clock, in the IMU's own axes (x right, y down, z out of the lens). */
export default function ImuGraph({ imu, duration, time, trim, className = '' }) {
  const [show, setShow] = useState('gyro')
  const series = useMemo(() => split(imu[show]), [imu, show])

  const toggles = (
    <span className="flex gap-1">
      {Object.keys(PLOTS).map((k) => (
        <button key={k} type="button" onClick={() => setShow(k)}
                className={cn('text-[10px] px-1.5 leading-4 rounded border transition-colors',
                  show === k ? 'border-primary bg-primary/20 text-primary'
                    : 'border-white/20 text-white/40 hover:text-white/70')}>
          {k}
        </button>
      ))}
    </span>
  )

  return (
    <LinePlot className={className} label="IMU" unit={PLOTS[show].unit} t={imu.t} series={series}
              right={toggles} duration={duration} time={time} trim={trim}
              minSpan={PLOTS[show].minSpan} digits={show === 'gyro' ? 1 : 2} />
  )
}
