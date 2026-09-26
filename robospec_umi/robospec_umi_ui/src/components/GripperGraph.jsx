import { useMemo } from 'react'
import LinePlot from '@/components/LinePlot'

/** Gripper opening (mm). The range floors at 0-100 so a constant reads as flat. */
export default function GripperGraph({ gripper, duration, time, trim }) {
  const series = useMemo(() => [{ name: 'width', color: '#f59e0b', values: gripper.width }], [gripper])
  const range = useMemo(() => [0, Math.max(100, ...gripper.width) * 1.05], [gripper])
  return (
    <LinePlot label="gripper" unit="mm" t={gripper.t} series={series} range={range}
              duration={duration} time={time} trim={trim} />
  )
}
