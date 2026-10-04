import { useMemo } from 'react'
import LinePlot from '@/components/LinePlot'

const RANGE = [0, 120]   // mm; the gripper opens to 115

/** Gripper opening (mm), or a note when the episode has no gripper recording. */
export default function GripperGraph({ gripper, duration, time, trim, className = '' }) {
  const series = useMemo(() => (gripper ? [{ name: 'width', color: '#f59e0b', values: gripper.width }] : []),
    [gripper])
  if (!gripper) {
    return (
      <div className={`min-h-0 flex items-center justify-center bg-black rounded-md text-xs text-white/50 ${className}`}>
        no gripper recording
      </div>
    )
  }
  return (
    <LinePlot className={className} label="gripper" unit="mm" t={gripper.t} series={series} range={RANGE}
              duration={duration} time={time} trim={trim} />
  )
}
