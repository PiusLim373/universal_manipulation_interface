import { Link, useLocation } from 'react-router-dom'
import { ChevronLeft } from 'lucide-react'
import logo from '/robospec_logo.png'
import { Badge } from '@/components/ui/badge'
import { useCamera } from '@/hooks/useCamera'

const READY = '#22c55e'
const IDLE = '#4b5563'

/** One device's lease, named. Green when free, accent while held. */
function DeviceBadge({ label, lease }) {
  const held = !!lease
  const colour = held ? 'oklch(0.700 0.158 258.0)' : READY
  return (
    <div className="flex items-center gap-1.5" title={
      held ? `${label} held by ${lease.owner} for ${lease.held_s}s` : `${label} free`}>
      <span className="inline-block size-2 rounded-full"
            style={{ background: colour, boxShadow: `0 0 6px ${colour}` }} />
      <span className="text-[11px] text-muted-foreground">
        {label}{held && <span className="text-foreground"> · {lease.owner.split('.').pop()}</span>}
      </span>
    </div>
  )
}

export default function Header({ title, back }) {
  const { devices, reachable } = useCamera()
  const loc = useLocation()
  return (
    <header className="flex items-center justify-between h-14 px-4 border-b border-border glass shrink-0">
      <div className="flex items-center gap-3 min-w-0">
        <Link to="/" className="shrink-0">
          <img src={logo} alt="Robospec" className="h-7 select-none" />
        </Link>
        {back && loc.pathname !== '/' && (
          <Link to={back}
                className="inline-flex items-center gap-1 text-sm text-muted-foreground
                           hover:text-foreground transition-colors">
            <ChevronLeft className="size-4" /> Back
          </Link>
        )}
        {title && <span className="text-sm font-medium truncate">{title}</span>}
      </div>
      <div className="flex items-center gap-4">
        <DeviceBadge label="scene" lease={devices.scene} />
        <DeviceBadge label="wrist" lease={devices.wrist} />
        <Badge variant={reachable ? 'secondary' : 'destructive'}>
          {reachable ? 'server ok' : 'server offline'}
        </Badge>
      </div>
    </header>
  )
}

export { IDLE, READY }
