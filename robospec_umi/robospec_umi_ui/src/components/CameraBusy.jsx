import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Lock } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { api } from '@/lib/api'

const WHERE = {
  'calibration.tune':    { label: 'Calibration preview', to: '/calibration/new/lock' },
  'calibration.capture': { label: 'Calibration capture', to: '/calibration/new/capture' },
  'calibration.test':    { label: 'Calibration test',    to: '/calibration/test' },
  'capture.session':     { label: 'Capture',             to: '/capture/new/record' },
}

/**
 * Shown instead of a failed request when another feature holds the camera.
 *
 * Only one process can stream a V4L2 device, so this is a normal state, not an
 * error. It matters more once the capture page exists and two features
 * genuinely compete -- an unexplained failure would then be the common case.
 */
export default function CameraBusy({ owner, device = 'scene', onReleased }) {
  const info = WHERE[owner] || { label: owner, to: null }
  const [err, setErr] = useState(null)
  return (
    <Card className="max-w-lg">
      <CardContent className="space-y-4 pt-6">
        <div className="flex items-start gap-3">
          <Lock className="size-5 text-primary mt-0.5 shrink-0" />
          <div className="space-y-1">
            <p className="font-medium">The {device} camera is in use</p>
            <p className="text-sm text-muted-foreground">
              {info.label} currently holds it. Only one thing can stream a camera
              at a time, so finish or stop that first.
            </p>
          </div>
        </div>
        {err && <p className="text-sm text-destructive">{err}</p>}
        <div className="flex gap-2">
          {info.to && <Link to={info.to}><Button size="sm">Go to {info.label}</Button></Link>}
          <Button size="sm" variant="outline"
                  onClick={async () => {
                    setErr(null)
                    // Refused with a 409 while an episode is armed -- taking a
                    // camera mid-take would destroy the recording.
                    try { await api.releaseCam(device); onReleased?.() }
                    catch (e) { setErr(e.message) }
                  }}>
            Take over
          </Button>
        </div>
        <p className="text-[11px] text-muted-foreground">
          Take over stops the other session. Use it if something crashed and left
          the camera held.
        </p>
      </CardContent>
    </Card>
  )
}
