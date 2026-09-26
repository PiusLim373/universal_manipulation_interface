import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Loader2, Lock, Upload, Camera } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { api } from '@/lib/api'

/** Why a dataset is locked, from its intrinsics_state (or {status: 'none'}). */
export function lockReason({ status, run, active }) {
  const now = active ? `the active one, ${active.run}` : 'an intrinsic (none is active)'
  if (status === 'none') {
    return ['No scene intrinsic is active',
      'A dataset is bound to the scene intrinsic it starts with. Calibrate the camera or upload an intrinsic .json, then activate it.']
  }
  if (status === 'inactive') {
    return [`Scene intrinsic ${run} is not active`,
      `This dataset is bound to ${run}. Activate it again to carry on, or rebind the dataset to ${now}.`]
  }
  return [run ? `Scene intrinsic ${run} was removed` : 'This dataset has no scene intrinsic',
    `Its calibration run is gone or was re-solved, so its poses cannot be reproduced. Rebind the dataset to ${now}.`]
}

/** Activate the bound intrinsic, rebind to the active one, or go get one. */
export function IntrinsicActions({ state, pid, exported, onChange, onError, size = 'sm',
                                  className = 'justify-center' }) {
  const [busy, setBusy] = useState(null)
  const { status, run, active } = state

  const act = async (what, fn) => {
    setBusy(what)
    try { await onChange(await fn()) }
    catch (e) { onError?.(e.message) }
    finally { setBusy(null) }
  }
  const activate = () => {
    if (active && !confirm(`Activate ${run}? New datasets use it too, and datasets `
      + `bound to ${active.run} lock until it is active again.`)) return
    act('activate', () => api.activate(run))
  }
  const rebind = () => {
    if (!confirm(`Rebind dataset ${pid} from ${run || 'nothing'} to ${active.run}?\n\n`
      + `Every episode re-renders its poses and videos with ${active.run} (about 15 s each, `
      + 'in the background; board detections are reused). Verify runs again. Trims and '
      + 'exclusions are kept.'
      + (exported ? `\n\nThe exported .zarr.zip stays as built with ${run} until you export again.` : ''))) return
    act('rebind', () => api.rebindProject(pid))
  }

  return (
    <div className={`flex flex-wrap gap-2 ${className}`}>
      {status === 'inactive' && (
        <Button size={size} onClick={activate} disabled={!!busy}>
          {busy === 'activate' && <Loader2 className="size-4 animate-spin" />}
          Activate {run}
        </Button>
      )}
      {active && status !== 'none' && (
        <Button size={size} variant={status === 'inactive' ? 'outline' : 'default'}
                onClick={rebind} disabled={!!busy}>
          {busy === 'rebind' && <Loader2 className="size-4 animate-spin" />}
          Rebind to {active.run}
        </Button>
      )}
      {!active && status !== 'inactive' && (
        <>
          <Link to="/calibration/new">
            <Button size={size}><Camera className="size-4" /> Calibrate the camera</Button>
          </Link>
          <Link to="/calibration?from=edit">
            <Button size={size} variant="outline"><Upload className="size-4" /> Upload an intrinsic .json</Button>
          </Link>
        </>
      )}
    </div>
  )
}

/** Replaces a stage while the dataset's intrinsic is not the active one. */
export default function IntrinsicLock({ state, pid, exported, onChange }) {
  const [err, setErr] = useState(null)
  const [title, body] = lockReason(state)
  return (
    <div className="max-w-xl mx-auto w-full pt-10">
      <Card><CardContent className="pt-6 space-y-4 text-center">
        <Lock className="size-8 mx-auto text-[#f59e0b]" />
        <div className="space-y-1">
          <p className="font-heading font-medium">{title}</p>
          <p className="text-sm text-muted-foreground">{body}</p>
        </div>
        {err && <div className="rounded-md bg-destructive/10 text-destructive px-3 py-2 text-sm">{err}</div>}
        <IntrinsicActions state={state} pid={pid} exported={exported} onChange={onChange} onError={setErr} />
      </CardContent></Card>
    </div>
  )
}
