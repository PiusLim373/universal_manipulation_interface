import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Plus, Trash2, CheckCircle2, Circle, AlertTriangle, ScanEye } from 'lucide-react'
import Page from '@/components/Page'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { api } from '@/lib/api'
import { cn } from '@/lib/utils'

const fmt = (v, d = 4) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d))
const when = (t) => (t ? new Date(t * 1000).toLocaleString() : '—')

function ActiveCard({ active }) {
  if (!active) {
    return (
      <Card>
        <CardContent className="pt-6 flex items-start gap-3">
          <AlertTriangle className="size-5 text-destructive mt-0.5 shrink-0" />
          <div>
            <p className="font-medium">No active calibration</p>
            <p className="text-sm text-muted-foreground">
              Capture and verify will not cross-check focus and zoom until one is
              activated, which means a wrong lens setting would go unnoticed.
            </p>
          </div>
        </CardContent>
      </Card>
    )
  }
  const lc = active.locked_controls || {}
  return (
    <Card>
      <CardContent className="pt-6 space-y-3">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="size-5 text-[#22c55e]" />
          <span className="font-heading font-medium">Active calibration</span>
          {active.source_run
            ? <Badge variant="secondary">{active.source_run}</Badge>
            : <Badge variant="outline" title="solved before provenance was recorded">
                run unknown
              </Badge>}
        </div>
        <dl className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-2 text-sm">
          <div><dt className="text-muted-foreground text-xs">reprojection</dt>
               <dd className="tabular-nums">{fmt(active.reproj)} px</dd></div>
          <div><dt className="text-muted-foreground text-xs">held out</dt>
               <dd className="tabular-nums">{fmt(active.holdout)} px</dd></div>
          <div><dt className="text-muted-foreground text-xs">images</dt>
               <dd className="tabular-nums">{active.n_images ?? '—'}</dd></div>
          <div><dt className="text-muted-foreground text-xs">model</dt>
               <dd>{active.model ?? '—'}</dd></div>
          <div><dt className="text-muted-foreground text-xs">FOV</dt>
               <dd className="tabular-nums">
                 {active.fov ? `${fmt(active.fov.horizontal, 1)}° × ${fmt(active.fov.vertical, 1)}°` : '—'}
               </dd></div>
          <div><dt className="text-muted-foreground text-xs">focus / zoom</dt>
               <dd className="tabular-nums">{lc.focus_absolute ?? '—'} / {lc.zoom_absolute ?? '—'}</dd></div>
          <div className="col-span-2"><dt className="text-muted-foreground text-xs">solved</dt>
               <dd>{active.solved_at || '—'}</dd></div>
        </dl>
        <p className="text-[11px] text-muted-foreground">
          Recording at a different focus or zoom than these silently invalidates
          every pose. capture, verify and build_zarr all check against this file.
        </p>
        <Link to="/calibration/test">
          <Button size="sm" variant="outline">
            <ScanEye className="size-4" /> Test this calibration
          </Button>
        </Link>
      </CardContent>
    </Card>
  )
}

export default function Calibration() {
  const [runs, setRuns] = useState([])
  const [active, setActive] = useState(null)
  const [busy, setBusy] = useState(null)
  const [err, setErr] = useState(null)

  const load = useCallback(async () => {
    try {
      const d = await api.runs()
      setRuns(d.runs || [])
      setActive(d.active || null)
      setErr(null)
    } catch (e) { setErr(e.message) }
  }, [])

  useEffect(() => { load() }, [load])

  const activate = async (name) => {
    setBusy(name)
    try { await api.activate(name); await load() }
    catch (e) { setErr(e.message) }
    finally { setBusy(null) }
  }

  const remove = async (name) => {
    if (!confirm(`Delete calibration run ${name}? The captured frames go with it.`)) return
    setBusy(name)
    try { await api.deleteRun(name); await load() }
    catch (e) { setErr(e.message) }
    finally { setBusy(null) }
  }

  return (
    <Page title="Calibration" back="/">
      <div className="space-y-6">
        {err && (
          <div className="rounded-md bg-destructive/10 text-destructive px-3 py-2 text-sm">{err}</div>
        )}

        <ActiveCard active={active} />

        <div className="flex items-center justify-between">
          <h2 className="font-heading font-medium">Calibration runs</h2>
          <Link to="/calibration/new">
            <Button size="sm"><Plus className="size-4" /> New calibration</Button>
          </Link>
        </div>

        {runs.length === 0 ? (
          <Card><CardContent className="pt-6 text-sm text-muted-foreground">
            No runs yet. Start one to capture board views and solve intrinsics.
          </CardContent></Card>
        ) : (
          <div className="rounded-lg border border-border overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-muted/50 text-muted-foreground">
                <tr className="text-left">
                  <th className="px-3 py-2 font-medium">run</th>
                  <th className="px-3 py-2 font-medium">frames</th>
                  <th className="px-3 py-2 font-medium">reproj</th>
                  <th className="px-3 py-2 font-medium">images</th>
                  <th className="px-3 py-2 font-medium">modified</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => {
                  const isActive = active?.source_run === r.name
                  return (
                    <tr key={r.name} className={cn('border-t border-border',
                                                   isActive && 'bg-primary/5')}>
                      <td className="px-3 py-2 font-mono text-xs flex items-center gap-1.5">
                        {isActive
                          ? <CheckCircle2 className="size-3.5 text-[#22c55e] shrink-0" />
                          : <Circle className="size-3.5 text-muted-foreground/40 shrink-0" />}
                        {r.name}
                      </td>
                      <td className="px-3 py-2 tabular-nums">{r.frames}</td>
                      <td className="px-3 py-2 tabular-nums">
                        {r.solved ? `${fmt(r.reproj)} px` :
                          <span className="text-muted-foreground">not solved</span>}
                      </td>
                      <td className="px-3 py-2 tabular-nums">{r.n_images ?? '—'}</td>
                      <td className="px-3 py-2 text-muted-foreground text-xs">{when(r.mtime)}</td>
                      <td className="px-3 py-2">
                        <div className="flex gap-1.5 justify-end">
                          {/* Test BEFORE activating: otherwise the only way to
                              check a new calibration is to make it live first. */}
                          <Link to={`/calibration/test?run=${encodeURIComponent(r.name)}`}>
                            <Button size="xs" variant="ghost" disabled={!r.solved}
                                    title="test this run against the live camera">
                              <ScanEye className="size-3.5" />
                            </Button>
                          </Link>
                          <Button size="xs" variant="outline"
                                  disabled={!r.solved || isActive || busy === r.name}
                                  onClick={() => activate(r.name)}>
                            {isActive ? 'Active' : 'Activate'}
                          </Button>
                          <Button size="icon-xs" variant="ghost"
                                  disabled={busy === r.name}
                                  onClick={() => remove(r.name)} title="delete run">
                            <Trash2 className="size-3.5" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Page>
  )
}
