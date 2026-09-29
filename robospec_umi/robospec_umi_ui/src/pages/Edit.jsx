import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  Plus, Trash2, PackageCheck, PencilLine, Aperture, AlertTriangle, Camera, Upload, Lock, Download, Database,
} from 'lucide-react'
import Page from '@/components/Page'
import { IntrinsicActions, lockReason } from '@/components/IntrinsicLock'
import { useDatasetUpload } from '@/components/DatasetUpload'
import { Button, buttonVariants } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { api } from '@/lib/api'
import { cn } from '@/lib/utils'

const STAGE_LABEL = { select: 'selecting', verify: 'verifying', edit: 'editing', export: 'exporting' }
const fmt = (v, d) => (v == null ? '—' : Number(v).toFixed(d))

function IntrinsicBanner({ active }) {
  if (!active) {
    return (
      <Card className="ring-1 ring-destructive/40"><CardContent className="pt-6 flex items-start gap-3">
        <AlertTriangle className="size-5 text-destructive mt-0.5 shrink-0" />
        <div className="flex-1 space-y-3">
          <div>
            <p className="font-medium">No scene camera intrinsic is active</p>
            <p className="text-sm text-muted-foreground">
              Every pose is solved with it, so datasets cannot be created, edited or
              exported without one. Calibrate the camera, or upload an intrinsic
              <code> .json</code> and activate it.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Link to="/calibration/new">
              <Button size="sm"><Camera className="size-4" /> Calibrate the camera</Button>
            </Link>
            <Link to="/calibration?from=edit">
              <Button size="sm" variant="outline"><Upload className="size-4" /> Upload an intrinsic .json</Button>
            </Link>
          </div>
        </div>
      </CardContent></Card>
    )
  }
  const lc = active.locked_controls || {}
  const [w, h] = active.image_size || []
  return (
    <Card><CardContent className="pt-6 flex items-center gap-4">
      <Aperture className="size-6 text-[#22c55e] shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="text-xs text-muted-foreground">Active scene camera intrinsic</div>
        <div className="font-mono text-lg">{active.source_run || 'run unknown'}</div>
        <div className="text-xs text-muted-foreground tabular-nums">
          reproj {fmt(active.reproj, 3)} px · focus {lc.focus_absolute ?? '—'} / zoom {lc.zoom_absolute ?? '—'}
          {w && ` · ${w}×${h}`} · solved {active.solved_at?.replace('T', ' ') || '—'}
        </div>
        <div className="text-[11px] text-muted-foreground mt-1">
          New datasets are bound to it. A dataset bound to another intrinsic is locked
          until that one is active again, or it is rebound.
        </div>
      </div>
      <Link to="/calibration?from=edit">
        <Button size="sm" variant="outline">Use different Scene Camera Intrinsic?</Button>
      </Link>
    </CardContent></Card>
  )
}

function IntrinsicBadge({ st, built }) {
  const run = built && st.run && built !== st.run ? `${built}, rebound to ${st.run}` : built || st.run || '?'
  const label = st.status === 'inactive' ? `${run} · not active`
    : st.status === 'missing' ? `${run} · removed` : run
  return (
    <Badge variant="outline" title="scene intrinsic"
           className={cn('font-mono', st.status === 'inactive' && 'border-[#f59e0b]/60 text-[#f59e0b]',
             st.status === 'missing' && 'border-destructive/60 text-destructive')}>
      <Aperture /> {label}
    </Badge>
  )
}

function LockStrip({ r, exported, onChange, onError }) {
  return (
    <div className="mt-4 pt-3 border-t border-border flex items-center gap-3">
      <Lock className="size-4 text-[#f59e0b] shrink-0" />
      <span className="flex-1 text-xs text-muted-foreground">
        {lockReason(r.intrinsics)[0]}: editing and exporting are locked.
      </span>
      <IntrinsicActions state={r.intrinsics} pid={r.id} exported={exported}
                        onChange={onChange} onError={onError} className="justify-end" />
    </div>
  )
}

function FileLink({ file, label }) {
  return (
    <a href={api.datasetUrl(file)} download title={`download ${file}`}
       className={cn(buttonVariants({ size: 'sm', variant: 'ghost' }), 'text-xs gap-1 px-2')}>
      <Download className="size-3.5" /> {label}
    </a>
  )
}

export default function Edit() {
  const nav = useNavigate()
  const [rows, setRows] = useState(null)
  const [files, setFiles] = useState([])
  const [active, setActive] = useState(undefined)
  const [err, setErr] = useState(null)

  const load = useCallback(async () => {
    try {
      const [d, a] = await Promise.all([api.projects(), api.active()])
      setRows(d.projects); setFiles(d.files); setActive(a); setErr(null)
    } catch (e) { setErr(e.message); setRows([]) }
  }, [])
  useEffect(() => { load() }, [load])
  const up = useDatasetUpload(load)

  const remove = async (id) => {
    if (!confirm(`Delete draft ${id}? The capture sessions are not touched.`)) return
    try { await api.deleteProject(id); await load() }
    catch (e) { setErr(e.message) }
  }
  const removeDataset = async (file, mb, project) => {
    if (!confirm(`Delete ${file}${mb != null ? ` (${mb} MB)` : ''}${project ? ' and its project file' : ''}?`
      + ' The capture sessions are not touched.')) return
    try { await api.deleteDataset(file); await load() }
    catch (e) { setErr(e.message) }
  }

  const drafts = (rows || []).filter((r) => !r.export)
  const done = (rows || []).filter((r) => r.export)

  return (
    <Page title="Edit dataset" back="/">
      <div className="space-y-6">
        {err && <div className="rounded-md bg-destructive/10 text-destructive px-3 py-2 text-sm">{err}</div>}

        {active !== undefined && <IntrinsicBanner active={active} />}

        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="font-heading font-medium">Datasets</h2>
            <p className="text-sm text-muted-foreground">
              Pick capture sessions, verify them, trim or drop episodes, then export
              a <code>.zarr.zip</code> for training.
            </p>
          </div>
          <div className="flex gap-2 shrink-0">
            {up.button}
            <Link to="/edit/new/select" className={cn(!active && 'pointer-events-none')}>
              <Button size="sm" disabled={!active}><Plus className="size-4" /> New dataset</Button>
            </Link>
          </div>
        </div>
        {up.panel}

        {rows === null ? (
          <p className="text-sm text-muted-foreground">loading…</p>
        ) : rows.length === 0 && files.length === 0 ? (
          <Card><CardContent className="pt-6 text-sm text-muted-foreground">
            No datasets yet. Start one from the sessions in <code>data/capture/</code>, or
            upload a <code>.zarr.zip</code>.
          </CardContent></Card>
        ) : null}

        {drafts.length > 0 && (
          <section className="space-y-2">
            <h3 className="text-sm font-medium text-muted-foreground">In progress</h3>
            {drafts.map((r) => (
              <Card key={r.id}><CardContent className="pt-6"><div className="flex items-center gap-4">
                <PencilLine className="size-5 text-primary shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="font-medium">{r.id}</div>
                  <div className="text-xs text-muted-foreground truncate">
                    {r.sessions.length} session{r.sessions.length === 1 ? '' : 's'} · {r.episodes} episodes
                    {r.included != null && ` · ${r.included} included`} · {r.sessions.join(', ')}
                  </div>
                </div>
                <IntrinsicBadge st={r.intrinsics} />
                <Badge variant="outline">{STAGE_LABEL[r.stage] || r.stage}</Badge>
                <Button size="sm" disabled={r.intrinsics.status !== 'ok'}
                        onClick={() => nav(`/edit/${r.id}/${r.stage || 'verify'}`)}>Resume</Button>
                <Button size="sm" variant="ghost" onClick={() => remove(r.id)} title="Delete draft">
                  <Trash2 className="size-4" />
                </Button>
              </div>
              {r.intrinsics.status !== 'ok' && <LockStrip r={r} onChange={load} onError={setErr} />}
              </CardContent></Card>
            ))}
          </section>
        )}

        {done.length > 0 && (
          <section className="space-y-2">
            <h3 className="text-sm font-medium text-muted-foreground">Exported</h3>
            {done.map((r) => {
              const zip = `${r.id}_dataset.zarr.zip`
              // uploaded here from another machine: train-only, the intrinsic is informational
              const st = r.local ? r.intrinsics : { ...r.intrinsics, status: 'ok' }
              return (
                <Card key={r.id}><CardContent className="pt-6"><div className="flex items-center gap-4">
                  <PackageCheck className="size-5 text-[#22c55e] shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="font-medium flex items-center gap-2">
                      {zip}
                      {!r.local && <Badge variant="secondary" title={r.uploaded ? `uploaded ${r.uploaded}` : undefined}>uploaded</Badge>}
                    </div>
                    <div className="text-xs text-muted-foreground truncate">
                      {r.export.episodes} episodes · {r.export.steps} steps ({r.export.duration_s} s)
                      · {r.zip_mb != null ? `${r.zip_mb} MB` : <span className="text-destructive">zip missing</span>}
                      {' '}· exported {r.export.at?.replace('T', ' ')}
                    </div>
                  </div>
                  <span title="the intrinsic this .zarr.zip was built with">
                    <IntrinsicBadge st={st} built={r.export.intrinsics?.run} />
                  </span>
                  <div className="flex items-center">
                    {r.zip_mb != null && <FileLink file={zip} label="zip" />}
                    <FileLink file={`${r.id}_dataset.json`} label="json" />
                  </div>
                  <Button size="sm" variant="outline" disabled={!r.local || r.intrinsics.status !== 'ok'}
                          title={r.local ? undefined : 'its capture sessions are not on this machine'}
                          onClick={() => nav(`/edit/${r.id}/edit`)}>Open</Button>
                  <Button size="sm" variant="ghost" onClick={() => removeDataset(zip, r.zip_mb, true)}
                          title="Delete dataset">
                    <Trash2 className="size-4" />
                  </Button>
                </div>
                {r.local && r.intrinsics.status !== 'ok' && <LockStrip r={r} exported onChange={load} onError={setErr} />}
                </CardContent></Card>
              )
            })}
          </section>
        )}

        {files.length > 0 && (
          <section className="space-y-2">
            <h3 className="text-sm font-medium text-muted-foreground">Other datasets</h3>
            {files.map((f) => (
              <Card key={f.file}><CardContent className="pt-6 flex items-center gap-4">
                <Database className="size-5 text-muted-foreground shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="font-medium">{f.file}</div>
                  <div className="text-xs text-muted-foreground">
                    {f.size_mb} MB · {new Date(f.mtime * 1000).toLocaleString()} · no project file
                  </div>
                </div>
                <FileLink file={f.file} label="zip" />
                <Button size="sm" variant="ghost" onClick={() => removeDataset(f.file, f.size_mb, false)}
                        title="Delete dataset">
                  <Trash2 className="size-4" />
                </Button>
              </CardContent></Card>
            ))}
          </section>
        )}
      </div>
    </Page>
  )
}
