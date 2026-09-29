import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Copy, Check, Download, Cpu, Terminal, Loader2 } from 'lucide-react'
import Page from '@/components/Page'
import { useDatasetUpload } from '@/components/DatasetUpload'
import { Button, buttonVariants } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { api } from '@/lib/api'
import { cn, fmtMB } from '@/lib/utils'

// Defaults are the last GCP run. `config` fields default to the yaml and are
// only emitted when changed.
const FIELDS = [
  { key: 'batch', label: 'Batch size (train)', def: '16', min: 1, arg: 'dataloader.batch_size' },
  { key: 'val_batch', label: 'Batch size (val)', def: '16', min: 1, arg: 'val_dataloader.batch_size' },
  { key: 'workers', label: 'Workers (train)', def: '10', min: 0, arg: 'dataloader.num_workers' },
  { key: 'val_workers', label: 'Workers (val)', def: '4', min: 0, arg: 'val_dataloader.num_workers' },
  { key: 'accum', label: 'Gradient accumulation', def: '4', min: 1, arg: 'training.gradient_accumulate_every' },
  { key: 'warmup', label: 'LR warmup steps', def: '500', min: 0, arg: 'training.lr_warmup_steps' },
  { key: 'topk', label: 'Top-k checkpoints', def: '3', min: 1, arg: 'checkpoint.topk.k' },
  { key: 'epochs', label: 'Epochs', config: 'num_epochs', min: 1, arg: 'training.num_epochs' },
  { key: 'lr', label: 'Learning rate', config: 'lr', float: true, arg: 'optimizer.lr' },
]
// one command line per group, as in the original command
const LINES = [['batch', 'val_batch'], ['workers', 'val_workers'], ['accum'], ['warmup'], ['topk'], ['epochs'], ['lr']]
const LOGGING = ['disabled', 'online', 'offline']
const ENV = 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'

const shq = (s) => (/^[\w./:=@%+-]+$/.test(s) ? s : `'${s.replace(/'/g, `'\\''`)}'`)
const when = (t) => new Date(t * 1000).toLocaleString()

function invalid(f, v) {
  if (f.float) return !(Number(v) > 0) && 'must be a number > 0'
  return !(/^\d+$/.test(v) && Number(v) >= f.min) && `must be an integer ≥ ${f.min}`
}

function buildCmd({ dataset, form, logging, mode, container, info }) {
  const emit = (f) => !f.config || Number(form[f.key]) !== Number(info.defaults[f.config])
  const byKey = Object.fromEntries(FIELDS.map((f) => [f.key, f]))
  const args = [
    `--config-name=${info.config}`,
    `task.dataset_path=${shq(`data/dataset/${dataset}`)}`,
    `logging.mode=${logging}`,
    ...LINES.map((ks) => ks.map((k) => byKey[k]).filter(emit)
      .map((f) => `${f.arg}=${form[f.key]}`).join(' ')).filter(Boolean),
  ]
  const head = mode === 'docker'
    ? [`docker exec -it -w /workspace -e ${ENV} ${shq(container)}`, 'python train.py']
    : [`cd ${shq(info.repo)} && conda activate umi2 &&`, `${ENV} python train.py`]
  return [...head, ...args].map((l, i, a) => (i ? '  ' : '') + l + (i < a.length - 1 ? ' \\' : '')).join('\n')
}

async function copyText(text) {
  // navigator.clipboard needs https or localhost; a remote box on plain http has none
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch { /* fall back below */ }
  const t = Object.assign(document.createElement('textarea'), { value: text })
  t.style.position = 'fixed'
  t.style.opacity = '0'
  document.body.appendChild(t)
  t.select()
  const ok = document.execCommand('copy')
  t.remove()
  return ok
}

function Section({ n, title, right, children }) {
  return (
    <section className="space-y-3">
      <div className="flex items-center gap-3">
        <h2 className="font-heading font-medium flex-1">{n && <span className="text-muted-foreground">{n} · </span>}{title}</h2>
        {right}
      </div>
      {children}
    </section>
  )
}

function GpuLine() {
  const [g, setG] = useState(null)
  useEffect(() => {
    let dead = false
    const tick = () => api.gpu().then((d) => !dead && setG(d)).catch(() => {})
    tick()
    const id = setInterval(tick, 5000)
    return () => { dead = true; clearInterval(id) }
  }, [])
  if (!g) return null
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-sm">
      <Cpu className="size-4 text-muted-foreground" />
      {g.error ? <span className="text-muted-foreground">No NVIDIA GPU visible ({g.error})</span>
        : g.gpus.map((x, i) => (
          <span key={i} className="tabular-nums">
            {x.name} · <span className="text-muted-foreground">
              {(x.mem_used_mb / 1024).toFixed(1)} / {(x.mem_total_mb / 1024).toFixed(1)} GB · {x.util}% util
            </span>
          </span>
        ))}
    </div>
  )
}

function Datasets({ picked, setPicked }) {
  const [rows, setRows] = useState(null)
  const load = useCallback(() => api.datasets().then((d) => {
    setRows(d)
    setPicked((p) => (d.some((r) => r.file === p) ? p : d[0]?.file ?? null))
  }).catch(() => setRows([])), [setPicked])
  useEffect(() => { load() }, [load])
  const up = useDatasetUpload(load)

  return (
    <Section n="1" title="Dataset" right={<>{up.button}
      <Link to="/edit" className="text-xs text-muted-foreground hover:text-foreground">manage in Edit dataset</Link></>}>
      {up.panel}
      {rows === null ? <p className="text-sm text-muted-foreground">loading…</p>
        : rows.length === 0 ? (
          <Card><CardContent className="pt-6 text-sm text-muted-foreground">
            No <code>.zarr.zip</code> in <code>data/dataset/</code>. Export one from Edit dataset, or upload one.
          </CardContent></Card>
        ) : (
          <div className="rounded-lg border border-border overflow-hidden">
            {rows.map((r) => (
              <label key={r.file} className={cn('flex items-center gap-3 px-3 py-2 border-t border-border first:border-t-0 cursor-pointer hover:bg-muted/40',
                picked === r.file && 'bg-primary/5')}>
                <input type="radio" name="dataset" className="accent-primary" checked={picked === r.file}
                       onChange={() => setPicked(r.file)} />
                <span className="font-mono text-sm flex-1 truncate">{r.file}</span>
                <span className="text-xs text-muted-foreground tabular-nums">
                  {r.episodes != null && `${r.episodes} episodes · ${r.steps} steps (${r.duration_s} s) · `}
                  {fmtMB(r.size_mb)} · {when(r.mtime)}
                </span>
                {r.intrinsic && <Badge variant="outline" className="font-mono">{r.intrinsic}</Badge>}
                {r.uploaded && <Badge variant="secondary">uploaded</Badge>}
              </label>
            ))}
          </div>
        )}
    </Section>
  )
}

function Runs() {
  const [runs, setRuns] = useState(null)
  useEffect(() => {
    let dead = false
    const tick = () => api.trainRuns().then((d) => !dead && setRuns(d)).catch(() => {})
    tick()
    const id = setInterval(tick, 10000)
    return () => { dead = true; clearInterval(id) }
  }, [])

  return (
    <Section n="4" title="Checkpoints" right={<span className="text-xs text-muted-foreground">data/outputs/ · refreshes every 10 s</span>}>
      {runs === null ? <p className="text-sm text-muted-foreground">loading…</p>
        : runs.length === 0 ? (
          <Card><CardContent className="pt-6 text-sm text-muted-foreground">
            No training runs yet. They appear here once <code>train.py</code> starts writing to <code>data/outputs/</code>.
          </CardContent></Card>
        ) : runs.map((r) => {
          const p = r.progress
          const frac = p && r.num_epochs ? Math.min(1, (p.epoch + 1) / r.num_epochs) : null
          return (
            <Card key={r.run}><CardContent className="pt-6 space-y-3">
              <div className="flex items-center gap-3">
                <span className="font-mono text-sm flex-1 truncate">{r.run}</span>
                {r.dataset && <span className="text-xs text-muted-foreground font-mono truncate">{r.dataset.split('/').pop()}</span>}
                {r.running
                  ? <Badge className="bg-[#22c55e]/15 text-[#22c55e] border-[#22c55e]/40">
                      <span className="size-1.5 rounded-full bg-[#22c55e] animate-pulse" /> running
                    </Badge>
                  : <Badge variant="outline">stopped</Badge>}
              </div>
              {p && (
                <div className="space-y-1">
                  <div className="text-xs text-muted-foreground tabular-nums">
                    epoch {p.epoch + 1}{r.num_epochs ? ` / ${r.num_epochs}` : ''} · step {p.global_step}
                    {p.train_loss != null && ` · train loss ${p.train_loss.toFixed(4)}`}
                  </div>
                  {frac != null && (
                    <div className="h-1.5 rounded bg-muted overflow-hidden">
                      <div className="h-full bg-primary" style={{ width: `${frac * 100}%` }} />
                    </div>
                  )}
                </div>
              )}
              {r.checkpoints.length === 0
                ? <p className="text-xs text-muted-foreground">no checkpoint yet</p>
                : (
                  <div className="divide-y divide-border rounded-md border border-border">
                    {r.checkpoints.map((c) => (
                      <div key={c.path} className="flex items-center gap-3 px-3 py-1.5 text-sm">
                        <span className="font-mono text-xs flex-1 truncate">{c.name}</span>
                        <span className="text-xs text-muted-foreground tabular-nums">
                          {fmtMB(c.size_mb)} · {when(c.mtime)}
                        </span>
                        {c.saving
                          ? <span className="text-xs text-muted-foreground inline-flex items-center gap-1 w-24 justify-end">
                              <Loader2 className="size-3 animate-spin" /> saving…
                            </span>
                          : <a href={api.ckptUrl(c.path)} download
                               className={cn(buttonVariants({ size: 'sm', variant: 'outline' }), 'w-24')}>
                              <Download className="size-3.5" /> Download
                            </a>}
                      </div>
                    ))}
                  </div>
                )}
            </CardContent></Card>
          )
        })}
    </Section>
  )
}

export default function Training() {
  const [info, setInfo] = useState(null)
  const [dataset, setDataset] = useState(null)
  const [form, setForm] = useState(() => Object.fromEntries(FIELDS.filter((f) => f.def).map((f) => [f.key, f.def])))
  const [logging, setLogging] = useState('disabled')
  const [mode, setMode] = useState('docker')
  const [container, setContainer] = useState('robospec_umi')
  const [errs, setErrs] = useState({})
  const [cmd, setCmd] = useState(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    api.trainInfo().then((d) => {
      setInfo(d)
      setMode(d.in_container ? 'docker' : 'host')
      setContainer(d.container)
      setForm((f) => ({ ...f, epochs: String(d.defaults.num_epochs ?? ''), lr: String(d.defaults.lr ?? '') }))
    }).catch(() => {})
  }, [])
  // any change invalidates a generated command until it is confirmed again
  const changed = (fn) => (v) => { fn(v); setCmd(null) }
  const pickDataset = useCallback((v) => { setDataset(v); setCmd(null) }, [])

  const set = (k, v) => { setForm((f) => ({ ...f, [k]: v })); setErrs((e) => ({ ...e, [k]: null })); setCmd(null) }
  const generate = () => {
    const e = Object.fromEntries(FIELDS.map((f) => [f.key, invalid(f, form[f.key])]).filter(([, v]) => v))
    if (mode === 'docker' && !container.trim()) e.container = 'required'
    setErrs(e)
    if (Object.keys(e).length) return
    setCmd(buildCmd({ dataset, form, logging, mode, container: container.trim(), info }))
    setCopied(false)
  }
  const copy = async () => {
    setCopied((await copyText(cmd)) ? 'ok' : 'fail')
    setTimeout(() => setCopied(false), 2500)
  }

  const eff = Number(form.batch) * Number(form.accum)

  return (
    <Page title="Training" back="/">
      <div className="space-y-8">
        <div className="space-y-2">
          <p className="text-sm text-muted-foreground">
            Pick a dataset and the hyperparameters, then copy the command into a terminal
            on the machine that runs the container. Checkpoints show up below as the run
            writes them.
          </p>
          <GpuLine />
        </div>

        <Datasets picked={dataset} setPicked={pickDataset} />

        <Section n="2" title="Hyperparameters">
          <Card><CardContent className="pt-6 space-y-5">
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
              {FIELDS.map((f) => (
                <label key={f.key} className="space-y-1 text-sm">
                  <span className="text-muted-foreground">{f.label}</span>
                  <input value={form[f.key] ?? ''} onChange={(e) => set(f.key, e.target.value.trim())}
                         inputMode={f.float ? 'decimal' : 'numeric'}
                         className={cn('w-full rounded-md border bg-transparent px-2 py-1.5 font-mono text-sm outline-none focus:border-primary',
                           errs[f.key] ? 'border-destructive' : 'border-border')} />
                  <span className="text-[11px] text-muted-foreground block h-4">
                    {errs[f.key] ? <span className="text-destructive">{errs[f.key]}</span>
                      : f.config ? (Number(form[f.key]) === Number(info?.defaults[f.config])
                        ? 'config default' : `config: ${info?.defaults[f.config]}`)
                        : f.key === 'accum' && eff > 0 ? `effective batch ${eff}` : ''}
                  </span>
                </label>
              ))}
              <label className="space-y-1 text-sm">
                <span className="text-muted-foreground">Logging (wandb)</span>
                <select value={logging} onChange={(e) => changed(setLogging)(e.target.value)}
                        className="w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm">
                  {LOGGING.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
                <span className="text-[11px] text-muted-foreground block h-4">
                  {logging === 'online' && 'needs wandb login on that machine'}
                </span>
              </label>
            </div>

            <div className="flex flex-wrap items-end gap-4 pt-1 border-t border-border">
              <div className="space-y-1 text-sm pt-4">
                <span className="text-muted-foreground block">Run in</span>
                <div className="inline-flex rounded-md border border-border overflow-hidden">
                  {[['docker', 'Docker container'], ['host', 'This machine']].map(([m, l]) => (
                    <button key={m} type="button" onClick={() => changed(setMode)(m)}
                            className={cn('px-3 py-1.5 text-sm', mode === m ? 'bg-primary text-primary-foreground' : 'hover:bg-muted')}>
                      {l}
                    </button>
                  ))}
                </div>
              </div>
              {mode === 'docker' ? (
                <label className="space-y-1 text-sm">
                  <span className="text-muted-foreground block">Container</span>
                  <input value={container} onChange={(e) => changed(setContainer)(e.target.value)}
                         className={cn('w-44 rounded-md border bg-transparent px-2 py-1.5 font-mono text-sm outline-none focus:border-primary',
                           errs.container ? 'border-destructive' : 'border-border')} />
                </label>
              ) : (
                <span className="text-xs text-muted-foreground pb-2">
                  runs from <code>{info?.repo}</code> in the <code>umi2</code> conda env
                </span>
              )}
              <span className="flex-1" />
              <Button onClick={generate} disabled={!dataset || !info}>
                <Terminal className="size-4" /> Generate command
              </Button>
            </div>
          </CardContent></Card>
        </Section>

        {cmd && (
          <Section n="3" title="Run it" right={
            <Button size="sm" variant="outline" onClick={copy}>
              {copied === 'ok' ? <Check className="size-4 text-[#22c55e]" /> : <Copy className="size-4" />}
              {copied === 'ok' ? 'Copied' : copied === 'fail' ? 'Select it and copy' : 'Copy'}
            </Button>}>
            <pre className="rounded-lg border border-border bg-black/60 p-4 text-xs font-mono overflow-x-auto select-all">{cmd}</pre>
            <p className="text-xs text-muted-foreground">
              Paste it into a terminal {mode === 'docker'
                ? <>on the machine running the <code>{container}</code> container, not inside it</>
                : 'on this machine'}.
              {' '}A checkpoint is written every {info?.defaults.checkpoint_every ?? 10} epochs; they appear under Checkpoints.
            </p>
          </Section>
        )}

        <Section title="Training on a remote machine?">
          <Card><CardContent className="pt-6 space-y-3 text-sm">
            <p className="text-muted-foreground">
              Start it inside <code>tmux</code>, so the run survives your SSH session closing.
            </p>
            <div className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 items-baseline">
              <code className="font-mono text-xs">tmux new -s umi_train</code><span className="text-muted-foreground">start a session</span>
              <span className="font-mono text-xs">paste the command</span><span className="text-muted-foreground">start training in it</span>
              <span className="font-mono text-xs"><kbd>Ctrl</kbd>+<kbd>b</kbd>, then <kbd>d</kbd></span>
              <span className="text-muted-foreground">detach; training keeps running after you disconnect</span>
              <code className="font-mono text-xs">tmux attach -t umi_train</code><span className="text-muted-foreground">attach back to it</span>
              <code className="font-mono text-xs">tmux ls</code><span className="text-muted-foreground">list sessions</span>
            </div>
          </CardContent></Card>
        </Section>

        <Runs />
      </div>
    </Page>
  )
}
