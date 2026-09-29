import { useRef, useState } from 'react'
import { Upload, Loader2, X, CheckCircle2, XCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { uploadDataset } from '@/lib/api'
import { fmtMB } from '@/lib/utils'

const isZip = (f) => f.name.endsWith('.zarr.zip')
const isJson = (f) => f.name.endsWith('.json')
const mb = (b) => b / 1e6

/**
 * Upload .zarr.zip files and, optionally, their <id>_dataset.json. Zips go
 * first: the server takes a project json only once its zip is there.
 * -> { button, panel } so the page places each.
 */
export function useDatasetUpload(onDone, { size = 'sm', variant = 'outline' } = {}) {
  const input = useRef(null)
  const [items, setItems] = useState([])
  const busy = items.some((i) => i.status === 'wait' || i.status === 'up')
  const set = (k, patch) => setItems((xs) => xs.map((x, i) => (i === k ? { ...x, ...patch } : x)))

  const start = async (files) => {
    const all = [...files]
    const order = [...all.filter(isZip), ...all.filter((f) => !isZip(f) && isJson(f))]
    const bad = all.filter((f) => !isZip(f) && !isJson(f))
    setItems([...order.map((f) => ({ name: f.name, loaded: 0, total: f.size, status: 'wait' })),
              ...bad.map((f) => ({ name: f.name, status: 'err', err: 'not a .zarr.zip or a .json' }))])
    input.current.value = ''
    for (let k = 0; k < order.length; k++) {
      set(k, { status: 'up' })
      try {
        await uploadDataset(order[k], (loaded, total) => set(k, { loaded, total }))
        set(k, { status: 'done', loaded: order[k].size })
      } catch (e) { set(k, { status: 'err', err: e.message }) }
    }
    onDone?.()
  }

  const button = (
    <>
      <input ref={input} type="file" multiple accept=".zip,.json" className="hidden"
             onChange={(e) => e.target.files.length && start(e.target.files)} />
      <Button size={size} variant={variant} disabled={busy} onClick={() => input.current.click()}
              title="a .zarr.zip, optionally with its <id>_dataset.json">
        {busy ? <Loader2 className="size-4 animate-spin" /> : <Upload className="size-4" />}
        Upload dataset
      </Button>
    </>
  )

  const panel = items.length > 0 && (
    <div className="rounded-md border border-border px-3 py-2 space-y-2 text-sm">
      {items.map((it) => (
        <div key={it.name} className="space-y-1">
          <div className="flex items-center gap-2">
            {it.status === 'done' && <CheckCircle2 className="size-4 text-[#22c55e] shrink-0" />}
            {it.status === 'err' && <XCircle className="size-4 text-destructive shrink-0" />}
            {(it.status === 'up' || it.status === 'wait') && (
              <Loader2 className={`size-4 shrink-0 ${it.status === 'up' ? 'animate-spin' : 'opacity-30'}`} />
            )}
            <span className="font-mono text-xs truncate flex-1">{it.name}</span>
            <span className="text-xs text-muted-foreground tabular-nums">
              {it.status === 'err' ? '' : it.status === 'done' ? fmtMB(mb(it.total))
                : `${fmtMB(mb(it.loaded))} / ${fmtMB(mb(it.total))} · ${Math.floor((100 * it.loaded) / (it.total || 1))}%`}
            </span>
          </div>
          {it.status === 'up' && (
            <div className="h-1 rounded bg-muted overflow-hidden">
              <div className="h-full bg-primary transition-[width]"
                   style={{ width: `${(100 * it.loaded) / (it.total || 1)}%` }} />
            </div>
          )}
          {it.err && <div className="text-xs text-destructive pl-6">{it.err}</div>}
        </div>
      ))}
      {!busy && (
        <div className="flex justify-end">
          <Button size="xs" variant="ghost" onClick={() => setItems([])}><X className="size-3" /> Dismiss</Button>
        </div>
      )}
    </div>
  )

  return { button, panel }
}
