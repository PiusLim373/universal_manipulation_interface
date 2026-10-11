import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'

const COLOUR = { good: '#22c55e', ok: '#f59e0b', bad: '#ef4444', fail: '#ef4444' }

/** The capture test's score from verify.score_episode: the number, the verdict
 *  and one row per metric, with what to change where a metric is not good. */
export default function TestScore({ s }) {
  const [open, setOpen] = useState(false)
  const c = COLOUR[s.verdict]
  // shrinks and scrolls, so the buttons around it stay on screen
  return (
    <Card className="min-h-0"><CardContent className="pt-6 space-y-3 min-h-0 overflow-y-auto">
      <div className="flex items-center gap-3">
        <span className="text-4xl font-heading tabular-nums" style={{ color: c }}>{s.score}</span>
        <div className="space-y-0.5">
          <p className="text-sm font-medium" style={{ color: c }}>{s.label}</p>
          <p className="text-[11px] text-muted-foreground">{s.info.join(' · ')}</p>
        </div>
      </div>
      {s.fails.map((f) => (
        <p key={f} className="text-xs text-destructive">verify: {f}</p>
      ))}
      <div className="space-y-1.5">
        {s.metrics.map((m) => (
          <div key={m.key} className="text-xs">
            <div className="flex items-baseline gap-2">
              <span className="size-2 rounded-full shrink-0 self-center" style={{ background: COLOUR[m.grade] }} />
              <span>{m.label}</span>
              <span className="flex-1" />
              <span className="tabular-nums text-muted-foreground">{m.value}</span>
            </div>
            {m.hint && <p className="pl-4 text-[11px] text-muted-foreground">{m.hint}</p>}
          </div>
        ))}
      </div>
      {s.warnings.length > 0 && (
        <p className="text-[11px] text-[#f59e0b]">{s.warnings.join(' · ')}</p>
      )}
      <button type="button" onClick={() => setOpen((o) => !o)}
              className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground">
        {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
        verify report
      </button>
      {open && (
        <pre className={cn('text-[10px] leading-snug font-mono whitespace-pre-wrap break-all',
                           'max-h-64 overflow-auto bg-muted/40 rounded p-2')}>
          {s.lines.join('\n').trim()}
        </pre>
      )}
    </CardContent></Card>
  )
}
