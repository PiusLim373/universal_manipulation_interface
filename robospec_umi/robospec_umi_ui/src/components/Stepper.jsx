import { Check } from 'lucide-react'
import { cn } from '@/lib/utils'

/**
 * Wizard progress. `stages` is [{id, label, icon}]; `reached(id)` greys out the
 * stages not yet open. With `onPick`, reached stages become clickable.
 */
export default function Stepper({ stages, current, reached, onPick }) {
  const at = stages.findIndex((s) => s.id === current)
  return (
    <div className="flex items-center gap-1 flex-wrap">
      {stages.map((s, i) => {
        const Icon = s.icon
        const done = i < at
        const now = i === at
        const open = reached(s.id)
        const pick = onPick && open && !now
        return (
          <div key={s.id} className="flex items-center gap-1">
            <button
              type="button"
              disabled={!pick}
              onClick={pick ? () => onPick(s.id) : undefined}
              className={cn(
                'flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm transition-colors',
                now && 'bg-primary text-primary-foreground font-medium',
                done && 'text-[#22c55e]',
                !now && !done && 'text-muted-foreground',
                !open && 'opacity-40',
                pick ? 'hover:bg-muted cursor-pointer' : 'cursor-default',
              )}>
              {done ? <Check className="size-4" /> : <Icon className="size-4" />}
              {s.label}
            </button>
            {i < stages.length - 1 && <div className="w-5 h-px bg-border" />}
          </div>
        )
      })}
    </div>
  )
}
