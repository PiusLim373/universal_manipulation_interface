import { cn } from '@/lib/utils'

/**
 * The three coverage meters, the same ones the cv2 overlay draws.
 *
 * Kept as three independent meters rather than one product of bins, because a
 * rectilinear lens fails three separate ways and a combined score would never
 * fill while telling you nothing about what is actually missing:
 *   grid   distortion grows with radius, so the frame corners must be visited
 *   scale  near and far views break the focal-length/distance correlation
 *   tilt   all-fronto-parallel views leave the solve ill-conditioned
 */

function Bar({ name, have, need }) {
  const frac = Math.min(have / need, 1)
  const done = frac >= 1
  return (
    <div className="flex items-center gap-2">
      <span className="w-12 text-[11px] text-muted-foreground">{name}</span>
      <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
        <div className={cn('h-full rounded-full transition-all',
                           done ? 'bg-[#22c55e]' : 'bg-primary')}
             style={{ width: `${frac * 100}%` }} />
      </div>
      <span className={cn('w-10 text-right text-[11px] tabular-nums',
                          done ? 'text-[#22c55e]' : 'text-muted-foreground')}>
        {have}/{need}
      </span>
    </div>
  )
}

export default function CoverageMeters({ coverage }) {
  if (!coverage) return null
  const { grid, grid_nx, grid_ny, target, per_bin,
          scale, scale_names, tilt, tilt_names, advice, done } = coverage

  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-baseline justify-between mb-1.5">
          <span className="text-xs font-medium">frame coverage</span>
          <span className="text-[11px] text-muted-foreground">
            {coverage.grid_done}/{coverage.grid_total} cells
          </span>
        </div>
        {/* Corner hits per cell, not frames -- this is where corners actually
            landed, which is what constrains distortion. */}
        <div className="grid gap-1"
             style={{ gridTemplateColumns: `repeat(${grid_nx}, minmax(0,1fr))` }}>
          {Array.from({ length: grid_ny }).flatMap((_, y) =>
            Array.from({ length: grid_nx }).map((_, x) => {
              const hits = grid?.[y]?.[x] ?? 0
              const frac = Math.min(hits / target, 1)
              return (
                <div key={`${y}-${x}`}
                     title={`${hits}/${target} corner hits`}
                     className="aspect-[4/3] rounded-sm border border-border/60 transition-colors"
                     style={{
                       background: frac >= 1
                         ? 'oklch(0.700 0.158 258 / 0.55)'
                         : `oklch(0.700 0.158 258 / ${0.06 + frac * 0.40})`,
                     }} />
              )
            }))}
        </div>
      </div>

      <div className="space-y-1.5">
        <span className="text-xs font-medium">scale</span>
        {scale_names?.map((n, i) => <Bar key={n} name={n} have={scale[i]} need={per_bin} />)}
      </div>
      <div className="space-y-1.5">
        <span className="text-xs font-medium">tilt</span>
        {tilt_names?.map((n, i) => <Bar key={n} name={n} have={tilt[i]} need={per_bin} />)}
      </div>

      <div className={cn('rounded-md px-3 py-2 text-sm',
                         done ? 'bg-[#22c55e]/10 text-[#22c55e]'
                              : 'bg-primary/10 text-foreground')}>
        {done ? 'Coverage complete — ready to solve' : advice}
      </div>
    </div>
  )
}
