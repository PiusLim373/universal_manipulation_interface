import { useEffect, useRef, useState } from 'react'
import { Lightbulb, Info } from 'lucide-react'
import { cn } from '@/lib/utils'
import { SCHEMAS, WRIST_SCHEMAS } from '@/lib/controlSchemas'

/**
 * Photometric sliders for one camera, driven by a schema.
 *
 * Which controls exist, and what to say about them, lives in lib/controlSchemas.
 * Whether a control is writable comes from the server's `adjustable` flag rather
 * than being re-derived here — both backends already work it out, and the one
 * that owns the camera is the one that knows.
 *
 * Sliders stop at the MEASURED useful ceiling rather than the hardware maximum,
 * because past it the image stops changing and the extra travel is a slider that
 * does nothing.
 */

// A drag fires onChange on every pixel of travel. Each one is a v4l2-ctl
// subprocess or an rs.set_option on a camera that must be serviced every 11.1 ms,
// so send only the value the user settles on. The thumb still tracks the finger
// because the displayed value is local.
const SETTLE_MS = 80

function Slider({ label, name, ctrl, hint, onChange }) {
  if (!ctrl) return null
  const max = ctrl.useful_max ?? ctrl.max
  const disabled = ctrl.adjustable === false
  const [local, setLocal] = useState(null)
  const timer = useRef(null)
  useEffect(() => () => clearTimeout(timer.current), [])

  const shown = local ?? Math.min(ctrl.value ?? ctrl.min, max)
  const push = (v) => {
    setLocal(v)
    clearTimeout(timer.current)
    timer.current = setTimeout(() => { setLocal(null); onChange(name, v) }, SETTLE_MS)
  }

  return (
    <div className={cn('space-y-1.5', disabled && 'opacity-40')}>
      <div className="flex items-baseline justify-between">
        <label className="text-xs font-medium">{label}</label>
        <span className="text-[11px] tabular-nums text-muted-foreground">
          {ctrl.stale ? '— driven by the camera' : shown}
          {!ctrl.stale && ctrl.useful_max && ctrl.useful_max < ctrl.max && (
            <span className="ml-1 opacity-60">/ {max} (hw {ctrl.max})</span>
          )}
        </span>
      </div>
      <input
        type="range"
        min={ctrl.min}
        max={max}
        step={ctrl.step || 1}
        value={shown}
        disabled={disabled}
        onChange={(e) => push(Number(e.target.value))}
        className="w-full accent-[oklch(0.700_0.158_258.0)] disabled:cursor-not-allowed"
      />
      {hint && <p className="text-[10px] text-muted-foreground leading-snug">{hint}</p>}
    </div>
  )
}

function Toggle({ label, on, onChange, hint }) {
  return (
    <div className="space-y-1">
      <button
        type="button"
        onClick={(e) => { e.currentTarget.blur(); onChange(!on) }}
        className={cn(
          'w-full flex items-center justify-between rounded-md border px-3 py-2 text-sm transition-colors',
          on ? 'border-primary/50 bg-primary/10 text-foreground'
             : 'border-border text-muted-foreground hover:text-foreground',
        )}>
        <span>{label}</span>
        <span className={cn('inline-flex h-4 w-7 items-center rounded-full transition-colors',
                            on ? 'bg-primary' : 'bg-muted')}>
          <span className={cn('h-3 w-3 rounded-full bg-background transition-transform',
                              on ? 'translate-x-3.5' : 'translate-x-0.5')} />
        </span>
      </button>
      {hint && <p className="text-[10px] text-muted-foreground leading-snug">{hint}</p>}
    </div>
  )
}

export default function ControlPanel({ cam = 'scene', controls, onSet,
                                       mask, onMask, mean }) {
  if (!controls) return <p className="text-sm text-muted-foreground">reading camera…</p>
  const schema = (cam === 'wrist' && WRIST_SCHEMAS[controls._model]) || SCHEMAS[cam] || SCHEMAS.scene
  const autoExp = controls._auto_exposure

  // Out of road: still dark with both light-collecting controls at their
  // measured ceiling. Gamma would brighten the picture, but it is applied after
  // digitisation, so it amplifies the noise along with the signal rather than
  // collecting more light.
  const [expName, gainName] = schema.brightnessAxes
  const exp = controls[expName]
  const gain = controls[gainName]
  const maxedOut = !autoExp && mean != null && mean < 40 && exp && gain &&
    exp.value >= (exp.useful_max ?? exp.max) &&
    gain.value >= (gain.useful_max ?? gain.max)

  return (
    <div className="space-y-4">
      {schema.banner && (
        <div className="rounded-md bg-muted/50 text-muted-foreground px-2.5 py-2
                        text-[11px] flex items-start gap-1.5">
          <Info className="size-3.5 mt-0.5 shrink-0" />
          <span>{schema.banner}</span>
        </div>
      )}

      {schema.groups.map((g, i) => {
        const t = g.toggle
        const on = t ? controls[t.name]?.value === t.on : false
        return (
          <div key={i} className="space-y-4">
            {i > 0 && <div className="h-px bg-border" />}
            {t && (
              <Toggle label={t.label} on={on} hint={t.hint}
                      onChange={(v) => onSet(t.name, v ? t.on : t.off)} />
            )}
            {g.sliders.map((s) => {
              const ctrl = controls[s.name]
              const driven = ctrl && ctrl.adjustable === false
              return (
                <Slider key={s.name} label={s.label} name={s.name} ctrl={ctrl}
                        onChange={onSet}
                        hint={driven ? (s.hintWhenAuto || s.hint) : s.hint} />
              )
            })}
          </div>
        )
      })}

      {maxedOut && (
        <div className="rounded-md bg-[#f59e0b]/10 text-[#f59e0b] px-2.5 py-2
                        text-[11px] flex items-start gap-1.5">
          <Lightbulb className="size-3.5 mt-0.5 shrink-0" />
          <span>
            Exposure and gain are both at maximum and the image is still dark.
            Add light to the scene rather than raising gamma — gamma amplifies
            noise instead of collecting photons.
          </span>
        </div>
      )}

      {onMask && (
        <>
          <div className="h-px bg-border" />
          <Toggle label="Clipping mask" on={mask} onChange={onMask}
                  hint="red = blown, blue = crushed. Blown highlights destroy the gradients subpixel corner refinement needs." />
        </>
      )}
    </div>
  )
}
