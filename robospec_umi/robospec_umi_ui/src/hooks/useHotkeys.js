import { useEffect, useRef } from 'react'

// Only text entry is exempt. The obvious guard is "skip every INPUT", but this
// app is full of range sliders -- nudge one, press SPACE, and nothing happens.
// A silently dead key is worse than a stolen one.
const TEXT_TYPES = new Set(['text', 'number', 'search', 'email', 'password',
                            'url', 'tel', 'date', 'time'])

/**
 * Window-level keyboard shortcuts.
 *
 * `map` is read through a ref so the handler identity stays stable: StrictMode
 * double-invokes effects, and a handler that changed every render would add and
 * remove the listener constantly and fire twice in dev.
 *
 * preventDefault() on keydown is what stops a focused <button> also firing --
 * Enter activates a button on keydown and Space on keyup, and cancelling the
 * keydown suppresses both, along with Space scrolling the page.
 *
 * Auto-repeat is ignored except for the keys listed in `repeat`.
 */
export default function useHotkeys(map, enabled = true, repeat = null) {
  const ref = useRef(map)
  ref.current = map
  const rep = useRef(repeat)
  rep.current = repeat

  useEffect(() => {
    if (!enabled) return undefined
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if (e.repeat && !rep.current?.includes(e.key)) return
      const t = e.target
      if (t?.isContentEditable) return
      const tag = t?.tagName
      if (tag === 'TEXTAREA' || tag === 'SELECT') return
      if (tag === 'INPUT' && TEXT_TYPES.has((t.type || 'text').toLowerCase())) return
      const fn = ref.current[e.key]
      if (!fn) return
      e.preventDefault()
      fn(e)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [enabled])
}
