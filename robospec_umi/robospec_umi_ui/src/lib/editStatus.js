// Index = status code from timeline.py (ok, lost, spike, outside, pose-gap, no-wrist).
export const STATUS_COLOR = ['#22c55e', '#ef4444', '#ef4444', '#52525b', '#f59e0b', '#f59e0b']
export const STATUS_LEGEND = [
  ['#22c55e', 'tracked'], ['#ef4444', 'lost / spike'],
  ['#f59e0b', 'pose gap / no wrist frame'], ['#52525b', 'outside overlap'],
]
