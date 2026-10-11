import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

/**
 * Merge class names (clsx + tailwind-merge).
 */
export function cn(...inputs) {
  return twMerge(clsx(inputs))
}

/** Megabytes -> "44 KB" / "199.9 MB" / "1.37 GB". */
export function fmtMB(mb) {
  if (mb == null) return '—'
  if (mb < 1) return `${Math.round(mb * 1000)} KB`
  return mb < 1000 ? `${mb.toFixed(1)} MB` : `${(mb / 1000).toFixed(2)} GB`
}
