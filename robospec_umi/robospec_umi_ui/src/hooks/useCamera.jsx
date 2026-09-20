import { useCallback, useEffect, useState } from 'react'
import { api } from '@/lib/api'

/**
 * Poll the camera broker.
 *
 * Camera ownership is global UI, not an error path: only one process can stream
 * a V4L2 device, so every page that wants a camera has to be able to say who
 * currently has it. Polling rather than SSE because this is a couple of small
 * fields every few seconds, and a dropped poll costs nothing.
 */
export function useCamera(intervalMs = 2500) {
  const [devices, setDevices] = useState({ scene: null, wrist: null })
  const [sessions, setSessions] = useState({})
  const [reachable, setReachable] = useState(true)

  const refresh = useCallback(async () => {
    try {
      const d = await api.camera()
      setDevices(d.devices || {})
      setSessions(d.sessions || {})
      setReachable(true)
    } catch {
      setReachable(false)
    }
  }, [])

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, intervalMs)
    return () => clearInterval(t)
  }, [refresh, intervalMs])

  return { devices, sessions, reachable, refresh }
}
