/**
 * Thin wrapper over the Python backend.
 *
 * Relative /api paths throughout: Vite proxies them in dev, and the aiohttp
 * server serves both the UI and the API from one origin in production. No base
 * URL constant, so there is nothing to get wrong between the two.
 */

export class ApiError extends Error {
  constructor(status, body) {
    super(body?.error || `HTTP ${status}`)
    this.status = status
    this.body = body || {}
    // 409 carries the current camera owner. Pages use it to explain who is
    // holding the device instead of showing a bare failure.
    this.owner = this.body.owner
  }
}

// raw: send body (a File) as-is instead of as JSON
async function req(method, path, body, raw = false) {
  const res = await fetch(path, {
    method,
    headers: body && !raw ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? (raw ? body : JSON.stringify(body)) : undefined,
  })
  const text = await res.text()
  let data = null
  try { data = text ? JSON.parse(text) : null } catch { data = { raw: text } }
  if (!res.ok) throw new ApiError(res.status, data)
  return data
}

export const api = {
  health:      ()            => req('GET',  '/api/health'),
  camera:      ()            => req('GET',  '/api/camera'),
  releaseCam:  (device)      => req('POST', '/api/camera/release', { device }),

  runs:        ()            => req('GET',  '/api/calibration/runs'),
  deleteRun:   (name)        => req('DELETE', `/api/calibration/runs/${encodeURIComponent(name)}`),
  activate:    (run)         => req('POST', '/api/calibration/activate', { run }),
  uploadIntrinsics: (file)   => req('POST', '/api/calibration/upload', file, true),
  active:      ()            => req('GET',  '/api/calibration/active'),
  runFile:     (run, file)   => `/api/calibration/runs/${encodeURIComponent(run)}/file/${encodeURIComponent(file)}`,

  // `device` is a LOGICAL camera name ('scene' | 'wrist'), not a /dev path.
  getControls: (device = 'scene') => req('GET', `/api/camera/controls?device=${device}`),
  setControls: (patch, device = 'scene') =>
    req('POST', '/api/camera/controls', { device, ...patch }),
  startTune:   (opts = {})   => req('POST', '/api/calibration/tune', opts),
  stopTune:    ()            => req('DELETE', '/api/calibration/tune'),
  tuneMask:    (on)          => req('POST', '/api/calibration/tune/mask', { on }),

  lock:        (opts = {})   => req('POST', '/api/calibration/lock', opts),
  sessionInfo: ()            => req('GET',  '/api/calibration/session'),
  startSession:(opts = {})   => req('POST', '/api/calibration/session', opts),
  resetSession:(keep = false)=> req('DELETE', `/api/calibration/session${keep ? '?keep=1' : ''}`),
  setAuto:     (on)          => req('POST', '/api/calibration/auto', { on }),
  keep:        ()            => req('POST', '/api/calibration/keep'),
  undo:        ()            => req('POST', '/api/calibration/undo'),

  solve:       (run)         => req('POST', '/api/calibration/solve', run ? { run } : {}),

  startTest:   (opts = {})   => req('POST', '/api/calibration/test', opts),
  stopTest:    ()            => req('DELETE', '/api/calibration/test'),

  captureInfo:   ()          => req('GET',  '/api/capture/session'),
  startCapture:  (opts = {}) => req('POST', '/api/capture/session', opts),
  stopCapture:   (discard=false) =>
    req('DELETE', `/api/capture/session${discard ? '?discard=1' : ''}`),
  episode:       (action)    => req('POST', '/api/capture/episode', { action }),
  finishCapture: ()          => req('POST', '/api/capture/finish'),
  capturePreview:(opts)      => req('POST', '/api/capture/preview', opts),

  // the capture test: one throwaway episode in the live session, never kept
  enterTest:     ()          => req('PUT',  '/api/capture/test'),
  leaveTest:     ()          => req('DELETE', '/api/capture/test'),
  resetTest:     ()          => req('POST', '/api/capture/test/reset'),
  testEpisode:   (action)    => req('POST', '/api/capture/test/episode', { action }),
  testReview:    ()          => req('GET',  '/api/capture/test/review'),
  testMedia:     (file, v)   => `/api/capture/test/media/${file}?v=${encodeURIComponent(v || '')}`,

  // edit: `key` is 'session/epNNN'
  editSessions:  ()          => req('GET',  '/api/edit/sessions'),
  projects:      ()          => req('GET',  '/api/edit/projects'),
  createProject: (sessions)  => req('POST', '/api/edit/projects', { sessions }),
  project:       (id)        => req('GET',  `/api/edit/projects/${id}`),
  saveProject:   (id, patch) => req('PUT',  `/api/edit/projects/${id}`, patch),
  deleteProject: (id)        => req('DELETE', `/api/edit/projects/${id}`),
  verifyProject: (id)        => req('POST', `/api/edit/projects/${id}/verify`),
  rebindProject: (id)        => req('POST', `/api/edit/projects/${id}/rebind`),
  prep:          (id, focus) => req('POST', `/api/edit/projects/${id}/prep`, focus ? { focus } : {}),
  exportProject: (id)        => req('POST', `/api/edit/projects/${id}/export`),
  timeline:      (key)       => req('GET',  `/api/edit/episode/${key}/timeline`),
  plan:          (key, trim) =>
    req('GET', `/api/edit/episode/${key}/plan?in=${trim?.[0] ?? ''}&out=${trim?.[1] ?? ''}`),
  media:         (key, file, v) => `/api/edit/media/${key}/${file}?v=${encodeURIComponent(v || '')}`,

  // dataset files in data/dataset/
  datasets:      ()          => req('GET',  '/api/datasets'),
  deleteDataset: (file)      => req('DELETE', `/api/datasets/${encodeURIComponent(file)}`),
  datasetUrl:    (file)      => `/api/datasets/${encodeURIComponent(file)}`,

  // training
  trainInfo:     ()          => req('GET',  '/api/train/info'),
  gpu:           ()          => req('GET',  '/api/train/gpu'),
  trainRuns:     ()          => req('GET',  '/api/train/runs'),
  ckptUrl:       (path)      => `/api/train/ckpt?path=${encodeURIComponent(path)}`,

  // evaluation without a robot: `path` is a checkpoint relative to data/outputs/
  evalCheckpoints: ()        => req('GET',  '/api/eval/checkpoints'),
  evalCheckpoint:  (path)    => req('GET',  `/api/eval/checkpoint?path=${encodeURIComponent(path)}`),
  evalRun:   (path, episode, force = false) => req('POST', '/api/eval/run', { path, episode, force }),
  evalResult:      (path, episode) =>
    req('GET', `/api/eval/result?path=${encodeURIComponent(path)}&episode=${encodeURIComponent(episode)}`),
}

/** Upload one file into data/dataset/. */
export const uploadDataset = (file, onProgress) => uploadFile('/api/datasets/upload', file, onProgress)

/** Upload a checkpoint; the server files it under data/outputs/<its run>/checkpoints/. */
export const uploadCheckpoint = (file, onProgress) => uploadFile('/api/eval/upload', file, onProgress)

/** Raw-body upload. XHR, because fetch reports no upload progress. */
function uploadFile(url, file, onProgress) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest()
    x.open('POST', `${url}?name=${encodeURIComponent(file.name)}`)
    x.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded, e.total)
    x.onload = () => {
      let data
      try { data = JSON.parse(x.responseText) } catch { data = { raw: x.responseText } }
      if (x.status >= 200 && x.status < 300) resolve(data)
      else reject(new ApiError(x.status, data))
    }
    x.onerror = () => reject(new ApiError(0, { error: 'connection lost during upload' }))
    x.send(file)
  })
}

/**
 * The two Preview-and-Lock configurations.
 *
 * Module-level constants, never inline literals: PreviewLock holds this in a ref
 * and an unstable identity would restart the camera session on every render.
 *
 * The one difference that matters is `stopOnUnmount`. Calibration's tune session
 * belongs to the stage and is torn down on the way out; capture's session spans
 * BOTH stages -- it is what stage 2 records with -- so leaving the page must not
 * close it.
 */
export const CALIB_PREVIEW = {
  cams: ['scene'],
  start:        () => api.startTune({ mask: false }),
  stopOnUnmount:() => api.stopTune(),
  stopOnDone:   () => api.stopTune(),
  stream:       () => '/api/calibration/tune/stream',
  state:        '/api/calibration/tune/state',
  setMask:      (on) => api.tuneMask(on),
  getControls:  (cam) => api.getControls(cam),
  setControls:  (cam, patch) => api.setControls(patch, cam),
  lock:         () => api.lock(),
  statsOf:      (s) => s,                    // the whole payload IS the scene stats
  cta:          'Continue to capture',
}

export const CAPTURE_PREVIEW = {
  cams: ['scene', 'wrist'],
  start:        () => api.startCapture(),
  stopOnUnmount: null,                       // the session spans both stages
  stopOnDone:   null,
  stream:       (cam) => `/api/capture/stream/${cam}`,
  state:        '/api/capture/state',
  setMask:      (on) => api.capturePreview({ annotate: true, mask: on }),
  getControls:  (cam) => api.getControls(cam),
  setControls:  (cam, patch) => api.setControls(patch, cam),
  lock:         () => api.lock(),
  statsOf:      (s, cam) => s?.stats?.[cam],
  cta:          'Continue to capture test',
}

/**
 * Subscribe to an SSE endpoint. Returns an unsubscribe function.
 *
 * EventSource reconnects on its own, which is what makes a mid-capture browser
 * refresh reattach to the running session rather than losing it.
 */
export function sse(path, onMessage, onError) {
  const es = new EventSource(path)
  es.onmessage = (e) => {
    try { onMessage(JSON.parse(e.data)) } catch { /* ignore a partial frame */ }
  }
  if (onError) es.onerror = onError
  return () => es.close()
}
