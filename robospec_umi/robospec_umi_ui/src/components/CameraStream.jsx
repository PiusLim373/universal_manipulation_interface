import { useEffect, useRef, useState } from 'react'
import { VideoOff } from 'lucide-react'

/**
 * An MJPEG stream in an <img>.
 *
 * The browser does the whole job: one GET that never completes, the server
 * writes multipart parts, each replaces the last. No canvas, no decode loop,
 * no polling. The only code needed is the reconnect, because a dropped stream
 * leaves the element blank and it will not retry on its own -- and the
 * cache-buster matters, or the browser may not re-request the same URL.
 */
export default function CameraStream({ src, alt = 'camera', className = '' }) {
  const [url, setUrl] = useState(src)
  const [failed, setFailed] = useState(false)
  const img = useRef(null)

  useEffect(() => { setUrl(`${src}?t=${Date.now()}`); setFailed(false) }, [src])

  // A removed <img> keeps its stream open until it is garbage-collected, and the
  // browser allows 6 connections per host: a few remounts and every request on
  // the page queues behind dead streams. End it on unmount.
  useEffect(() => {
    const el = img.current
    return () => el?.removeAttribute('src')
  }, [])

  // The border and background live on the IMG, not on a wrapper. A bordered
  // wrapper filling the grid cell would show its own background wherever the
  // cell is not 16:9 -- which reads as the video having black bars, when really
  // it is the box being the wrong shape. With the frame on the image itself,
  // the visible box is exactly the picture.
  //
  // w-full + aspect-video + max-h-full is what makes it FILL rather than sit at
  // its natural size: an <img> is never scaled up past its intrinsic pixels by
  // max-* alone. Width drives the box, the ratio derives the height, and when
  // max-height binds the browser recomputes width from the same ratio -- so it
  // grows to fit either axis without ever letterboxing.
  return (
    <div className={`relative flex items-center justify-center min-h-0 min-w-0 ${className}`}>
      <img
        ref={img}
        src={url}
        alt={alt}
        className="w-full max-w-full max-h-full aspect-video object-contain
                   rounded-lg border border-border bg-black"
        onError={() => {
          setFailed(true)
          setTimeout(() => { setUrl(`${src}?t=${Date.now()}`); setFailed(false) }, 1200)
        }}
      />
      {failed && (
        <div className="absolute inset-0 grid place-items-center text-muted-foreground gap-2">
          <VideoOff className="size-6" />
          <span className="text-xs">stream lost — reconnecting</span>
        </div>
      )}
    </div>
  )
}
