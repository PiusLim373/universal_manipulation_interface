import Header from '@/components/Header'

/**
 * One page shell, two modes.
 *
 * `centred` (default) vertically centres content that does not fill the
 * viewport. Top-aligned content under a band of empty space reads as broken
 * rather than as spacious.
 *
 * `fill` gives the content region the remaining height instead, for pages whose
 * job is a video feed — there the stream should take the room, not float in it.
 */
export default function Page({ title, back, fill = false, wide = false, children }) {
  return (
    <div className="h-screen flex flex-col bg-background overflow-hidden">
      {(title || back) && <Header title={title} back={back} />}
      {fill ? (
        <div className={`flex-1 min-h-0 p-6 w-full mx-auto flex flex-col
                         ${wide ? 'max-w-[110rem]' : 'max-w-6xl'}`}>
          {children}
        </div>
      ) : (
        <div className="flex-1 min-h-0 overflow-auto">
          <div className={`min-h-full grid place-items-center p-6 w-full mx-auto
                           ${wide ? 'max-w-[110rem]' : 'max-w-5xl'}`}>
            <div className="w-full py-8">{children}</div>
          </div>
        </div>
      )}
    </div>
  )
}
