import { Link } from 'react-router-dom'
import { Aperture, Video, Scissors, Brain, FlaskConical } from 'lucide-react'
import logo from '/robospec_logo.png'
import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'

const FEATURES = [
  { id: 'calibration', to: '/calibration', icon: Aperture, title: 'Calibration',
    desc: 'Measure the scene camera’s intrinsics from a printed ChArUco board.',
    ready: true },
  { id: 'capture', to: '/capture', icon: Video, title: 'Capture',
    desc: 'Record synchronised scene and wrist demonstrations.',
    ready: true },
  { id: 'edit', to: '/edit', icon: Scissors, title: 'Edit dataset',
    desc: 'Verify, trim and pick episodes, then export the training dataset.',
    ready: true },
  { id: 'training', to: '/training', icon: Brain, title: 'Training',
    desc: 'Generate the training command and download the checkpoints.',
    ready: true },
  { id: 'evaluation', to: '/evaluation', icon: FlaskConical, title: 'Evaluation',
    desc: 'Score a checkpoint against recorded data, with no robot attached.' },
]

export default function Home() {
  return (
    <div className="h-screen overflow-auto bg-background bg-dotgrid">
      {/* min-h-full + place-items-center: centred when it fits, scrolls when it
          does not, rather than clipping the top on a short window. */}
      <div className="min-h-full grid place-items-center px-6 py-12">
        <div className="flex flex-col items-center gap-10 w-full max-w-5xl">
          <img src={logo} alt="Robospec" className="h-12 select-none"
               style={{ filter: 'drop-shadow(0 0 14px oklch(0.7 0.158 258 / 0.45))' }} />

          <div className="text-center space-y-1">
            <h1 className="text-2xl font-semibold tracking-tight">What would you like to do?</h1>
            <p className="text-sm text-muted-foreground">
              The pipeline runs in order — calibrate before you capture, capture before you train.
            </p>
          </div>

          {/* Six columns, each tile spanning two, and the fourth starting at
              column 2. That puts the second row's two tiles centred under the
              first row's three instead of left-aligned beneath them. */}
          <div className="grid grid-cols-1 md:grid-cols-6 gap-4 w-full">
            {FEATURES.map(({ id, to, icon: Icon, title, desc, ready }, i) => (
              <Link key={id} to={to}
                    className={cn('block md:col-span-2', i === 3 && 'md:col-start-2')}>
                <Card className={cn(
                  'h-full transition-all duration-200 hover:-translate-y-1',
                  'hover:shadow-[0_10px_32px_-8px_oklch(0.7_0.158_258_/_0.35)]',
                  !ready && 'opacity-40',
                )}>
                  <CardContent className="pt-6 space-y-2">
                    <Icon className={cn('size-6', ready ? 'text-primary' : 'text-muted-foreground')} />
                    <div className="font-heading font-medium">{title}</div>
                    <p className="text-sm text-muted-foreground leading-snug">{desc}</p>
                    {!ready && (
                      <span className="inline-block text-[10px] uppercase tracking-wide
                                       text-muted-foreground border border-border rounded px-1.5 py-0.5">
                        next phase
                      </span>
                    )}
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
