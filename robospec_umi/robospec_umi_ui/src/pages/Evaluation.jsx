import { Link } from 'react-router-dom'
import { Film, Bot } from 'lucide-react'
import Page from '@/components/Page'
import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'

const MODES = [
  { id: 'offline', to: '/evaluation/offline', icon: Film, title: 'Eval Without Robot', ready: true,
    desc: 'Run a checkpoint along a recording and see its forecasts over what the hand actually did.' },
  { id: 'robot', icon: Bot, title: 'Eval With Robot',
    desc: 'Run a checkpoint on the robot, closing the loop on its own actions.' },
]

export default function Evaluation() {
  return (
    <Page title="Evaluation" back="/">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-3xl mx-auto">
        {MODES.map(({ id, to, icon: Icon, title, desc, ready }) => {
          const card = (
            <Card className={cn('h-full transition-all duration-200',
              ready ? 'hover:-translate-y-1 hover:shadow-[0_10px_32px_-8px_oklch(0.7_0.158_258_/_0.35)]'
                : 'opacity-40')}>
              <CardContent className="pt-6 space-y-2">
                <Icon className={cn('size-6', ready ? 'text-primary' : 'text-muted-foreground')} />
                <div className="font-heading font-medium">{title}</div>
                <p className="text-sm text-muted-foreground leading-snug">{desc}</p>
                {!ready && (
                  <span className="inline-block text-[10px] uppercase tracking-wide
                                   text-muted-foreground border border-border rounded px-1.5 py-0.5">
                    coming soon
                  </span>
                )}
              </CardContent>
            </Card>
          )
          return ready ? <Link key={id} to={to} className="block">{card}</Link>
            : <div key={id} className="cursor-not-allowed" aria-disabled>{card}</div>
        })}
      </div>
    </Page>
  )
}
