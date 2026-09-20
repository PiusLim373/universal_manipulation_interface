import { Link } from 'react-router-dom'
import { Construction } from 'lucide-react'
import Page from '@/components/Page'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'

export default function Placeholder({ title, blurb, cli }) {
  return (
    <Page title={title} back="/">
      <div className="grid place-items-center">
        <Card className="max-w-lg">
          <CardContent className="pt-6 space-y-4">
            <Construction className="size-6 text-muted-foreground" />
            <div className="space-y-1">
              <h2 className="font-heading text-lg font-medium">{title}</h2>
              <p className="text-sm text-muted-foreground">{blurb}</p>
            </div>
            {cli && (
              <div className="space-y-1.5">
                <p className="text-xs text-muted-foreground">
                  Available from the command line today:
                </p>
                <pre className="text-[11px] bg-muted rounded-md p-3 overflow-x-auto">{cli}</pre>
              </div>
            )}
            <Link to="/"><Button size="sm" variant="outline">Back to start</Button></Link>
          </CardContent>
        </Card>
      </div>
    </Page>
  )
}
