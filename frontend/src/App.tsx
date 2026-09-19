import { useState } from 'react'
import { Landing } from './flow/Landing'
import { Workspace } from './flow/Workspace'
import type { Mode } from './flow/data'

type Route = { name: 'landing' } | { name: 'workspace'; view: Mode | 'sentinel' }

export default function App() {
  const [route, setRoute] = useState<Route>({ name: 'landing' })

  return (
    <div className="h-screen w-screen overflow-hidden">
      {route.name === 'landing' ? (
        <Landing onEnter={(view) => setRoute({ name: 'workspace', view })} />
      ) : (
        <Workspace
          view={route.view}
          onHome={() => setRoute({ name: 'landing' })}
          onSwitch={(view) => setRoute({ name: 'workspace', view })}
        />
      )}
    </div>
  )
}
