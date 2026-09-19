import { type Mode } from './data'
import { api, useApi } from './api'
import { IndiaMap } from './Map'
import { Icon } from './ui'

export function Landing({ onEnter }: { onEnter: (mode: Mode | 'sentinel') => void }) {
  const cities = useApi(api.cities, 'cities')
  const overview = useApi(() => api.overview(), 'overview')
  const o = overview.data
  return (
    <div className="relative h-full w-full overflow-y-auto">
      <div className="fixed inset-0">
        <IndiaMap cities={cities.data ?? []} />
        <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(105deg,rgba(242,239,231,0.94)_0%,rgba(242,239,231,0.86)_38%,rgba(242,239,231,0.5)_70%,rgba(242,239,231,0.25)_100%)]" />
      </div>

      <div className="relative z-10 flex min-h-full flex-col">
        {/* header */}
        <header className="flex shrink-0 items-center justify-between gap-4 px-8 py-5">
          <div className="flex items-center gap-3">
            <span className="grid size-8 shrink-0 place-items-center rounded-[4px] bg-[var(--color-accent)] text-[var(--color-on-accent)]">
              <Icon name="globe" className="size-5" />
            </span>
            <div className="leading-tight">
              <p className="font-mono text-[15px] font-semibold tracking-[0.14em] text-[var(--color-ink)]">FLOW STATE</p>
              <p className="eyebrow mt-0.5 text-[var(--color-mute-2)]">Satellite intelligence for India's urban waters</p>
            </div>
          </div>
          <span className="eyebrow hidden shrink-0 items-center gap-2 text-[var(--color-mute)] sm:flex">
            <span className={`size-1.5 rounded-full ${overview.error ? 'bg-[var(--color-risk-high)]' : 'bg-[var(--color-risk-low)] fs-pulse'}`} />
            {overview.error ? 'API offline' : o ? `${o.stations.toLocaleString()} CPCB stations · Sentinel-2 · ${o.date_range[0].slice(-4)}–${o.date_range[1].slice(-4)}` : 'Loading…'}
          </span>
        </header>

        {/* hero */}
        <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col justify-center gap-10 px-6 py-16">
        <p className="eyebrow text-[var(--color-accent)]">Water intelligence platform · Two modes, one map</p>
        <h1 className="display -mt-6 max-w-3xl text-5xl font-medium leading-[1.04] text-[var(--color-ink)] md:text-[3.4rem]">
          Flow State doesn't tell a city what to build.
          <span className="text-[var(--color-mute)]"> It tells them where they need to look.</span>
        </h1>

        {o?.model.roc_auc != null && (
          <p className="tnum -mt-4 max-w-3xl text-[12px] leading-relaxed text-[var(--color-mute)]">
            Screening model: AUC {o.model.roc_auc.toFixed(3)} for {o.model.target}, validated on unseen stations (site-blocked CV). The top 10% of ranked stations breach the limit {Math.round((o.model.precision_top_10pct ?? 0) * 100)}% of the time against a {Math.round((o.model.base_rate ?? 0) * 100)}% base rate. A triage tool, not a lab replacement.
          </p>
        )}

        <div className="grid gap-5 md:grid-cols-2">
          <ModeCard
            tone="var(--color-accent)"
            icon="city"
            title="PLAN & PREVENT"
            audience="For Municipalities & Urban Planners"
            desc="Explore an area, identify water-quality risks and understand how rivers, drains, sewage infrastructure and land use interact."
            cta="ENTER PLANNING MODE"
            onClick={() => onEnter('plan')}
          />
          <ModeCard
            tone="var(--color-risk-low)"
            icon="wave"
            title="RESTORE & PRIORITIZE"
            audience="For Water Authorities & Policymakers"
            desc="Find deteriorating water bodies, identify pollution hotspots and prioritize areas for intervention."
            cta="ENTER RESTORATION MODE"
            onClick={() => onEnter('restore')}
          />
        </div>

        <button
          onClick={() => onEnter('sentinel')}
          className="mt-6 flex w-fit items-center gap-2 text-[13px] text-[var(--color-mute)] transition-colors hover:text-[var(--color-accent)]"
        >
          <Icon name="globe" className="size-4" /> Explore Sentinel-2 <span className="text-[var(--color-mute-2)]">— raw satellite layer, no mode</span>
        </button>
        </main>
      </div>
    </div>
  )
}

function ModeCard({ tone, icon, title, audience, desc, cta, onClick }: { tone: string; icon: string; title: string; audience: string; desc: string; cta: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="group relative overflow-hidden rounded-[10px] border border-[var(--color-hair)] bg-[color-mix(in_oklab,var(--color-panel)_94%,transparent)] p-7 text-left backdrop-blur-md transition-all hover:-translate-y-0.5 hover:border-[var(--color-accent)]"
      style={{ boxShadow: '0 18px 44px -30px rgba(30,40,50,0.4)' }}
    >
      <span className="absolute inset-x-0 top-0 h-[3px]" style={{ background: tone }} />
      <div className="flex items-center gap-3.5">
        <span className="grid size-11 shrink-0 place-items-center rounded-[8px] border" style={{ color: tone, borderColor: `color-mix(in oklab, ${tone} 30%, transparent)`, background: `color-mix(in oklab, ${tone} 8%, transparent)` }}>
          <Icon name={icon} className="size-5" />
        </span>
        <div>
          <h2 className="text-[17px] font-semibold tracking-tight text-[var(--color-ink)]">{title}</h2>
          <p className="text-[12.5px] text-[var(--color-mute)]">{audience}</p>
        </div>
      </div>
      <p className="mt-5 text-[14px] leading-relaxed text-[var(--color-mute)]">{desc}</p>
      <span className="mt-6 flex items-center gap-2 font-mono text-[12px] font-semibold tracking-wider" style={{ color: tone }}>
        {cta}
        <Icon name="arrow" className="size-4 transition-transform group-hover:translate-x-1" />
      </span>
    </button>
  )
}
