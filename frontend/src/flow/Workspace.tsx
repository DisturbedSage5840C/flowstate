import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ASK_CHIPS, LAYER_GROUPS, RISK, riskWord, type Mode, type Risk } from './data'
import {
  api,
  useApi,
  type Alert,
  type AskResult,
  type Overview,
  type Priority,
  type Rec,
  type SearchHit,
  type Station,
  type StationDetail,
} from './api'
import { SatelliteMap, WaterMap, type Basemap, type ColorBy, type FlyTarget } from './Map'
import {
  DistanceRow,
  GuidancePill,
  Icon,
  InfraChain,
  LegendChip,
  RecommendationCard,
  RiskBadge,
  TrendCell,
  WhyDisclosure,
} from './ui'

type View = Mode | 'sentinel'
type Filter = { label: string; ids: Set<string> }
type FieldItem = { id: string; name: string; state: string; risk: Risk }

const FIELD_KEY = 'flowstate.fieldlist'

/* Field-investigation shortlist: a per-browser convenience, so localStorage may be unavailable */
function useFieldList() {
  const [items, setItems] = useState<FieldItem[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(FIELD_KEY) ?? '[]')
    } catch {
      return []
    }
  })
  const save = (next: FieldItem[]) => {
    setItems(next)
    try {
      localStorage.setItem(FIELD_KEY, JSON.stringify(next))
    } catch {
      /* ignore */
    }
  }
  return {
    items,
    has: (id: string) => items.some((i) => i.id === id),
    toggle: (d: FieldItem) => save(items.some((i) => i.id === d.id) ? items.filter((i) => i.id !== d.id) : [...items, d]),
    clear: () => save([]),
  }
}

const fmtReading = (v: number | null, unit: string) => (v == null ? 'no data' : `${v} ${unit}`)

function legendNote(o: Overview | null, year: number | null, colorBy: ColorBy) {
  if (year != null) return `${year}: measured water-quality index for that year. High ≥ 60, moderate 40–60. Not the screening model.`
  switch (colorBy) {
    case 'overall':
      return o ? `Screening model: high = P(BOD > 3 mg/L) ≥ ${o.thresholds.high * 100}%, moderate ≥ ${o.thresholds.mod * 100}%.` : undefined
    case 'bod':
      return 'Latest measured BOD: high > 6, moderate > 3 mg/L (CPCB). Grey = not measured.'
    case 'do':
      return 'Latest measured DO: high < 4, moderate < 5 mg/L (CPCB). Grey = not measured.'
    case 'turb':
      return 'Latest measured turbidity: high > 25, moderate > 5 NTU (IS 10500). Grey = not measured.'
    default:
      return 'Red = water-quality index worse than earlier visits, green = better, grey = stable or single visit.'
  }
}

export function Workspace({ view, onHome, onSwitch }: { view: View; onHome: () => void; onSwitch: (v: View) => void }) {
  const overview = useApi(() => api.overview(), 'overview')
  const cities = useApi(api.cities, 'cities')
  const [yearIdx, setYearIdx] = useState(0) // 0 = all years (screening); i > 0 = overview.years[i - 1]
  const year = yearIdx === 0 ? null : (overview.data?.years[yearIdx - 1] ?? null)
  const stations = useApi(() => api.stations(year ?? undefined), year)

  const [selected, setSelected] = useState<string | null>(null)
  const [secondary, setSecondary] = useState<string | null>(null)
  const [comparing, setComparing] = useState(false)
  const [fly, setFly] = useState<FlyTarget | null>(null)
  const [filter, setFilter] = useState<Filter | null>(null)
  const [filterNonce, setFilterNonce] = useState(0)
  const [scope, setScope] = useState('') // state chosen in the Plan rail (independent of Ask filters)
  const [askOpen, setAskOpen] = useState(false)
  const [alertsOpen, setAlertsOpen] = useState(false)
  const [colorBy, setColorBy] = useState<ColorBy>('overall')
  const [base, setBase] = useState<Basemap>('gray')
  const [showCities, setShowCities] = useState(false)
  const field = useFieldList()

  const detail = useApi(selected ? () => api.station(selected) : null, selected)
  const d = detail.data

  const flyTo = useCallback((lat: number, lon: number, zoom = 11) => setFly({ lat, lon, zoom, nonce: Date.now() }), [])

  const select = useCallback(
    (id: string, at?: { lat: number; lon: number }) => {
      if (comparing && selected && id !== selected) {
        setSecondary(id)
        setComparing(false)
        return
      }
      setSelected(id)
      setSecondary(null)
      setComparing(false)
      if (at) flyTo(at.lat, at.lon)
    },
    [comparing, selected, flyTo],
  )

  const applyFilter = (f: Filter | null) => {
    setFilter(f)
    setFilterNonce((n) => n + 1)
  }

  const onLayer = (id: string, kind?: string) => {
    if (kind === 'color') setColorBy(id as ColorBy)
    else if (kind === 'base') setBase((b) => (b === id ? 'gray' : (id as Basemap)))
    else if (kind === 'overlay' && id === 'cities') setShowCities((v) => !v)
  }
  const isChecked = (id: string, kind?: string) =>
    kind === 'color' ? colorBy === id : kind === 'base' ? base === id : id === 'cities' ? showCities : false

  const list = stations.data ?? []
  const inScope = filter ? list.filter((s) => filter.ids.has(s.id)) : list
  const shownHigh = inScope.filter((s) => s.risk === 'high').length

  return (
    <div className="flex h-full flex-col bg-[var(--color-abyss)]">
      <Header
        view={view}
        onHome={onHome}
        onSwitch={onSwitch}
        askOpen={askOpen}
        setAskOpen={setAskOpen}
        alertsOpen={alertsOpen}
        setAlertsOpen={setAlertsOpen}
        onPick={(h) => {
          setYearIdx(0)
          select(h.id, h)
        }}
      />

      {alertsOpen && (
        <AlertsPanel
          onClose={() => setAlertsOpen(false)}
          onView={(a) => {
            select(a.id, a)
            setAlertsOpen(false)
          }}
        />
      )}

      {view === 'sentinel' ? (
        <SentinelView overview={overview.data} />
      ) : (
        <div className="flex min-h-0 flex-1">
          <LayersPanel isChecked={isChecked} onLayer={onLayer} />

          {/* MAP */}
          <div className="relative min-w-0 flex-1">
            {stations.error || overview.error ? (
              <div className="absolute inset-0 grid place-items-center text-[13px] text-[var(--color-risk-high)]">
                Could not reach the API ({stations.error ?? overview.error}). Start it with: uvicorn src.api.server:app --port 8000
              </div>
            ) : (
              <WaterMap
                stations={list}
                colorBy={colorBy}
                base={base}
                cities={showCities ? cities.data : null}
                selected={selected}
                secondary={secondary}
                filterIds={filter?.ids ?? null}
                filterNonce={filterNonce}
                fly={fly}
                onSelect={(id) => select(id)}
              />
            )}

            <div className="absolute left-4 top-4 z-10">
              <LegendChip note={legendNote(overview.data, year, colorBy)} />
            </div>

            {comparing && (
              <div className="fs-fade absolute left-1/2 top-4 z-10 flex -translate-x-1/2 items-center gap-3 rounded-[4px] border border-[var(--color-accent)]/40 bg-[color-mix(in_oklab,var(--color-panel)_92%,transparent)] px-3 py-2 text-[12px] backdrop-blur">
                <span className="text-[var(--color-accent)]">Compare:</span>
                <span className="text-[var(--color-ink)]">click a second station on the map</span>
                <button onClick={() => setComparing(false)} className="text-[var(--color-mute)] hover:text-[var(--color-ink)]">Cancel</button>
              </div>
            )}

            {filter && !comparing && (
              <div className="fs-fade absolute left-1/2 top-4 z-10 flex max-w-[60%] -translate-x-1/2 items-center gap-3 rounded-[4px] border border-[var(--color-accent)]/40 bg-[color-mix(in_oklab,var(--color-panel)_92%,transparent)] px-3 py-2 text-[12px] backdrop-blur">
                <span className="text-[var(--color-accent)]">Filtered:</span>
                <span className="truncate text-[var(--color-ink)]">{filter.label} ({filter.ids.size})</span>
                <button onClick={() => { applyFilter(null); setScope('') }} className="flex shrink-0 items-center gap-1 text-[var(--color-mute)] hover:text-[var(--color-ink)]">
                  <Icon name="reset" className="size-3.5" /> Reset filter
                </button>
              </div>
            )}

            {askOpen && (
              <AskPanel
                onClose={() => setAskOpen(false)}
                onFilter={(r) => {
                  applyFilter({ label: r.label, ids: new Set(r.ids) })
                  setAskOpen(false)
                }}
              />
            )}

            <TimeSlider years={overview.data?.years ?? []} idx={yearIdx} setIdx={setYearIdx} />

            {/* status bar */}
            <div className="absolute inset-x-0 bottom-0 z-10 flex items-center justify-between border-t border-[var(--color-hair)] bg-[color-mix(in_oklab,var(--color-panel)_92%,transparent)] px-4 py-2 backdrop-blur">
              <span className="text-[12px] text-[var(--color-mute)]">
                Selected area: <span className="text-[var(--color-ink)]">{d ? `${d.name}, ${d.state}` : (filter?.label ?? 'India')}</span> —{' '}
                {d ? (
                  <>
                    Risk: <span style={{ color: RISK[d.risk].hex }} className="font-mono font-semibold">{riskWord(d.risk)}</span>
                  </>
                ) : (
                  <>
                    <span className="tnum">{inScope.length.toLocaleString()}</span> stations · <span style={{ color: RISK.high.hex }} className="font-mono font-semibold">{shownHigh}</span> high
                  </>
                )}
              </span>
              <span className="tnum text-[11px] text-[var(--color-mute-2)]">
                {d ? `${d.lat.toFixed(4)}° N, ${d.lon.toFixed(4)}° E · ${d.scene.date ? `Sentinel-2 ${d.scene.date}` : 'no scene'}` : 'CPCB stations + Sentinel-2'}
              </span>
            </div>
          </div>

          {/* RIGHT RAIL */}
          <aside className="relative z-[600] hidden min-h-0 w-[360px] shrink-0 overflow-y-auto border-l border-[var(--color-hair)] bg-[var(--color-panel)] shadow-[-6px_0_28px_-22px_rgba(30,40,50,0.55)] xl:block">
            {selected && secondary ? (
              <ComparePanel
                a={selected}
                b={secondary}
                onBack={() => setSecondary(null)}
                onFit={(a, b) => flyTo((a.lat + b.lat) / 2, (a.lon + b.lon) / 2, 7)}
              />
            ) : selected ? (
              detail.loading && !d ? (
                <p className="p-4 text-[12px] text-[var(--color-mute)]">Loading station…</p>
              ) : d ? (
                <StationPanel
                  d={d}
                  mode={view}
                  onClose={() => {
                    setSelected(null)
                    setComparing(false)
                  }}
                  inList={field.has(d.id)}
                  onToggleList={() => field.toggle({ id: d.id, name: d.name, state: d.state, risk: d.risk })}
                  comparing={comparing}
                  onCompare={() => setComparing((c) => !c)}
                />
              ) : (
                <p className="p-4 text-[12px] text-[var(--color-risk-high)]">Could not load station.</p>
              )
            ) : view === 'plan' ? (
              <PlanRail
                overview={overview.data}
                onScope={async (state) => {
                  setScope(state)
                  if (!state) return applyFilter(null)
                  const all = await api.stations()
                  applyFilter({ label: state, ids: new Set(all.filter((s) => s.state === state).map((s) => s.id)) })
                }}
                scope={scope}
              />
            ) : (
              <RestoreRail onOpen={(p) => select(p.id, p)} field={field} />
            )}
          </aside>
        </div>
      )}
    </div>
  )
}

/* ─────────────────────────── Header ─────────────────────────── */
function Header({
  view,
  onHome,
  onSwitch,
  askOpen,
  setAskOpen,
  alertsOpen,
  setAlertsOpen,
  onPick,
}: {
  view: View
  onHome: () => void
  onSwitch: (v: View) => void
  askOpen: boolean
  setAskOpen: (v: boolean) => void
  alertsOpen: boolean
  setAlertsOpen: (v: boolean) => void
  onPick: (h: SearchHit) => void
}) {
  const label = view === 'plan' ? 'PLAN & PREVENT' : view === 'restore' ? 'RESTORE & PRIORITIZE' : 'SENTINEL-2 EXPLORER'
  const [q, setQ] = useState('')
  const [hits, setHits] = useState<SearchHit[]>([])
  const [open, setOpen] = useState(false)
  const alerts = useApi(api.alerts, 'alerts')

  useEffect(() => {
    if (q.trim().length < 2) {
      setHits([])
      return
    }
    const t = setTimeout(() => api.search(q).then(setHits, () => setHits([])), 220)
    return () => clearTimeout(t)
  }, [q])

  return (
    <header className="flex items-center gap-4 border-b border-[var(--color-hair)] bg-[var(--color-panel)] px-4 py-2.5">
      <button onClick={onHome} className="flex items-center gap-2.5">
        <span className="grid size-7 place-items-center rounded-[4px] bg-[var(--color-accent)] text-[var(--color-on-accent)]"><Icon name="globe" className="size-4" /></span>
        <span className="font-mono text-[13px] font-semibold tracking-[0.14em] text-[var(--color-ink)]">FLOW STATE</span>
      </button>
      <span className="h-4 w-px bg-[var(--color-hair-2)]" />
      <span className="eyebrow text-[var(--color-accent)]">{label}</span>

      <div className="relative mx-2 max-w-md flex-1">
        <Icon name="search" className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-[var(--color-mute-2)]" />
        <input
          value={q}
          onChange={(e) => {
            setQ(e.target.value)
            setOpen(true)
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          placeholder="Search station, river, lake, city or state"
          className="w-full rounded-[4px] border border-[var(--color-hair)] bg-[var(--color-abyss)] py-1.5 pl-9 pr-3 text-[13px] text-[var(--color-ink)] outline-none placeholder:text-[var(--color-mute-2)] focus:border-[var(--color-accent)]"
        />
        {open && hits.length > 0 && view !== 'sentinel' && (
          <ul className="absolute inset-x-0 top-full z-[900] mt-1 overflow-hidden rounded-[5px] border border-[var(--color-hair)] bg-[var(--color-panel)] shadow-[0_18px_50px_-24px_rgba(30,40,50,0.4)]">
            {hits.map((h) => (
              <li key={h.id}>
                <button
                  onMouseDown={() => {
                    onPick(h)
                    setQ('')
                    setOpen(false)
                  }}
                  className="flex w-full items-center gap-2.5 px-3 py-2 text-left hover:bg-[var(--color-panel-2)]"
                >
                  <span className="size-2 shrink-0 rounded-full" style={{ background: RISK[h.risk].hex }} />
                  <span className="min-w-0 flex-1 truncate text-[13px] text-[var(--color-ink)]">{h.name}</span>
                  <span className="shrink-0 text-[11px] text-[var(--color-mute-2)]">{h.state}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="ml-auto flex items-center gap-1.5">
        {view !== 'sentinel' && (
          <button onClick={() => setAskOpen(!askOpen)} className={`flex items-center gap-1.5 rounded-[4px] border px-2.5 py-1.5 text-[12px] transition-colors ${askOpen ? 'border-[var(--color-accent)] text-[var(--color-accent)]' : 'border-[var(--color-hair)] text-[var(--color-mute)] hover:text-[var(--color-ink)]'}`}>
            <Icon name="sparkles" className="size-4" /> Ask Flow State
          </button>
        )}
        <button onClick={() => setAlertsOpen(!alertsOpen)} className="relative grid size-8 place-items-center rounded-[4px] border border-[var(--color-hair)] text-[var(--color-mute)] hover:text-[var(--color-ink)]">
          <Icon name="bell" className="size-4" />
          {!!alerts.data?.length && <span className="absolute right-1 top-1 size-1.5 rounded-full bg-[var(--color-risk-high)]" />}
        </button>
        <div className="ml-1 flex rounded-[4px] border border-[var(--color-hair)] p-0.5">
          {(['plan', 'restore'] as Mode[]).map((m) => (
            <button key={m} onClick={() => onSwitch(m)} className={`rounded-[3px] px-2 py-1 font-mono text-[11px] font-semibold tracking-wide transition-colors ${view === m ? 'bg-[var(--color-accent)] text-[var(--color-on-accent)]' : 'text-[var(--color-mute)] hover:text-[var(--color-ink)]'}`}>
              {m === 'plan' ? 'PLAN' : 'RESTORE'}
            </button>
          ))}
        </div>
      </div>
    </header>
  )
}

/* ─────────────────────────── Layers panel ─────────────────────────── */
function LayersPanel({ isChecked, onLayer }: { isChecked: (id: string, kind?: string) => boolean; onLayer: (id: string, kind?: string) => void }) {
  const [open, setOpen] = useState(true)
  if (!open)
    return (
      <button onClick={() => setOpen(true)} className="flex w-11 shrink-0 flex-col items-center gap-2 border-r border-[var(--color-hair)] bg-[var(--color-panel)] pt-3 text-[var(--color-mute)] hover:text-[var(--color-ink)]">
        <Icon name="layers" className="size-4" />
      </button>
    )
  return (
    <aside className="relative z-[600] flex min-h-0 w-[280px] shrink-0 flex-col border-r border-[var(--color-hair)] bg-[var(--color-panel)] shadow-[6px_0_28px_-22px_rgba(30,40,50,0.55)]">
      <div className="flex items-center justify-between px-4 py-3">
        <span className="eyebrow flex items-center gap-2 text-[var(--color-ink)]"><Icon name="layers" className="size-4 text-[var(--color-accent)]" /> Layers</span>
        <button onClick={() => setOpen(false)} className="text-[var(--color-mute-2)] hover:text-[var(--color-ink)]"><Icon name="chevron" className="size-4 rotate-90" /></button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-4">
        {LAYER_GROUPS.map((g) => (
          <div key={g.group} className="mb-3">
            <p className="eyebrow px-2 py-1.5 text-[var(--color-mute-2)]">{g.group}</p>
            {g.layers.map((l) => {
              const on = isChecked(l.id, l.kind)
              return (
                <label key={l.id} className={`group relative flex items-center gap-2.5 rounded-[4px] px-2 py-1.5 ${l.soon ? 'cursor-not-allowed opacity-50' : 'cursor-pointer hover:bg-[var(--color-panel-2)]'}`} title={l.hint}>
                  <input type={l.kind === 'color' ? 'radio' : 'checkbox'} name={l.kind === 'color' ? 'colorBy' : undefined} disabled={l.soon} checked={on} onChange={() => onLayer(l.id, l.kind)} className="peer sr-only" />
                  <span className={`grid size-4 place-items-center border border-[var(--color-hair-2)] peer-checked:border-[var(--color-accent)] peer-checked:bg-[var(--color-accent)] ${l.kind === 'color' ? 'rounded-full' : 'rounded-[3px]'}`}>
                    {on && <Icon name="chevron" className="size-3 text-[var(--color-on-accent)]" />}
                  </span>
                  <span className="size-2.5 shrink-0 rounded-[2px]" style={{ background: l.swatch }} />
                  <span className="flex-1 text-[12.5px] text-[var(--color-ink)]">{l.label}</span>
                  {l.soon ? (
                    <span className="eyebrow text-[9px] text-[var(--color-mute-2)]">soon</span>
                  ) : (
                    <span className="text-[10px] text-[var(--color-mute-2)] opacity-0 transition-opacity group-hover:opacity-100">{l.hint}</span>
                  )}
                </label>
              )
            })}
          </div>
        ))}
      </div>
    </aside>
  )
}

/* ─────────────────────────── Station detail panel ─────────────────────────── */
function StationPanel({
  d,
  mode,
  onClose,
  inList,
  onToggleList,
  comparing,
  onCompare,
}: {
  d: StationDetail
  mode: View
  onClose: () => void
  inList: boolean
  onToggleList: () => void
  comparing: boolean
  onCompare: () => void
}) {
  const title = d.risk === 'high' ? 'WATER QUALITY ALERT' : d.risk === 'mod' ? 'ELEVATED SCREENING RISK' : 'STATION DETAIL'
  const guidance = d.risk === 'high' ? 'Investigate' : d.risk === 'mod' ? 'Monitor' : 'Routine'
  return (
    <div className="fs-slide-in flex flex-col">
      <div className="flex items-start justify-between border-b border-[var(--color-hair)] px-4 py-3">
        <div className="min-w-0">
          <p className="font-mono text-[13px] font-semibold tracking-wide" style={{ color: RISK[d.risk].hex }}>{title}</p>
          <p className="mt-1 truncate text-[13px] text-[var(--color-ink)]" title={d.name}>{d.name}</p>
          <p className="mt-0.5 text-[12px] text-[var(--color-mute)]">{d.position}</p>
        </div>
        <button onClick={onClose} className="shrink-0 text-[var(--color-mute-2)] hover:text-[var(--color-ink)]"><Icon name="close" className="size-4" /></button>
      </div>

      <div className="space-y-4 p-4">
        <div className="flex flex-wrap items-center gap-2">
          <RiskBadge risk={d.risk} />
          <GuidancePill><Icon name="target" className="size-3" /> {guidance}</GuidancePill>
          <span className="tnum text-[11px] text-[var(--color-mute-2)]">
            {d.basis === 'screening' ? `model P = ${Math.round((d.prob ?? 0) * 100)}%` : 'measured WQI basis'}
          </span>
        </div>

        {/* readings table */}
        <div className="rounded-[5px] border border-[var(--color-hair)]">
          <div className="grid grid-cols-[1fr_auto_auto] gap-2 border-b border-[var(--color-hair)] px-3 py-2">
            {['Indicator', 'Latest measured', 'Trend'].map((c) => <span key={c} className="eyebrow text-[var(--color-mute-2)]">{c}</span>)}
          </div>
          {d.readings.map((r) => (
            <div key={r.name} className="grid grid-cols-[1fr_auto_auto] items-center gap-2 px-3 py-2 text-[13px] [&:not(:last-child)]:border-b [&:not(:last-child)]:border-[var(--color-hair)]">
              <span className="flex items-center gap-2 text-[var(--color-ink)]">
                {r.band && <span className="size-1.5 rounded-full" style={{ background: RISK[r.band].hex }} />}
                {r.name}
              </span>
              <span className="tnum text-[var(--color-mute)]">{fmtReading(r.value, r.unit)}</span>
              <TrendCell dir={r.trend} badWhen={r.name === 'Dissolved oxygen' ? 'down' : 'up'} />
            </div>
          ))}
        </div>
        <p className="-mt-2 text-[10.5px] leading-snug text-[var(--color-mute-2)]">
          {d.visits} visit{d.visits === 1 ? '' : 's'} · {d.first_date}{d.visits > 1 ? ` → ${d.last_date}` : ''}. Trend compares the latest value with this station's earlier visits.
          {d.tier && ` WQI ${d.wqi?.toFixed(0)} (${d.tier}${d.cpcb_class ? `, CPCB class ${d.cpcb_class}` : ''}).`}
        </p>

        {/* what's around it */}
        {d.around.length > 0 && (
          <div>
            <p className="eyebrow mb-1 text-[var(--color-mute-2)]">What's around it?</p>
            <ul className="divide-y divide-[var(--color-hair)]">
              {d.around.map((a) => <DistanceRow key={a.kind} {...a} />)}
            </ul>
          </div>
        )}

        {d.yearly.length > 0 && (
          <div>
            <p className="eyebrow mb-2 text-[var(--color-mute-2)]">Measured trend — by year</p>
            <div className="grid grid-cols-3 gap-2">
              {d.yearly.map((t) => (
                <div key={t.year} className="overflow-hidden rounded-[4px] border border-[var(--color-hair)]">
                  <div className="h-12" style={{ background: t.risk ? `radial-gradient(circle at 50% 60%, color-mix(in oklab, ${RISK[t.risk].hex} 70%, transparent), transparent 70%), var(--color-water)` : 'var(--color-water)' }} />
                  <p className="tnum bg-[var(--color-panel-2)] px-2 py-1 text-center text-[11px] text-[var(--color-mute)]">{t.year} · {t.visits}×</p>
                </div>
              ))}
            </div>
          </div>
        )}

        <p className="rounded-[5px] border border-[var(--color-hair)] bg-[var(--color-panel-2)] p-3 text-[13px] leading-relaxed text-[var(--color-mute)]">{d.rationale}</p>

        <WhyDisclosure signals={d.signals} />

        {mode === 'plan' ? (
          <>
            <SiteCard d={d} />
            <button onClick={onCompare} className={`w-full rounded-[4px] border py-2 font-mono text-[11px] font-semibold tracking-wider transition-colors ${comparing ? 'border-[var(--color-accent)] text-[var(--color-accent)]' : 'border-[var(--color-hair-2)] text-[var(--color-ink)] hover:border-[var(--color-accent)]'}`}>
              {comparing ? 'PICK A SECOND STATION ON THE MAP…' : 'COMPARE WITH ANOTHER SITE'}
            </button>
          </>
        ) : (
          <div className="space-y-2">
            {d.recommendations.map((r) => <RecommendationCard key={r.n} {...r} />)}
          </div>
        )}

        <button onClick={onToggleList} className={`w-full rounded-[4px] py-2.5 font-mono text-[12px] font-semibold tracking-wider transition-opacity hover:opacity-90 ${inList ? 'border border-[var(--color-accent)] text-[var(--color-accent)]' : 'bg-[var(--color-accent)] text-[var(--color-on-accent)]'}`}>
          {inList ? 'ON FIELD LIST — CLICK TO REMOVE' : 'PRIORITIZE FOR FIELD INVESTIGATION'}
        </button>
      </div>
    </div>
  )
}

/* ─────────────────────────── Planning site card ─────────────────────────── */
function SiteCard({ d, name, compact }: { d: StationDetail; name?: string; compact?: boolean }) {
  return (
    <div className="rounded-[5px] border border-[var(--color-hair)] bg-[var(--color-panel)] p-3">
      <div className={compact ? 'space-y-1.5' : 'flex items-center justify-between gap-2'}>
        <p className={`font-mono text-[12px] font-semibold tracking-wide text-[var(--color-ink)] ${compact ? 'break-words' : 'truncate'}`}>{name ?? 'PLANNING RISK'}</p>
        <RiskBadge risk={d.risk} label={compact ? `RISK: ${riskWord(d.risk)}` : `WATER-QUALITY RISK: ${riskWord(d.risk)}`} size="sm" />
      </div>
      <ul className="mt-3 space-y-1.5">
        {d.factors.map((f) => (
          <li key={f.text} className="flex items-start gap-2 text-[12px] text-[var(--color-mute)]">
            <span className="mt-1.5 size-1.5 shrink-0 rounded-full" style={{ background: RISK[f.hard ? 'high' : 'mod'].hex }} />
            {f.text}
          </li>
        ))}
      </ul>
      {!compact && (
        <p className="mt-3 border-t border-[var(--color-hair)] pt-2 text-[11.5px] italic leading-snug text-[var(--color-mute-2)]">
          Planning consideration: further environmental assessment and infrastructure capacity review recommended before development.
        </p>
      )}
    </div>
  )
}

/* ─────────────────────────── Compare two stations ─────────────────────────── */
function ComparePanel({ a, b, onBack, onFit }: { a: string; b: string; onBack: () => void; onFit: (a: StationDetail, b: StationDetail) => void }) {
  const da = useApi(() => api.station(a), a)
  const db = useApi(() => api.station(b), b)
  const cmp = useApi(() => api.compare(a, b), a + b)
  const fitted = useRef(false)
  useEffect(() => {
    if (da.data && db.data && !fitted.current) {
      fitted.current = true
      onFit(da.data, db.data)
    }
  }, [da.data, db.data, onFit])

  const rows: { label: string; key: string; unit?: string; nd?: number }[] = [
    { label: 'Model breach probability', key: 'prob', unit: '', nd: 2 },
    { label: 'BOD (latest)', key: 'bod', unit: ' mg/L', nd: 1 },
    { label: 'DO (latest)', key: 'do', unit: ' mg/L', nd: 1 },
    { label: 'Turbidity (latest)', key: 'turbidity', unit: ' NTU', nd: 1 },
    { label: 'WQI (higher = worse)', key: 'wqi', nd: 0 },
    { label: 'Distance to nearest city', key: 'dist_city_km', unit: ' km', nd: 0 },
    { label: 'Visits on record', key: 'visits', nd: 0 },
  ]
  const cell = (v: unknown, r: (typeof rows)[number]) => (typeof v === 'number' ? `${v.toFixed(r.nd ?? 1)}${r.unit ?? ''}` : '—')

  return (
    <div className="fs-slide-in space-y-3 p-4">
      <button onClick={onBack} className="flex items-center gap-1.5 text-[12px] text-[var(--color-mute)] hover:text-[var(--color-ink)]">
        <Icon name="chevron" className="size-4 rotate-90" /> Back to site
      </button>
      <p className="font-mono text-[13px] font-semibold tracking-wide text-[var(--color-ink)]">SPATIAL COMPARISON</p>
      {cmp.data && <p className="tnum text-[11px] text-[var(--color-mute-2)]">{cmp.data.distance_km} km apart</p>}
      <div className="grid grid-cols-2 gap-2">
        {da.data ? <SiteCard compact d={da.data} name={`A · ${da.data.name}`} /> : <p className="text-[12px] text-[var(--color-mute)]">Loading…</p>}
        {db.data ? <SiteCard compact d={db.data} name={`B · ${db.data.name}`} /> : <p className="text-[12px] text-[var(--color-mute)]">Loading…</p>}
      </div>
      {cmp.data && (
        <div className="rounded-[5px] border border-[var(--color-hair)]">
          <div className="grid grid-cols-[1.4fr_1fr_1fr] gap-2 border-b border-[var(--color-hair)] px-3 py-2">
            {['', 'Site A', 'Site B'].map((c) => <span key={c} className="eyebrow text-[var(--color-mute-2)]">{c}</span>)}
          </div>
          {rows.map((r) => (
            <div key={r.key} className="grid grid-cols-[1.4fr_1fr_1fr] items-center gap-2 px-3 py-1.5 text-[12px] [&:not(:last-child)]:border-b [&:not(:last-child)]:border-[var(--color-hair)]">
              <span className="text-[var(--color-mute)]">{r.label}</span>
              <span className="tnum text-[var(--color-ink)]">{cell(cmp.data!.a[r.key], r)}</span>
              <span className="tnum text-[var(--color-ink)]">{cell(cmp.data!.b[r.key], r)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ─────────────────────────── Plan & Prevent rail ─────────────────────────── */
function PlanRail({ overview, scope, onScope }: { overview: Overview | null; scope: string; onScope: (state: string) => void }) {
  const [o, setO] = useState<Overview | null>(overview)
  useEffect(() => {
    api.overview(scope || undefined).then(setO, () => setO(overview))
  }, [scope, overview])
  const stats = o
    ? [
        { n: o.stations.toLocaleString(), label: 'Stations monitored' },
        { n: o.high.toLocaleString(), label: 'High screening risk' },
        { n: o.confirmed_breaches.toLocaleString(), label: 'Latest BOD over the 3 mg/L limit' },
        { n: o.deteriorating.toLocaleString(), label: 'WQI worse than earlier visits' },
      ]
    : []
  const chain = [{ icon: 'built', label: 'Major city', sub: 'urban discharge source (proxy)' }, { icon: 'drain', label: 'Discharge into river / lake', sub: 'not mapped: no drain or outlet data' }, { icon: 'target', label: 'Flagged station', sub: 'select one on the map' }]
  return (
    <div className="fs-fade space-y-5 p-4">
      <section>
        <div className="mb-2 flex items-center justify-between gap-2">
          <p className="eyebrow text-[var(--color-mute-2)]">{scope || 'India'} Water Intelligence</p>
          <select
            value={scope}
            onChange={(e) => onScope(e.target.value)}
            className="max-w-[150px] rounded-[4px] border border-[var(--color-hair)] bg-[var(--color-panel)] px-1.5 py-1 text-[11px] text-[var(--color-ink)]"
          >
            <option value="">All states</option>
            {overview?.states.map((s) => <option key={s}>{s}</option>)}
          </select>
        </div>
        <div className="grid grid-cols-2 gap-2">
          {stats.map((s) => (
            <div key={s.label} className="rounded-[5px] border border-[var(--color-hair)] bg-[var(--color-panel-2)] p-3">
              <p className="tnum text-2xl font-semibold text-[var(--color-ink)]">{s.n}</p>
              <p className="mt-0.5 text-[11px] leading-tight text-[var(--color-mute)]">{s.label}</p>
            </div>
          ))}
        </div>
      </section>

      <section>
        <p className="eyebrow mb-2 text-[var(--color-mute-2)]">Planning Risk</p>
        <p className="rounded-[5px] border border-dashed border-[var(--color-hair-2)] p-3 text-[12.5px] leading-relaxed text-[var(--color-mute)]">
          Select a station on the map to see its planning risk factors, or compare two sites side by side.
        </p>
      </section>

      <InfraChain steps={chain} />
    </div>
  )
}

/* ─────────────────────────── Restore & Prioritize rail ─────────────────────────── */
function RestoreRail({ onOpen, field }: { onOpen: (p: Priority) => void; field: ReturnType<typeof useFieldList> }) {
  const priorities = useApi(api.priorities, 'p')
  const recs = useApi(api.recommendations, 'r')
  const exportCsv = () => {
    const csv = ['station_id,name,state,screening_risk', ...field.items.map((i) => `${i.id},"${i.name.replace(/"/g, '""')}","${i.state}",${i.risk}`)].join('\n')
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
    a.download = 'field_investigation_list.csv'
    a.click()
  }
  return (
    <div className="fs-fade space-y-5 p-4">
      <section>
        <p className="eyebrow mb-2 text-[var(--color-mute-2)]">Top priority stations</p>
        <div className="space-y-1.5">
          {priorities.loading && <p className="text-[12px] text-[var(--color-mute)]">Loading…</p>}
          {priorities.data?.map((p) => (
            <button key={p.id} onClick={() => onOpen(p)} className="flex w-full items-center gap-3 rounded-[5px] border border-[var(--color-hair)] bg-[var(--color-panel-2)] p-2.5 text-left transition-colors hover:border-[var(--color-hair-2)]">
              <span className="size-2.5 shrink-0 rounded-full" style={{ background: RISK[p.risk].hex }} />
              <div className="min-w-0 flex-1">
                <p className="truncate text-[13px] text-[var(--color-ink)]">{p.name}</p>
                <p className="truncate text-[11px] text-[var(--color-mute)]">{p.reason}</p>
              </div>
              <Icon name="chevron" className="size-4 shrink-0 -rotate-90 text-[var(--color-mute-2)]" />
            </button>
          ))}
        </div>
        <p className="mt-2 text-[10px] leading-snug text-[var(--color-mute-2)]">Ranked by screening-model breach probability (out-of-fold). Priority categories, not a precise national ranking.</p>
      </section>

      <section>
        <p className="eyebrow mb-2 text-[var(--color-mute-2)]">Recommendations</p>
        <div className="space-y-2">
          {recs.data?.map((r: Rec) => <RecommendationCard key={r.n} {...r} />)}
        </div>
      </section>

      {field.items.length > 0 && (
        <section>
          <div className="mb-2 flex items-center justify-between">
            <p className="eyebrow text-[var(--color-mute-2)]">Field investigation list ({field.items.length})</p>
            <span className="flex gap-3 text-[11px]">
              <button onClick={exportCsv} className="text-[var(--color-accent)] hover:underline">Export CSV</button>
              <button onClick={field.clear} className="text-[var(--color-mute)] hover:underline">Clear</button>
            </span>
          </div>
          <ul className="divide-y divide-[var(--color-hair)] rounded-[5px] border border-[var(--color-hair)]">
            {field.items.map((i) => (
              <li key={i.id} className="flex items-center gap-2 px-3 py-2 text-[12px]">
                <span className="size-2 shrink-0 rounded-full" style={{ background: RISK[i.risk].hex }} />
                <button onClick={() => onOpen({ id: i.id } as Priority)} className="min-w-0 flex-1 truncate text-left text-[var(--color-ink)] hover:underline">{i.name}</button>
                <span className="shrink-0 text-[var(--color-mute-2)]">{i.state}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}

/* ─────────────────────────── Time slider ─────────────────────────── */
function TimeSlider({ years, idx, setIdx }: { years: number[]; idx: number; setIdx: (i: number) => void }) {
  if (years.length === 0) return null
  const labels = ['All', ...years.map(String)]
  return (
    <div className="absolute bottom-11 left-1/2 z-10 w-[min(520px,80%)] -translate-x-1/2 rounded-[6px] border border-[var(--color-hair)] bg-[color-mix(in_oklab,var(--color-panel)_92%,transparent)] px-4 py-3 backdrop-blur">
      <div className="flex items-center justify-between">
        <span className="eyebrow text-[var(--color-mute-2)]">Time</span>
        <span className="tnum text-[12px] text-[var(--color-accent)]">{idx === 0 ? 'All years · screening model' : `${labels[idx]} · measured`}</span>
      </div>
      <input type="range" min={0} max={years.length} step={1} value={idx} onChange={(e) => setIdx(Number(e.target.value))} className="mt-2 w-full accent-[var(--color-accent)]" />
      <div className="tnum mt-1 flex justify-between text-[10px] text-[var(--color-mute-2)]">
        {labels.map((l) => <span key={l}>{l}</span>)}
      </div>
      {idx > 0 && <p className="mt-1.5 text-[11px] text-[var(--color-mute)]">Stations visited in {labels[idx]}, banded by that year's measured water-quality index.</p>}
    </div>
  )
}

/* ─────────────────────────── Ask Flow State ─────────────────────────── */
function AskPanel({ onClose, onFilter }: { onClose: () => void; onFilter: (r: AskResult) => void }) {
  const [state, setState] = useState<'empty' | 'thinking' | 'result' | 'error'>('empty')
  const [q, setQ] = useState('')
  const [res, setRes] = useState<AskResult | null>(null)

  const run = (query: string) => {
    setQ(query)
    setState('thinking')
    api.ask(query).then(
      (r) => {
        setRes(r)
        setState('result')
      },
      () => setState('error'),
    )
  }

  return (
    <div className="fs-fade absolute inset-x-4 top-4 z-20 mx-auto max-w-lg rounded-[6px] border border-[var(--color-accent)]/40 bg-[color-mix(in_oklab,var(--color-panel)_96%,transparent)] p-4 shadow-[0_18px_50px_-24px_rgba(30,40,50,0.4)] backdrop-blur-md">
      <div className="flex items-center gap-2">
        <Icon name="sparkles" className="size-4 text-[var(--color-accent)]" />
        <input
          autoFocus
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && q && run(q)}
          placeholder="Ask Flow State anything about these stations..."
          className="flex-1 bg-transparent text-[13px] text-[var(--color-ink)] outline-none placeholder:text-[var(--color-mute-2)]"
        />
        <button onClick={onClose} className="text-[var(--color-mute-2)] hover:text-[var(--color-ink)]"><Icon name="close" className="size-4" /></button>
      </div>

      {state === 'empty' && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {ASK_CHIPS.map((c) => (
            <button key={c} onClick={() => run(c)} className="rounded-full border border-[var(--color-hair)] px-2.5 py-1 text-[11.5px] text-[var(--color-mute)] transition-colors hover:border-[var(--color-accent)] hover:text-[var(--color-ink)]">
              {c}
            </button>
          ))}
        </div>
      )}

      {state === 'thinking' && (
        <div className="mt-3 flex items-center gap-2 text-[12px] text-[var(--color-mute)]">
          <span className="fs-pulse size-2 rounded-full bg-[var(--color-accent)]" /> Reading station records…
        </div>
      )}

      {state === 'error' && <p className="mt-3 text-[12px] text-[var(--color-risk-high)]">The API did not respond. Is the backend running?</p>}

      {state === 'result' && res && (
        <div className="mt-3 space-y-3">
          <p className="text-[12.5px] leading-relaxed text-[var(--color-mute)]">{res.summary}</p>
          {res.notes.map((n) => (
            <p key={n} className="rounded-[4px] border border-[var(--color-risk-mod)]/40 bg-[color-mix(in_oklab,var(--color-risk-mod)_8%,transparent)] px-2.5 py-1.5 text-[11.5px] leading-snug text-[var(--color-mute)]">{n}</p>
          ))}
          {res.understood && res.count > 0 && (
            <button onClick={() => onFilter(res)} className="w-full rounded-[4px] bg-[var(--color-accent)] py-2 font-mono text-[11px] font-semibold tracking-wider text-[var(--color-on-accent)]">
              APPLY FILTER TO MAP ({res.count})
            </button>
          )}
          <button onClick={() => setState('empty')} className="text-[11px] text-[var(--color-mute)] hover:text-[var(--color-ink)]">Ask something else</button>
        </div>
      )}
    </div>
  )
}

/* ─────────────────────────── Alerts panel ─────────────────────────── */
function AlertsPanel({ onClose, onView }: { onClose: () => void; onView: (a: Alert) => void }) {
  const alerts = useApi(api.alerts, 'alerts')
  return (
    <div className="fs-slide-in absolute right-4 top-14 z-[900] max-h-[80vh] w-[340px] overflow-y-auto rounded-[6px] border border-[var(--color-hair)] bg-[var(--color-panel)] p-2 shadow-[0_18px_50px_-24px_rgba(30,40,50,0.4)]">
      <div className="flex items-center justify-between px-2 py-2">
        <span className="eyebrow text-[var(--color-ink)]">Alerts — most recent visits</span>
        <button onClick={onClose} className="text-[var(--color-mute-2)] hover:text-[var(--color-ink)]"><Icon name="close" className="size-4" /></button>
      </div>
      <div className="space-y-1.5">
        {alerts.loading && <p className="px-2 pb-2 text-[12px] text-[var(--color-mute)]">Loading…</p>}
        {alerts.data?.length === 0 && <p className="px-2 pb-2 text-[12px] text-[var(--color-mute)]">No alerts.</p>}
        {alerts.data?.map((a) => (
          <div key={a.id} className="rounded-[5px] border border-[var(--color-hair)] bg-[var(--color-panel-2)] p-3">
            <div className="flex items-center gap-2">
              <span className="size-2 rounded-full" style={{ background: RISK[a.sev].hex }} />
              <span className="font-mono text-[12px] font-semibold tracking-wide text-[var(--color-ink)]">{a.title}</span>
            </div>
            <p className="mt-1 text-[11px] text-[var(--color-mute-2)]">{a.loc} · {a.date}</p>
            <p className="mt-1 text-[12px] leading-snug text-[var(--color-mute)]">{a.evidence}</p>
            <button onClick={() => onView(a)} className="mt-2 flex items-center gap-1 text-[11px] text-[var(--color-accent)] hover:underline">
              VIEW STATION <Icon name="arrow" className="size-3.5" />
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ─────────────────────────── Sentinel-2 view ─────────────────────────── */
type SBand = 'TRUE COLOUR' | 'HIGH-RES' | 'MODEL RISK'

function SentinelView({ overview }: { overview: Overview | null }) {
  const [band, setBand] = useState<SBand>('TRUE COLOUR')
  const [center, setCenter] = useState<[number, number]>([22.5, 79])
  const [picked, setPicked] = useState<string | null>(null)
  const pipeline = ['Real world', 'Sentinel-2', 'AI model', 'BOD breach probability', 'Screening shortlist']
  const stations = useApi(band === 'MODEL RISK' ? () => api.stations() : null, band === 'MODEL RISK')
  const scene = useApi(() => api.scene(center[0], center[1]), center.join(','))
  const pick = useApi(picked ? () => api.station(picked) : null, picked)
  const m = overview?.model

  // debounce map-move -> scene lookup
  const t = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const onMove = useMemo(() => (lat: number, lon: number) => {
    clearTimeout(t.current)
    t.current = setTimeout(() => setCenter([lat, lon]), 400)
  }, [])

  const basemap: Basemap = band === 'TRUE COLOUR' ? 's2t' : 'esri'
  const sc = scene.data
  return (
    <div className="flex min-h-0 flex-1">
      <div className="relative min-w-0 flex-1">
        <SatelliteMap base={basemap} stations={band === 'MODEL RISK' ? (stations.data as Station[] | null) : null} onMove={onMove} onSelect={setPicked} />
        <div className="absolute left-14 top-4 z-[500] flex rounded-[5px] border border-[var(--color-hair)] bg-[color-mix(in_oklab,var(--color-panel)_88%,transparent)] p-0.5 backdrop-blur">
          {(['TRUE COLOUR', 'HIGH-RES', 'MODEL RISK'] as SBand[]).map((b) => (
            <button key={b} onClick={() => setBand(b)} className={`rounded-[3px] px-3 py-1.5 font-mono text-[11px] font-semibold tracking-wide transition-colors ${band === b ? 'bg-[var(--color-accent)] text-[var(--color-on-accent)]' : 'text-[var(--color-mute)] hover:text-[var(--color-ink)]'}`}>{b}</button>
          ))}
        </div>
        {pick.data && (
          <div className="fs-fade absolute right-4 top-4 z-[500] w-[260px] rounded-[5px] border border-[var(--color-hair)] bg-[color-mix(in_oklab,var(--color-panel)_94%,transparent)] p-3 backdrop-blur">
            <div className="flex items-start justify-between gap-2">
              <p className="text-[13px] text-[var(--color-ink)]">{pick.data.name}</p>
              <button onClick={() => setPicked(null)} className="text-[var(--color-mute-2)] hover:text-[var(--color-ink)]"><Icon name="close" className="size-3.5" /></button>
            </div>
            <p className="mt-0.5 text-[11px] text-[var(--color-mute)]">{pick.data.state}</p>
            <div className="mt-2"><RiskBadge risk={pick.data.risk} size="sm" /></div>
            {pick.data.scene.date && <p className="tnum mt-2 text-[10.5px] text-[var(--color-mute-2)]">Scene {pick.data.scene.date} · cloud {pick.data.scene.cloud?.toFixed(0) ?? '—'}%</p>}
          </div>
        )}
        <div className="tnum absolute bottom-4 left-4 z-[500] max-w-[70%] rounded-[4px] border border-[var(--color-hair)] bg-[color-mix(in_oklab,var(--color-panel)_88%,transparent)] px-3 py-2 text-[11px] text-[var(--color-mute)] backdrop-blur">
          {band === 'TRUE COLOUR' ? 'Sentinel-2 cloudless mosaic (2020), 10 m' : 'Esri World Imagery (basemap, not Sentinel-2)'}
          {sc ? ` · Nearest matched scene: ${sc.date ?? 'n/a'} (${sc.station}, ${sc.distance_km} km) · cloud ${sc.cloud?.toFixed(0) ?? '—'}%` : ' · No matched station scene in view'}
        </div>
      </div>
      <aside className="w-[320px] shrink-0 overflow-y-auto border-l border-[var(--color-hair)] bg-[var(--color-panel)] p-4">
        <p className="eyebrow mb-3 text-[var(--color-mute-2)]">Processing pipeline</p>
        <div className="space-y-1">
          {pipeline.map((s, i) => (
            <div key={s}>
              <div className="flex items-center gap-3 rounded-[4px] border border-[var(--color-hair)] bg-[var(--color-panel-2)] px-3 py-2.5">
                <span className="tnum text-[11px] text-[var(--color-mute-2)]">{String(i + 1).padStart(2, '0')}</span>
                <span className="text-[13px] text-[var(--color-ink)]">{s}</span>
              </div>
              {i < pipeline.length - 1 && <span className="ml-6 block h-3 w-px bg-[var(--color-hair-2)]" />}
            </div>
          ))}
        </div>
        <p className="mt-4 text-[11px] leading-snug text-[var(--color-mute-2)]">
          A triage screening flow{m?.roc_auc != null ? ` — AUC ${m.roc_auc.toFixed(3)} for ${m.target}, ${m.validation}` : ''}. Concentration regression is near zero out-of-fold, so it is not used. Not a lab replacement.
        </p>
      </aside>
    </div>
  )
}
