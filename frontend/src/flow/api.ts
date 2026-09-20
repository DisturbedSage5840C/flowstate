import { useEffect, useState } from 'react'
import type { Risk } from './data'

/* Types mirror src/api/service.py. `risk` values from the API are 'low' | 'mod' | 'high'. */
export type Band = Risk | null
export type Trend = 'up' | 'down' | 'flat'

export type Station = {
  id: string
  name: string
  state: string
  lat: number
  lon: number
  type: string
  risk: Risk
  p: number | null
  rb: Band
  rd: Band
  rt: Band
  ch: 'worse' | 'better' | 'stable'
}

export type Reading = { name: string; unit: string; value: number | null; trend: Trend; band: Band; n: number }
export type Rec = { n: string; title: string; evidence: string; action: string }
export type Estimate = {
  name: string
  unit: string
  date: string
  estimate: number
  measured: number | null
  nearest_km: number | null
  skill: number | null
  skill_metric: 'R2' | 'R2_log'
  spearman: number | null
}
export type StationDetail = {
  id: string
  name: string
  state: string
  lat: number
  lon: number
  type: string
  risk: Risk
  basis: 'screening' | 'measured'
  prob: number | null
  position: string
  visits: number
  first_date: string
  last_date: string
  readings: Reading[]
  around: { dist: string; kind: string; icon: string }[]
  rationale: string
  signals: string[]
  factors: { hard: boolean; text: string }[]
  wqi: number | null
  tier: string | null
  cpcb_class: string | null
  estimates: { items: Estimate[] } | null
  yearly: { year: number; visits: number; risk: Band; wqi: number | null; bod: number | null; do: number | null; turbidity: number | null }[]
  recommendations: Rec[]
  scene: { id: string | null; date: string | null; cloud: number | null }
}

export type Overview = {
  scope: string
  stations: number
  visits: number
  high: number
  moderate: number
  low: number
  confirmed_breaches: number
  deteriorating: number
  states: string[]
  years: number[]
  date_range: [string, string]
  model: { target: string; roc_auc: number | null; base_rate: number | null; precision_top_10pct: number | null; lift_top_10pct: number | null; validation: string; n: number }
  thresholds: { high: number; mod: number }
}
export type City = { name: string; lat: number; lon: number; stations: number; high_share: number; risk: Risk }
export type Priority = { id: string; name: string; risk: Risk; lat: number; lon: number; reason: string }
export type Alert = { id: string; sev: Risk; title: string; loc: string; date: string; evidence: string; lat: number; lon: number }
export type SearchHit = { id: string; name: string; state: string; lat: number; lon: number; risk: Risk }
export type AskResult = { understood: boolean; count: number; ids: string[]; label: string; summary: string; notes: string[] }
export type Scene = { station: string; distance_km: number; scene_id: string; date: string | null; cloud: number | null } | null
export type Compare = {
  a: Record<string, number | string | null>
  b: Record<string, number | string | null>
  distance_km: number
}

/* In dev, Vite proxies /api (see vite.config.ts) so this stays empty. In production the
   frontend (Vercel) and API (Render) are different origins, so the build needs VITE_API_BASE. */
const API_BASE = import.meta.env.VITE_API_BASE ?? ''

async function get<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const qs = params
    ? '?' + new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined).map(([k, v]) => [k, String(v)])).toString()
    : ''
  const r = await fetch(`${API_BASE}/api${path}${qs}`)
  if (!r.ok) throw new Error(`${path}: ${r.status}`)
  return r.json()
}

export const api = {
  overview: (state?: string) => get<Overview>('/overview', { state }),
  stations: (year?: number) => get<Station[]>('/stations', { year }),
  station: (id: string) => get<StationDetail>(`/stations/${id}`),
  compare: (a: string, b: string) => get<Compare>('/compare', { a, b }),
  search: (q: string) => get<SearchHit[]>('/search', { q }),
  cities: () => get<City[]>('/cities'),
  priorities: () => get<Priority[]>('/priorities', { n: 8 }),
  recommendations: () => get<Rec[]>('/recommendations'),
  alerts: () => get<Alert[]>('/alerts'),
  scene: (lat: number, lon: number) => get<Scene>('/scene', { lat, lon }),
  ask: async (query: string): Promise<AskResult> => {
    const r = await fetch(`${API_BASE}/api/ask`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query }) })
    if (!r.ok) throw new Error(`ask: ${r.status}`)
    return r.json()
  },
}

/* Minimal fetch hook: re-runs when `key` changes; ignores stale responses. Pass fn=null to skip. */
export function useApi<T>(fn: (() => Promise<T>) | null, key: unknown): { data: T | null; error: string | null; loading: boolean } {
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>({ data: null, error: null, loading: !!fn })
  useEffect(() => {
    if (!fn) {
      setState({ data: null, error: null, loading: false })
      return
    }
    let live = true
    setState((s) => ({ ...s, loading: true, error: null }))
    fn().then(
      (data) => live && setState({ data, error: null, loading: false }),
      (e) => live && setState({ data: null, error: String(e.message ?? e), loading: false }),
    )
    return () => {
      live = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  return state
}
