export type Risk = 'low' | 'mod' | 'high'
export type Mode = 'plan' | 'restore'

export const RISK: Record<Risk, { hex: string; label: string }> = {
  low: { hex: 'var(--color-risk-low)', label: 'Lower risk' },
  mod: { hex: 'var(--color-risk-mod)', label: 'Moderate' },
  high: { hex: 'var(--color-risk-high)', label: 'High' },
}

/* Concrete hex (kept in sync with theme tokens) for Leaflet vector styling */
export const RISK_HEX: Record<Risk, string> = {
  low: '#2f7d5b',
  mod: '#c0872d',
  high: '#bc493a',
}

/* Colour for a station with no reading for the active layer */
export const NO_DATA_HEX = '#b9b6ab'

export const riskWord = (r: Risk) => (r === 'high' ? 'HIGH' : r === 'mod' ? 'MODERATE' : 'LOWER')

export type Layer = { id: string; label: string; swatch: string; hint: string; kind?: 'color' | 'base' | 'overlay'; soon?: boolean }
export type LayerGroup = { group: string; layers: Layer[] }

/* `kind` says how a live layer behaves: 'color' (what stations are coloured by), 'base' (which basemap),
   'overlay' (toggle on top). Layers flagged `soon` have no data source in the backend yet. */
export const LAYER_GROUPS: LayerGroup[] = [
  {
    group: 'Water',
    layers: [
      { id: 'overall', kind: 'color', label: 'Overall Water Risk', swatch: 'var(--color-risk-high)', hint: 'Screening model: P(BOD > 3 mg/L)' },
      { id: 'turb', kind: 'color', label: 'Turbidity', swatch: '#c084fc', hint: 'Measured, latest visit' },
      { id: 'bod', kind: 'color', label: 'BOD', swatch: '#f472b6', hint: 'Measured, latest visit' },
      { id: 'do', kind: 'color', label: 'Dissolved oxygen', swatch: '#38bdf8', hint: 'Measured, latest visit' },
      { id: 'hist', kind: 'color', label: 'Historical Change', swatch: '#94a3b8', hint: 'WQI vs earlier visits' },
      { id: 'pred', label: 'Predicted Change', swatch: '#fb923c', hint: 'No forecasting model yet', soon: true },
    ],
  },
  {
    group: 'Satellite',
    layers: [
      { id: 's2t', kind: 'base', label: 'Sentinel-2 True Colour', swatch: '#60a5fa', hint: 'Cloudless mosaic, 10 m' },
      { id: 'esri', kind: 'base', label: 'High-res imagery', swatch: '#f87171', hint: 'Esri World Imagery' },
      { id: 'ndwi', label: 'NDWI', swatch: '#22d3ee', hint: 'Not served yet', soon: true },
      { id: 'ndci', label: 'NDCI', swatch: '#4ade80', hint: 'Chlorophyll — not modeled', soon: true },
      { id: 'ndti', label: 'NDTI', swatch: '#a78bfa', hint: 'Not served yet', soon: true },
    ],
  },
  {
    group: 'Urban Infrastructure',
    layers: [
      { id: 'cities', kind: 'overlay', label: 'Major cities', swatch: '#475569', hint: 'Urban-load proxy input' },
      { id: 'drain', label: 'Drainage network', swatch: '#38bdf8', hint: 'No data source', soon: true },
      { id: 'sewage', label: 'Sewage outlets', swatch: '#f4523b', hint: 'No data source', soon: true },
      { id: 'stp', label: 'STPs', swatch: '#34d399', hint: 'No data source', soon: true },
      { id: 'ind', label: 'Industrial discharge points', swatch: '#fbbf24', hint: 'No data source', soon: true },
    ],
  },
  {
    group: 'Planning',
    layers: [
      { id: 'land', label: 'Land use', swatch: '#a3e635', hint: 'No data source', soon: true },
      { id: 'flood', label: 'Flood-prone areas', swatch: '#22d3ee', hint: 'No data source', soon: true },
      { id: 'pop', label: 'Population density', swatch: '#f97316', hint: 'No data source', soon: true },
    ],
  },
]

export const ASK_CHIPS = [
  'Where are the major water-quality hotspots?',
  'Show me stations where water quality has deteriorated.',
  'Show hotspots within 25 km of a major city.',
  'Which lakes have BOD above the limit?',
  'Show hotspots within 2 km of sewage outlets.',
]
