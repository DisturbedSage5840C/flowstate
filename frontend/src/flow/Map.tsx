import { useEffect, useMemo } from 'react'
import { CircleMarker, MapContainer, TileLayer, Tooltip, useMap, useMapEvents } from 'react-leaflet'
import L from 'leaflet'
import { NO_DATA_HEX, RISK_HEX, type Risk } from './data'
import type { City, Station } from './api'

// Esri light-gray canvas — key-free, matches the light editorial ground
const GRAY_BASE = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'
const GRAY_LABELS = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}'
const ESRI_IMAGERY = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
// Sentinel-2 cloudless 2020 mosaic (EOX). Attribution is required by the licence.
const S2_CLOUDLESS = 'https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg'
const ATTR = '© Esri · HERE · Garmin · OpenStreetMap contributors'
const S2_ATTR = 'Sentinel-2 cloudless — s2maps.eu by EOX IT Services GmbH (contains modified Copernicus Sentinel data 2020)'

export type Basemap = 'gray' | 's2t' | 'esri'
export type ColorBy = 'overall' | 'turb' | 'bod' | 'do' | 'hist'

const INDIA_CENTER: [number, number] = [22.5, 79]

export function stationColor(s: Station, by: ColorBy): string {
  switch (by) {
    case 'turb':
      return s.rt ? RISK_HEX[s.rt] : NO_DATA_HEX
    case 'bod':
      return s.rb ? RISK_HEX[s.rb] : NO_DATA_HEX
    case 'do':
      return s.rd ? RISK_HEX[s.rd] : NO_DATA_HEX
    case 'hist':
      return s.ch === 'worse' ? RISK_HEX.high : s.ch === 'better' ? RISK_HEX.low : NO_DATA_HEX
    default:
      return RISK_HEX[s.risk]
  }
}

function riskDot(risk: Risk) {
  return { color: '#ffffff', weight: 2, fillColor: RISK_HEX[risk], fillOpacity: 1 }
}

/* ── Landing: real map of India, non-interactive backdrop, one dot per city with station coverage ── */
export function IndiaMap({ cities }: { cities: City[] }) {
  return (
    <div className="fs-static absolute inset-0">
      <MapContainer
        center={INDIA_CENTER}
        zoom={5}
        zoomControl={false}
        attributionControl={false}
        dragging={false}
        scrollWheelZoom={false}
        doubleClickZoom={false}
        touchZoom={false}
        keyboard={false}
        className="size-full"
        style={{ background: 'var(--color-abyss)' }}
      >
        <TileLayer url={GRAY_BASE} maxZoom={16} />
        {cities.map((c) => (
          <CircleMarker key={c.name} center={[c.lat, c.lon]} radius={4 + Math.min(6, Math.sqrt(c.stations) / 1.6)} pathOptions={riskDot(c.risk)}>
            <Tooltip direction="right" offset={[8, 0]} opacity={1} className="fs-tooltip">
              {c.name} · {c.stations} stations · {Math.round(c.high_share * 100)}% high risk
            </Tooltip>
          </CircleMarker>
        ))}
      </MapContainer>
    </div>
  )
}

/* Flies to a target whenever `nonce` changes; fits bounds when `bounds` is given */
function Fly({ target }: { target: { lat: number; lon: number; zoom?: number; nonce: number } | null }) {
  const map = useMap()
  useEffect(() => {
    if (target) map.flyTo([target.lat, target.lon], target.zoom ?? 11, { duration: 0.9 })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target?.nonce])
  return null
}

function FitIds({ stations, ids, nonce }: { stations: Station[]; ids: Set<string> | null; nonce: number }) {
  const map = useMap()
  useEffect(() => {
    if (!ids || ids.size === 0) return
    const pts = stations.filter((s) => ids.has(s.id)).map((s) => [s.lat, s.lon] as [number, number])
    if (pts.length) map.flyToBounds(L.latLngBounds(pts).pad(0.2), { maxZoom: 10, duration: 0.9 })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nonce])
  return null
}

function BaseTiles({ base }: { base: Basemap }) {
  if (base === 's2t') return <TileLayer key="s2t" url={S2_CLOUDLESS} maxZoom={13} attribution={S2_ATTR} />
  if (base === 'esri') return <TileLayer key="esri" url={ESRI_IMAGERY} maxZoom={18} attribution="Imagery © Esri, Maxar, Earthstar Geographics" />
  return (
    <>
      <TileLayer key="g" url={GRAY_BASE} maxZoom={16} attribution={ATTR} />
      <TileLayer key="gl" url={GRAY_LABELS} maxZoom={16} />
    </>
  )
}

export type FlyTarget = { lat: number; lon: number; zoom?: number; nonce: number }

/* ── Workspace: every station, coloured by the active layer ── */
export function WaterMap({
  stations,
  colorBy,
  base,
  cities,
  selected,
  secondary,
  filterIds,
  filterNonce,
  fly,
  onSelect,
  onMove,
}: {
  stations: Station[]
  colorBy: ColorBy
  base: Basemap
  cities: City[] | null
  selected: string | null
  secondary: string | null
  filterIds: Set<string> | null
  filterNonce: number
  fly: FlyTarget | null
  onSelect: (id: string) => void
  onMove?: (lat: number, lon: number) => void
}) {
  // draw order: dim/filtered-out first, then low -> high so the risky stations sit on top
  const ordered = useMemo(() => {
    const rank = (s: Station) => (filterIds && !filterIds.has(s.id) ? -1 : s.risk === 'high' ? 2 : s.risk === 'mod' ? 1 : 0)
    return [...stations].sort((a, b) => rank(a) - rank(b))
  }, [stations, filterIds])

  return (
    <div className="absolute inset-0 isolate">
      <MapContainer center={INDIA_CENTER} zoom={5} minZoom={4} zoomControl={false} attributionControl scrollWheelZoom preferCanvas className="size-full">
        <BaseTiles base={base} />
        <Fly target={fly} />
        <FitIds stations={stations} ids={filterIds} nonce={filterNonce} />
        {onMove && <MoveReporter onMove={onMove} />}
        {cities?.map((c) => (
          <CircleMarker key={c.name} center={[c.lat, c.lon]} interactive={false} radius={11} pathOptions={{ color: '#475569', weight: 1.2, dashArray: '3 3', fill: false, opacity: 0.8 }} />
        ))}
        {ordered.map((s) => {
          const dim = !!filterIds && !filterIds.has(s.id)
          const active = s.id === selected || s.id === secondary
          return (
            <CircleMarker
              key={s.id}
              center={[s.lat, s.lon]}
              radius={active ? 10 : dim ? 3 : 5}
              pathOptions={{
                color: active ? '#23262b' : '#ffffff',
                weight: active ? 2.5 : 1,
                fillColor: dim ? NO_DATA_HEX : stationColor(s, colorBy),
                fillOpacity: dim ? 0.35 : 0.92,
              }}
              eventHandlers={{ click: () => onSelect(s.id) }}
            >
              <Tooltip direction="top" offset={[0, -6]} opacity={1} className="fs-tooltip">
                {s.name} · {s.state}
              </Tooltip>
            </CircleMarker>
          )
        })}
      </MapContainer>
    </div>
  )
}

function MoveReporter({ onMove }: { onMove: (lat: number, lon: number) => void }) {
  useMapEvents({
    moveend: (e) => {
      const c = e.target.getCenter()
      onMove(c.lat, c.lng)
    },
  })
  return null
}

/* ── Sentinel explorer: satellite basemap, optional model-risk markers ── */
export function SatelliteMap({ base, stations, onMove, onSelect }: { base: Basemap; stations: Station[] | null; onMove: (lat: number, lon: number) => void; onSelect: (id: string) => void }) {
  return (
    <div className="absolute inset-0 isolate">
      <MapContainer center={INDIA_CENTER} zoom={5} minZoom={4} zoomControl attributionControl scrollWheelZoom preferCanvas className="size-full">
        <BaseTiles base={base} />
        <MoveReporter onMove={onMove} />
        {stations?.map((s) => (
          <CircleMarker key={s.id} center={[s.lat, s.lon]} radius={5} pathOptions={{ color: '#fff', weight: 1, fillColor: RISK_HEX[s.risk], fillOpacity: 0.95 }} eventHandlers={{ click: () => onSelect(s.id) }}>
            <Tooltip direction="top" offset={[0, -6]} opacity={1} className="fs-tooltip">
              {s.name} · {s.state}
            </Tooltip>
          </CircleMarker>
        ))}
      </MapContainer>
    </div>
  )
}
