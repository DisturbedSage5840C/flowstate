import { useState, type ReactNode } from 'react'
import { RISK, riskWord, type Risk } from './data'

/* ─────────────────────────── Icons (stroke, 1.6) ─────────────────────────── */
const S = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.6, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const }
export function Icon({ name, className = 'size-4', style }: { name: string; className?: string; style?: React.CSSProperties }) {
  const p: Record<string, ReactNode> = {
    search: <><circle cx="11" cy="11" r="7" {...S} /><path d="m20 20-3.2-3.2" {...S} /></>,
    layers: <><path d="M12 3 3 8l9 5 9-5-9-5Z" {...S} /><path d="m3 13 9 5 9-5" {...S} /></>,
    bell: <><path d="M6 9a6 6 0 1 1 12 0c0 5 2 6 2 6H4s2-1 2-6Z" {...S} /><path d="M10 20a2 2 0 0 0 4 0" {...S} /></>,
    chevron: <path d="m6 9 6 6 6-6" {...S} />,
    close: <><path d="m6 6 12 12M18 6 6 18" {...S} /></>,
    drain: <><path d="M4 8h16M6 8v8a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V8" {...S} /><path d="M10 12v3M14 12v3" {...S} /></>,
    stp: <><rect x="4" y="9" width="16" height="11" rx="1" {...S} /><circle cx="9" cy="14" r="2" {...S} /><circle cx="15" cy="14" r="2" {...S} /></>,
    ind: <><path d="M3 20V10l6 4V10l6 4V6l6 2v12H3Z" {...S} /></>,
    built: <><rect x="4" y="8" width="7" height="12" {...S} /><rect x="13" y="4" width="7" height="16" {...S} /><path d="M7 11v.01M7 15v.01M16 8v.01M16 12v.01" {...S} /></>,
    sparkles: <><path d="M12 4v4M12 16v4M4 12h4M16 12h4" {...S} /><path d="m7 7 2 2M17 7l-2 2M7 17l2-2M17 17l-2-2" {...S} /></>,
    up: <path d="M12 19V5m0 0-6 6m6-6 6 6" {...S} />,
    down: <path d="M12 5v14m0 0 6-6m-6 6-6-6" {...S} />,
    flat: <path d="M5 12h14" {...S} />,
    target: <><circle cx="12" cy="12" r="8" {...S} /><circle cx="12" cy="12" r="3" {...S} /><path d="M12 2v3M12 19v3M2 12h3M19 12h3" {...S} /></>,
    arrow: <path d="M5 12h14m0 0-6-6m6 6-6 6" {...S} />,
    reset: <><path d="M4 4v6h6" {...S} /><path d="M4 10a8 8 0 1 1-1 4" {...S} /></>,
    globe: <><circle cx="12" cy="12" r="9" {...S} /><path d="M3 12h18M12 3c3 3.5 3 14.5 0 18M12 3c-3 3.5-3 14.5 0 18" {...S} /></>,
    city: <><path d="M3 21h18" {...S} /><path d="M5 21V7l6-3v17" {...S} /><path d="M11 21V10l6 2v9" {...S} /><path d="M8 9v.01M8 13v.01M8 17v.01M14 14v.01M14 17v.01" {...S} /></>,
    wave: <><path d="M2 8c2 0 2 1.6 4 1.6S8 8 10 8s2 1.6 4 1.6S16 8 18 8s2 1.6 4 1.6" {...S} /><path d="M2 13c2 0 2 1.6 4 1.6S8 13 10 13s2 1.6 4 1.6S16 13 18 13s2 1.6 4 1.6" {...S} /><path d="M2 18c2 0 2 1.6 4 1.6S8 18 10 18s2 1.6 4 1.6S16 18 18 18s2 1.6 4 1.6" {...S} /></>,
  }
  return <svg viewBox="0 0 24 24" className={className} style={style} aria-hidden>{p[name]}</svg>
}

const TREND: Record<string, string> = { up: 'up', down: 'down', flat: 'flat' }

/* ─────────────────────────── RiskBadge ─────────────────────────── */
export function RiskBadge({ risk, label, size = 'md' }: { risk: Risk; label?: string; size?: 'sm' | 'md' }) {
  const c = RISK[risk].hex
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-[3px] font-mono font-semibold uppercase tracking-wider ${size === 'sm' ? 'px-1.5 py-0.5 text-[10px]' : 'px-2 py-1 text-[11px]'}`}
      style={{ color: c, background: `color-mix(in oklab, ${c} 16%, transparent)`, boxShadow: `inset 0 0 0 1px color-mix(in oklab, ${c} 40%, transparent)` }}
    >
      <span className="size-1.5 rounded-full" style={{ background: c }} />
      {label ?? `${riskWord(risk)} RISK`}
    </span>
  )
}

/* Guidance pills — never a pass/fail verdict */
export function GuidancePill({ children }: { children: ReactNode }) {
  return (
    <span className="eyebrow inline-flex items-center gap-1 rounded-[3px] bg-[color-mix(in_oklab,var(--color-accent)_14%,transparent)] px-1.5 py-0.5 text-[var(--color-accent)]">
      {children}
    </span>
  )
}

/* ─────────────────────────── WHY? disclosure (reusable) ─────────────────────────── */
export function WhyDisclosure({ signals }: { signals: string[] }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="rounded-[4px] border border-[var(--color-hair)] bg-[var(--color-panel-2)]">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-3 py-2 text-left"
      >
        <span className="eyebrow flex items-center gap-1.5 text-[var(--color-accent)]">
          <Icon name="sparkles" className="size-3.5" /> Why is this flagged?
        </span>
        <Icon name="chevron" className={`size-4 text-[var(--color-mute)] transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <ul className="fs-fade space-y-1.5 border-t border-[var(--color-hair)] px-3 py-2.5">
          <li className="text-[11px] leading-snug text-[var(--color-mute)]">Contributing spatial signals — spatial correlation, not a causal claim:</li>
          {signals.map((s) => (
            <li key={s} className="flex items-start gap-2 text-[12.5px] text-[var(--color-ink)]">
              <span className="mt-1.5 size-1 shrink-0 rounded-full bg-[var(--color-accent)]" />
              {s}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/* ─────────────────────────── Distance-list row (reusable) ─────────────────────────── */
export function DistanceRow({ dist, kind, icon }: { dist: string; kind: string; icon: string }) {
  return (
    <li className="flex items-center gap-3 py-1.5">
      <span className="grid size-7 shrink-0 place-items-center rounded-[4px] border border-[var(--color-hair)] bg-[var(--color-panel-2)] text-[var(--color-mute)]">
        <Icon name={icon} className="size-3.5" />
      </span>
      <span className="tnum w-14 shrink-0 text-[12px] text-[var(--color-accent)]">{dist}</span>
      <span className="text-[13px] text-[var(--color-ink)]">{kind}</span>
    </li>
  )
}

/* ─────────────────────────── Recommendation card (reusable) ─────────────────────────── */
export function RecommendationCard({ n, title, evidence, action }: { n: string; title: string; evidence: string; action: string }) {
  return (
    <article className="group rounded-[5px] border border-[var(--color-hair)] bg-[var(--color-panel)] p-4 transition-colors hover:border-[var(--color-hair-2)]">
      <div className="flex items-center gap-2">
        <span className="tnum text-[13px] text-[var(--color-mute-2)]">{n}</span>
        <h4 className="font-mono text-[13px] font-semibold uppercase tracking-wide text-[var(--color-ink)]">{title}</h4>
      </div>
      <p className="mt-2 text-[13px] leading-relaxed text-[var(--color-mute)]">{evidence}</p>
      <p className="mt-2 text-[13px] leading-relaxed text-[var(--color-ink)]">
        <span className="text-[var(--color-risk-low)]">›</span> {action}
      </p>
    </article>
  )
}

/* ─────────────────────────── Trend arrow ─────────────────────────── */
export function Trend({ dir }: { dir: 'up' | 'down' | 'flat' }) {
  return <Icon name={TREND[dir]} className="size-4" />
}
/* `badWhen`: the direction that means deterioration for this indicator (rising BOD is bad; falling DO is bad) */
export function TrendCell({ dir, badWhen = 'up' }: { dir: 'up' | 'down' | 'flat'; badWhen?: 'up' | 'down' }) {
  const color = dir === 'flat' ? 'var(--color-mute)' : dir === badWhen ? 'var(--color-risk-high)' : 'var(--color-risk-low)'
  return <span style={{ color }}><Trend dir={dir} /></span>
}

/* ─────────────────────────── Legend chip ─────────────────────────── */
export function LegendChip({ note }: { note?: string }) {
  return (
    <div className="rounded-[4px] border border-[var(--color-hair)] bg-[color-mix(in_oklab,var(--color-panel)_82%,transparent)] px-3 py-2 backdrop-blur">
      <div className="flex items-center gap-3">
        {(['low', 'mod', 'high'] as Risk[]).map((r) => (
          <span key={r} className="flex items-center gap-1.5 text-[11px] text-[var(--color-mute)]">
            <span className="size-2 rounded-full" style={{ background: RISK[r].hex }} />
            {RISK[r].label}
          </span>
        ))}
      </div>
      {note && <p className="mt-1 max-w-[260px] text-[10px] leading-snug text-[var(--color-mute-2)]">{note}</p>}
    </div>
  )
}

/* ─────────────────────────── Infrastructure chain (vertical schematic) ─────────────────────────── */
export function InfraChain({ steps }: { steps: { icon: string; label: string; sub: string }[] }) {
  return (
    <div className="rounded-[5px] border border-[var(--color-hair)] bg-[var(--color-panel)] p-4">
      <p className="eyebrow mb-3 text-[var(--color-mute-2)]">Chain of proximity</p>
      <div className="flex flex-col items-start gap-1">
        {steps.map((s, i) => (
          <div key={s.label} className="w-full">
            <div className="flex items-center gap-3">
              <span className="grid size-9 place-items-center rounded-[4px] border border-[var(--color-hair-2)] bg-[var(--color-panel-2)] text-[var(--color-accent)]">
                <Icon name={s.icon} className="size-4" />
              </span>
              <div>
                <p className="text-[13px] text-[var(--color-ink)]">{s.label}</p>
                <p className="tnum text-[11px] text-[var(--color-mute-2)]">{s.sub}</p>
              </div>
            </div>
            {i < steps.length - 1 && <span className="ml-4 block h-4 w-px bg-[var(--color-hair-2)]" />}
          </div>
        ))}
      </div>
      <p className="mt-3 text-[11px] leading-snug text-[var(--color-mute-2)]">
        Shows spatial proximity to major cities only (no drain, sewage or STP inventory is available), not proven causation.
      </p>
    </div>
  )
}
