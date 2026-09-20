# Flow State — Figma design brief

Source: `Prayas hackathon.docx` (the "MAP OUT UI" brainstorm + the two-portal redesign notes).
Purpose of this file: give you one paste-ready prompt for Claude in Figma / Figma Make, plus the
reasoning behind it, so the generated UI actually matches what the doc asked for instead of a
generic map dashboard.

---

## 1. What the doc actually asks for (condensed)

- **One platform, two modes** — not two separate sites: *Plan & Prevent* (for municipal/urban
  planners) and *Restore & Prioritize* (for water authorities/policymakers).
- **Map-first, not chart-first.** Landing page is a giant India map, not a graph dashboard. ~70%
  of every working screen is map.
- **Core interaction loop, repeated everywhere:** search/select a location → see a red/yellow/green
  risk overlay → click a hotspot → see *why* it's flagged (evidence + nearby infrastructure) → get
  an "investigate this" action. Never a bare number like "Turbidity = 82 NTU" with no context.
- **Causal honesty is a hard product constraint, not a nice-to-have.** The doc is explicit and
  repeats this multiple times: never say "a factory here WILL pollute the river" or "AI says build
  an STP here." Always phrase things as *planning risk*, *investigation priority*, or
  *AI-assisted recommendation* — spatial correlation, not a causal claim the model can't back up.
  The doc's own closing line is the product promise: **"Flow State doesn't tell a city what to
  build. It tells them where they need to look."**
- **A "Why?" affordance on every AI output**, expanding into the 2-4 contributing spatial signals
  (e.g. drainage connectivity, historical recurrence) — because a government user won't act on a
  black box.
- **Layer-driven GIS workspace**, not fixed views: water-quality layers, satellite layers
  (Sentinel-2 true/false colour, NDWI/NDCI/NDTI), infrastructure layers (drains, sewage outlets,
  STPs, industrial discharge), planning layers (land use, flood-prone, population density).
- **A time slider** (2024 → 2025 → 2026) that visibly changes hotspot colour over time — the doc
  calls this out as more powerful than any chart.
- **"Ask Flow State"** — a natural-language box that filters/manipulates the map itself (e.g. "show
  hotspots within 2 km of sewage outlets"), not a chatbot bolted on next to the product.

## 2. Data reality check — read this before writing the prompt

The doc's mockups show **Turbidity, Chlorophyll-a and pH** as live indicators everywhere. The
actual Aqua-Sense pipeline (this repo) only has real, validated model output for **DO, BOD and
turbidity** — pH has no source column at all, and chlorophyll-a only exists in the synthetic demo
table, not the real CPCB+Sentinel-2 one (see `src/models/schema.py`: `REAL_TARGET_COLS = ["do",
"bod", "turbidity"]`). The repo's own README is built around the same honesty rule this doc is
asking for in the product: it reports an AUC ≈ 0.72–0.78 pollution-screening result and is explicit
that exact-value regression is near-zero skill — a triage tool, not a lab replacement.

So for the Figma design, pick one of two honest paths (the prompt below defaults to the first):

- **A. Ship what's real.** Design every data-driven panel around DO, BOD and turbidity only, with
  a pollution-risk band (screening output) as the headline number instead of a fabricated exact
  reading. Drop pH/chlorophyll-a from the mockup, or keep them only as greyed-out "layer coming
  soon" toggles in the Layers panel — present as UI chrome, never populated with fake numbers.
- **B. Design the full vision, labelled.** Keep pH/chlorophyll-a in the mockups (fine for a pitch
  deck), but tag every panel that shows them with a small "sample data" / "not yet modeled" badge,
  so nobody mistakes a Figma mock for a working feature.

Tell the assistant which path you want before you paste the prompt — the prompt below is written
for **Path A**, since it matches this repo's actual results and its existing "honest, not
oversold" culture.

## 3. Prompt for Claude in Figma

Paste everything in the fenced block below as one message.

```
Design a complete UI for "Flow State" — a satellite water-intelligence platform for Indian urban
water bodies. Audience: municipal planners and water-authority policymakers, viewed on desktop
(this is a GIS command-console product, not a mobile app). Style direction: think Google Earth
crossed with a scientific GIS dashboard and a government-grade command console — map-dominant,
information-dense but uncluttered, semantic red/amber/green risk colour coding used consistently
everywhere, clean sans-serif type for UI chrome with a technical/monospace accent font for data
readouts (coordinates, percentages, dates). Support a dark theme as the primary theme (satellite
imagery and map overlays read better on dark) with a light theme as a secondary option.

PRODUCT STRUCTURE: one platform, two modes, switchable from a persistent header — not two separate
products. Build these screens:

── 1. LANDING PAGE ──
Full-bleed map of India as the background/hero, not a chart. Header: "FLOW STATE — Satellite
intelligence for India's urban waters". Two large, equal-weight mode-selector cards over the map:
  • "🏙️ PLAN & PREVENT — For Municipalities & Urban Planners. Explore an area, identify
    water-quality risks and understand how rivers, drains, sewage infrastructure and land use
    interact." → button "ENTER PLANNING MODE"
  • "🌊 RESTORE & PRIORITIZE — For Water Authorities & Policymakers. Find deteriorating water
    bodies, identify pollution hotspots and prioritize areas for intervention." → button "ENTER
    RESTORATION MODE"
A third, smaller link/button below both cards: "Explore Sentinel-2" (opens the raw satellite layer
without committing to a mode).

── 2. SHARED WORKSPACE SHELL (used by both modes) ──
Layout: persistent top header with the Flow State wordmark, active-mode label, and a location
search bar with a magnifying-glass icon (placeholder: "Search location, river or lake"). Below it,
a two-pane workspace: a collapsible left sidebar (~280px) for the Layers panel, and the map filling
the remaining ~70-75% of the viewport. A slim status bar pinned to the bottom of the map shows the
currently selected area name and its overall risk band (e.g. "Selected area: Mohali — Risk:
MODERATE", colour-coded).

── 3. LAYERS PANEL (left sidebar, checkbox list, grouped with section headers) ──
  WATER: Overall Water Risk (checked by default) · Turbidity · BOD Risk · DO Risk · Historical
  Change · Predicted Change
  SATELLITE: Sentinel-2 True Colour · Sentinel-2 False Colour · NDWI · NDCI · NDTI
  URBAN INFRASTRUCTURE: Drainage network · Sewage outlets · STPs · Industrial discharge points ·
  Roads · Built-up areas
  PLANNING: Land use · Flood-prone areas · Agricultural areas · Existing industrial zones ·
  Population density
Design one layer row as a component: checkbox, label, small colour swatch matching its map symbol,
and a subtle count/legend hint on hover.

── 4. LOCATION SEARCH → RISK MAP (the core interaction, build this flow in detail) ──
User types a location (e.g. "Varthur Lake, Bengaluru") into the header search bar. The map zooms
in and the selected water body renders as a segmented heatmap using ONLY three risk colours: red
(higher predicted pollution risk), amber (moderate), green (lower risk) — no gradient, no
continuous colour scale, discrete segments only, with a small legend chip "🟢 Lower risk 🟡
Moderate 🔴 High" pinned in a corner of the map.

── 5. HOTSPOT DETAIL PANEL ("click any red zone") ──
Clicking a red/amber segment slides in a right-hand panel (~360px) titled "WATER QUALITY ALERT" (or
"WATER QUALITY HOTSPOT" for view-only exploration). Structure, top to bottom:
  1. Location line: water body + rough position (e.g. "Sutlej River — X km upstream").
  2. A small 3-row table: indicator / current reading / trend arrow, for Turbidity, BOD risk, DO
     risk (NOT pH or chlorophyll-a — see the data-reality note above; if you want those two shown
     anyway, render them in a visually distinct "sample data" style, greyed slightly, with a small
     "not yet modeled" tag).
  3. "WHAT'S AROUND IT?" — a short distance list with icons, e.g. "420 m — Drain outlet", "1.2 km —
     Sewage treatment plant", "2.1 km — Industrial area", "600 m — Dense built-up area".
  4. A one-line plain-English rationale, e.g. "Water-quality deterioration overlaps with a
     drainage corridor and lies downstream of an identified discharge point."
  5. A small "WHY?" expandable/disclosure control that reveals 2-4 bullet "contributing spatial
     signals" (e.g. "Water-quality deterioration", "Nearby drainage connectivity", "Historical
     recurrence") — this must appear on every AI-flagged panel in the whole product, design it once
     as a reusable component.
  6. Primary action button: "PRIORITIZE FOR FIELD INVESTIGATION" — never a button that claims to
     "fix" or "build" anything.

── 6. PLAN & PREVENT MODE — SITE RISK / COMPARISON ──
A "Planning Risk" tool: the user draws or selects a proposed development site on the map. Output
card, titled "SITE A":
  "WATER-QUALITY RISK: HIGH" (colour-coded badge)
  Key factors, each a coloured bullet (🔴 for hard constraints, 🟠 for softer ones): "0.8 km from
  water body", "Connected to upstream drainage network", "Historical deterioration detected",
  "Limited nearby treatment capacity".
  A "Planning consideration" line in plain prose: "Further environmental assessment and
  infrastructure capacity review recommended before development." (Never a directive like "Do not
  build here.")
Design a two-column "Compare Sites" variant of this same card (Site A vs Site B side by side, same
structure, a "VIEW SPATIAL COMPARISON" button below) — reuse the single-site card as the unit.

── 7. RESTORE & PRIORITIZE MODE — PRIORITY LIST + RIVER DETAIL ──
Landing view for this mode: India map with water bodies coloured red/amber/green by condition,
plus a ranked side list "TOP PRIORITY AREAS" — 3-5 rows, each a river/lake name + one-line reason
(e.g. "River X — Water-quality deterioration detected"), styled as priority *categories*, not a
precise numbered national ranking (avoid implying false precision).
Clicking a river opens a detail screen: big map + right panel "SUTLEJ — WATER INTELLIGENCE" with a
"CURRENT STATE" mini-table (Turbidity / BOD-risk / DO-risk, arrows for trend) and a "TREND" row of
three small thumbnail maps labelled with years, showing the same area getting more/less red.

── 8. INFRASTRUCTURE OVERLAY VISUAL ──
A vertical schematic component (usable both as an illustration and as a literal toggle-driven map
state): industrial area icon → arrow down → sewage/drain icon → arrow down → hotspot marker → a
river band → showing the causal *chain of proximity* (not causation) visually. Use this same
visual language wherever "what's around the hotspot" needs explaining.

── 9. INTERVENTION / RECOMMENDATION CARDS (Restore & Prioritize mode) ──
A numbered list of recommendation cards, each: a short title in caps ("01 — INVESTIGATE
DISCHARGE"), one sentence of evidence ("A recurring hotspot overlaps with a downstream drainage
corridor."), and one sentence starting "Suggested action:" — always "suggested/recommended", never
imperative commands to build something. Include 3-4 example cards: Investigate Discharge, Review
STP Capacity, Monitor Upstream, Assess Land-Use Pressure.

── 10. SENTINEL-2 / SATELLITE VIEW ──
A dedicated satellite mode: toggle row "TRUE COLOUR | FALSE COLOUR | WATER QUALITY", a small
metadata strip ("Sentinel-2 — 20 m · Acquisition: 18 Sept 2026 · Cloud cover: 8% · Resolution:
10–20 m"), and a vertical "pipeline" diagram component showing: Real world → Sentinel-2 → AI Model
→ Turbidity/BOD/DO risk → Planning risk (a simple flow, not a chart).

── 11. TIME SLIDER ──
A horizontal slider component pinned to the bottom of the map in both modes, with tick labels
"2024 — 2025 — 2026". Dragging it should be shown (as a design state / interaction note) visibly
recolouring a hotspot marker green → yellow → orange → red across frames, with a caption like
"Deterioration detected over the last 18 months." Design at least 2 states of this (start position,
dragged position) so the interaction is unambiguous to a developer later.

── 12. BEFORE / AFTER (restoration tracking) ──
A simple two-panel comparison card: "BEFORE INTERVENTION" (red-heavy hotspot cluster) → arrow +
"STP / intervention" label → "AFTER" (green/yellow cluster). Same map-thumbnail style as the trend
thumbnails in screen 7.

── 13. ALERTS ──
A notification/alerts panel (bell icon in the header opens a dropdown or side panel): each alert
card has a severity icon, title ("NEW HOTSPOT DETECTED" / "DETERIORATION ALERT"), location + date,
one-line evidence sentence, and a "VIEW HOTSPOT →" or "INVESTIGATE →" link.

── 14. CITY WATER HEALTH SUMMARY ──
A city-level overview screen: title "MOHALI WATER INTELLIGENCE", four stat cards in a row ("12
Water bodies monitored", "7 Emerging hotspots", "4 High-risk drainage corridors", "3 Areas
requiring investigation"), with the full map below.

── 15. "ASK FLOW STATE" ──
A search-style input, styled distinctly from the location search bar (e.g. an icon that suggests
AI/assistant, placeholder "Ask Flow State anything about this area..."), with 3-4 example-query
chips underneath it (e.g. "Where are the major water-quality hotspots?", "Which areas are
downstream of polluted drains?", "Show me areas where water quality has deteriorated."). Design
the empty state, a "thinking" state, and a result state where the map itself gets a filter applied
(show a small toast/banner like "Filtered: hotspots within 2 km of sewage outlets" with a clear
"Reset filter" action).

DESIGN SYSTEM NOTES:
- Risk semantics are the backbone of the whole UI — define a single 3-colour risk scale (red/amber/
  green) once, as a shared token set, and reuse it identically across the map, badges, table rows,
  alerts and cards. Never introduce a fourth ad-hoc risk colour.
- Every AI-generated claim in the UI must visually read as *investigation guidance*, never as a
  verdict — favour badges/labels like "INVESTIGATE", "MONITOR", "ACTION REQUIRED", "PLANNING RISK"
  over anything sounding like a pass/fail grade.
- Build the "WHY?" disclosure, the risk badge, the distance-list row, and the recommendation card
  as reusable components first — nearly every screen above is a different arrangement of these
  four things plus a map.

DELIVERABLES: a landing page, the shared workspace shell with the layers panel, the hotspot detail
panel, the Plan & Prevent site-risk/comparison screens, the Restore & Prioritize priority-list and
river-detail screens, the Sentinel-2 view, the alerts panel, the city health summary, and the Ask
Flow State states — as a connected Figma flow (clickable prototype) from the landing page through
both modes into a hotspot detail panel. Desktop viewport (1440px) only for this pass.
```

## 4. MVP vs. stretch (from the doc, for sequencing the build after design)

**MVP (24-hour scope):** map with India → city → river/lake zoom, Sentinel-2 imagery, the
water-quality risk layer (turbidity/BOD/DO, not pH/chl-a), the time slider, infrastructure overlays
(drainage/sewage/STPs/land use), the hotspot click → "why is this flagged?" panel, and 2-3 canned
recommendation strings ("Prioritize field investigation", "Review nearby discharge", "Assess STP
capacity").

**Stretch:** the Planning Simulator (site risk + compare-two-sites).

**Biggest stretch / demo wow-factor:** "Ask Flow State" actually filtering the map from a typed
query.
