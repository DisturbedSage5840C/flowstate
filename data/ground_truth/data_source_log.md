# Data Source Log — Aqua-Sense Ground Truth

## Decision: Proxy Labels (locked at Hour 2)

No geo/time-matched in-situ Chl-a data was found for the chosen sites.
All labels are **literature-calibrated proxy labels** as per the plan's fallback path (Section 3, path 3).
This is openly disclosed.

---

## Sites and Sources

| Site | Type | State | Data Path | Notes |
|---|---|---|---|---|
| Bellandur | lake | Karnataka | `proxy_nechad_ndci` | NDCI-based Chl-a (Gitelson 1992); Nechad turbidity; DO inferred from eutrophication events |
| Varthur | lake | Karnataka | `proxy_nechad_ndci` | Same as Bellandur; froth incidents (2015–2024) cross-checked |
| Ulsoor | lake | Karnataka | `proxy_nechad_ndci` | Cleaner lake; lower Chl-a range |
| Yamuna Delhi | river | Delhi | `cpcb_wqms_proxy` | CPCB WQMS stations at Palla, Nizamuddin; BOD/DO published online; Chl-a proxied via OC3 |
| Ganga Kanpur | river | UP | `cpcb_nwmp_proxy` | CPCB NWMP data; Chl-a proxied; monsoon turbidity spike cross-checked |
| Ganga Varanasi | river | UP | `cpcb_nwmp_proxy` | Same as Kanpur |
| Sutlej Ludhiana | river | Punjab | `cpcb_ppcb_proxy` | PPCB reports; high industrial load |
| Buddha Nullah | river | Punjab | `cpcb_ppcb_proxy` | Extremely polluted drain; BOD >> 100 mg/L documented |
| Beas Amritsar | river | Punjab | `proxy_nechad_ndci` | Relatively cleaner; agricultural runoff |
| Ghaggar Patiala | river | Punjab | `proxy_nechad_ndci` | Seasonal; dry stretches excluded |
| Hussain Sagar | lake | Telangana | `proxy_nechad_ndci` | Urban eutrophic lake; Ganesh idol immersion events |
| Dal Lake | lake | J&K | `proxy_nechad_ndci` | Tourism pressure; NDCI moderate |
| Chilika | lagoon | Odisha | `proxy_nechad_ndci` | Saline lagoon; seasonal salinity affects Chl-a proxy |

---

## Proxy Algorithms Used

### Chlorophyll-a (Chl-a, µg/L)
- **NDCI** (Normalized Difference Chlorophyll Index): `(B5 - B4) / (B5 + B4)`  
  Empirical: `Chl-a = 10^(1.35 * NDCI + 1.58)` (Mishra & Mishra 2012, calibrated for Indian inland waters)
- For rivers with weak Chl-a signal, OC3-style: `log(Chl-a) = a0 + a1*R + a2*R² + a3*R³` where R = log(max(B3,B4)/B5)

### Turbidity (FNU/NTU)
- **Nechad et al. (2010):** `T = A_T * ρ_w(B4) / (1 - ρ_w(B4)/C_T) + B_T`
  - Default: A_T = 228.1, B_T = 0.1641, C_T = 0.1728
- **NIR branch** (high turbidity > 50 FNU): `T = A_T * ρ_w(B8) / (1 - ρ_w(B8)/C_T) + B_T`
  - Switch point calibrated per region (monsoon rivers use NIR branch)

### Dissolved Oxygen (DO, mg/L)
- **Surrogate model inputs:** Chl-a (photosynthesis proxy), Turbidity (light penetration), and season/month
- Empirical baseline: `DO = 14.62 - 0.3898*T_water - 0.006969*salinity + f(Chl-a)`
  - T_water estimated from Landsat Band 10 (LST) or climatological mean
- Cross-checked: Buddha Nullah DO < 1 mg/L (documented); Bellandur DO < 2 mg/L in summer

### BOD (mg/L)
- Where CPCB data available, used directly
- Proxy: `BOD ≈ 0.35 * (Chl-a^0.72) * turbidity_factor` (empirical calibration for Indian urban lakes)

---

## Matching Rule
Join satellite features to station readings within ±3 days (Sentinel-2) or ±5 days (Landsat),
within 500 m radius of the station.

---

## Sites Dropped
- Sites with < 3 matched pairs excluded (none dropped at this stage since all are proxy)
- Ghaggar dry sections excluded from feature join

---

## Disclosure for Pitch
> "All training labels are literature-calibrated proxy labels derived from empirical remote-sensing algorithms (NDCI, Nechad turbidity) and CPCB published BOD/DO values. The ML model acts as a nonlinear corrector on top of the empirical baseline. We validate against documented pollution events (Bellandur froth, Yamuna foam at Kalindi Kunj, Buddha Nullah) rather than withheld in-situ data. Real in-situ validation would be the next step for production deployment."
