"""Small, dependency-free water-chemistry helpers shared by features and WQI code."""

from __future__ import annotations

import numpy as np

DEFAULT_TEMP_C = 25.0  # typical Indian surface-water temperature when none is measured


def pressure_ratio(altitude_m) -> np.ndarray:
    """Atmospheric pressure at `altitude_m` relative to sea level (standard atmosphere)."""
    h = np.asarray(altitude_m, dtype=float)
    return np.power(1.0 - 2.25577e-5 * h, 5.25588)


def do_saturation_mg_l(temp_c, altitude_m=0.0) -> np.ndarray:
    """Freshwater dissolved-oxygen saturation concentration (mg/L).

    Benson & Krause (1984) fit as tabulated in APHA Standard Methods 4500-O
    (freshwater, 1 atm):
        ln(DO) = -139.34411 + 1.575701e5/T - 6.642308e7/T^2
                 + 1.243800e10/T^3 - 8.621949e11/T^4          (T in kelvin)
    An altitude term scales by standard-atmosphere pressure (water-vapour
    pressure correction ignored).
    """
    tk = np.asarray(temp_c, dtype=float) + 273.15
    ln_c = (-139.34411 + 1.575701e5 / tk - 6.642308e7 / tk**2
            + 1.243800e10 / tk**3 - 8.621949e11 / tk**4)
    return np.exp(ln_c) * pressure_ratio(altitude_m)


def free_ammonia_n(total_ammonia_n, ph, temp_c) -> np.ndarray:
    """Un-ionised (free) ammonia-N in mg/L from total ammonia-N, pH and temperature.

    Emerson et al. (1975): pKa = 0.09018 + 2729.92/T(K); fraction = 1/(1+10^(pKa-pH)).
    CPCB's Class D criterion is on *free* ammonia (as N), while monitoring
    programmes report total ammonia-N.
    """
    tk = np.asarray(temp_c, dtype=float) + 273.15
    pka = 0.09018 + 2729.92 / tk
    frac = 1.0 / (1.0 + np.power(10.0, pka - np.asarray(ph, dtype=float)))
    return np.asarray(total_ammonia_n, dtype=float) * frac
