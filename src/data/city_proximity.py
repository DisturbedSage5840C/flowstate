"""Coarse urban-load proxy: distance to, and population near, major Indian cities.

DO/BOD are driven by sewage and industrial discharge, which concentrates near cities, not by
anything a satellite band can see. This is a real, non-satellite signal, but a rough one: the city
list below is a static, hand-compiled snapshot of ~70 major urban agglomerations (approximate 2011
census population figures and city-centre coordinates, from general knowledge -- not a geocoded
dataset), so treat the resulting features as a coarse "how much urban/industrial activity is nearby"
proxy, not a precise pollution-source inventory. A station near a large but unlisted town, or one
whose actual discharge point is far from the city centroid, will be under/over-estimated.

CITIES: (name, lat, lon, population)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CITIES = [
    ("Mumbai", 19.0760, 72.8777, 12_442_373), ("Delhi", 28.7041, 77.1025, 16_787_941),
    ("Bangalore", 12.9716, 77.5946, 8_443_675), ("Hyderabad", 17.3850, 78.4867, 6_809_970),
    ("Ahmedabad", 23.0225, 72.5714, 5_570_585), ("Chennai", 13.0827, 80.2707, 4_681_087),
    ("Kolkata", 22.5726, 88.3639, 4_496_694), ("Surat", 21.1702, 72.8311, 4_467_797),
    ("Pune", 18.5204, 73.8567, 3_124_458), ("Jaipur", 26.9124, 75.7873, 3_046_163),
    ("Lucknow", 26.8467, 80.9462, 2_817_105), ("Kanpur", 26.4499, 80.3319, 2_767_031),
    ("Nagpur", 21.1458, 79.0882, 2_405_421), ("Indore", 22.7196, 75.8577, 1_994_397),
    ("Thane", 19.2183, 72.9781, 1_841_488), ("Bhopal", 23.2599, 77.4126, 1_798_218),
    ("Visakhapatnam", 17.6868, 83.2185, 1_730_320), ("Pimpri-Chinchwad", 18.6298, 73.7997, 1_729_359),
    ("Patna", 25.5941, 85.1376, 1_684_222), ("Vadodara", 22.3072, 73.1812, 1_666_703),
    ("Ghaziabad", 28.6692, 77.4538, 1_648_643), ("Ludhiana", 30.9010, 75.8573, 1_618_879),
    ("Agra", 27.1767, 78.0081, 1_585_704), ("Nashik", 19.9975, 73.7898, 1_486_973),
    ("Faridabad", 28.4089, 77.3178, 1_414_050), ("Meerut", 28.9845, 77.7064, 1_305_429),
    ("Rajkot", 22.3039, 70.8022, 1_286_995), ("Kalyan-Dombivli", 19.2403, 73.1305, 1_246_381),
    ("Vasai-Virar", 19.4914, 72.8054, 1_221_233), ("Varanasi", 25.3176, 82.9739, 1_198_491),
    ("Srinagar", 34.0837, 74.7973, 1_180_570), ("Aurangabad", 19.8762, 75.3433, 1_175_116),
    ("Dhanbad", 23.7957, 86.4304, 1_162_472), ("Amritsar", 31.6340, 74.8723, 1_132_761),
    ("Navi Mumbai", 19.0330, 73.0297, 1_119_477), ("Prayagraj", 25.4358, 81.8463, 1_112_544),
    ("Ranchi", 23.3441, 85.3096, 1_073_440), ("Howrah", 22.5958, 88.2636, 1_077_075),
    ("Coimbatore", 11.0168, 76.9558, 1_061_447), ("Jabalpur", 23.1815, 79.9864, 1_055_525),
    ("Gwalior", 26.2183, 78.1828, 1_053_505), ("Vijayawada", 16.5062, 80.6480, 1_048_240),
    ("Jodhpur", 26.2389, 73.0243, 1_033_918), ("Madurai", 9.9252, 78.1198, 1_017_865),
    ("Raipur", 21.2514, 81.6296, 1_010_087), ("Kota", 25.2138, 75.8648, 1_001_694),
    ("Guwahati", 26.1445, 91.7362, 962_334), ("Chandigarh", 30.7333, 76.7794, 960_787),
    ("Solapur", 17.6599, 75.9064, 951_558), ("Hubli-Dharwad", 15.3647, 75.1240, 943_857),
    ("Bareilly", 28.3670, 79.4304, 903_668), ("Moradabad", 28.8386, 78.7733, 889_810),
    ("Mysore", 12.2958, 76.6394, 887_446), ("Gurgaon", 28.4595, 77.0266, 876_824),
    ("Aligarh", 27.8974, 78.0880, 872_575), ("Jalandhar", 31.3260, 75.5762, 862_196),
    ("Tiruchirappalli", 10.7905, 78.7047, 847_387), ("Bhubaneswar", 20.2961, 85.8245, 837_737),
    ("Salem", 11.6643, 78.1460, 831_038), ("Warangal", 17.9689, 79.5941, 811_844),
    ("Guntur", 16.3067, 80.4365, 743_354), ("Saharanpur", 29.9640, 77.5460, 705_478),
    ("Gorakhpur", 26.7606, 83.3732, 673_446), ("Bikaner", 28.0229, 73.3119, 647_804),
    ("Amravati", 20.9374, 77.7796, 646_801), ("Noida", 28.5355, 77.3910, 642_381),
    ("Jamshedpur", 22.8046, 86.2029, 1_337_131), ("Bhilai", 21.1938, 81.3509, 1_064_222),
    ("Cuttack", 20.4625, 85.8828, 606_007), ("Kochi", 9.9312, 76.2673, 2_119_724),
    ("Bhavnagar", 21.7645, 72.1519, 605_882), ("Dehradun", 30.3165, 78.0322, 578_420),
    ("Durgapur", 23.5204, 87.3119, 566_937), ("Asansol", 23.6739, 86.9524, 563_917),
    ("Kolhapur", 16.7050, 74.2433, 549_236), ("Ajmer", 26.4499, 74.6399, 542_580),
    ("Jamnagar", 22.4707, 70.0577, 529_308), ("Ujjain", 23.1765, 75.7885, 515_215),
    ("Siliguri", 26.7271, 88.3953, 513_264), ("Jhansi", 25.4484, 78.5685, 507_293),
    ("Jammu", 32.7266, 74.8570, 502_197), ("Mangalore", 12.9141, 74.8560, 488_968),
    ("Belgaum", 15.8497, 74.4977, 488_292), ("Tirunelveli", 8.7139, 77.7567, 474_838),
    ("Gaya", 24.7955, 84.9994, 470_839), ("Udaipur", 24.5854, 73.7125, 475_150),
    ("Kozhikode", 11.2588, 75.7804, 609_224), ("Kurnool", 15.8281, 78.0373, 460_184),
    ("Patiala", 30.3398, 76.3869, 405_000), ("Agartala", 23.8315, 91.2868, 400_004),
    ("Bilaspur", 22.0797, 82.1409, 331_030),
]
_CITY_LAT = np.array([c[1] for c in CITIES])
_CITY_LON = np.array([c[2] for c in CITIES])
_CITY_POP = np.array([c[3] for c in CITIES], dtype=np.float64)

EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def urban_proxy_features(lat: pd.Series, lon: pd.Series) -> pd.DataFrame:
    """Per station: distance (km) to the nearest listed city, and an inverse-square population-
    weighted "urban load" index summed over all listed cities (a gravity-model style proxy: closer,
    bigger cities contribute more, distant/small ones contribute little)."""
    lat = np.asarray(lat, dtype=np.float64)[:, None]
    lon = np.asarray(lon, dtype=np.float64)[:, None]
    dist = _haversine_km(lat, lon, _CITY_LAT[None, :], _CITY_LON[None, :])
    nearest_km = dist.min(axis=1)
    load_index = (_CITY_POP[None, :] / (1.0 + dist) ** 2).sum(axis=1)
    return pd.DataFrame({
        "dist_nearest_city_km": nearest_km.astype(np.float32),
        "urban_load_index": np.log1p(load_index).astype(np.float32),
    })
