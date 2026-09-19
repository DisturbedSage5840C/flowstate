# Enabling Google Earth Engine for this project (project ID: `prayashack`)

**Current state (verified 2026-09-19):** the saved Earth Engine login works and reaches Google, but the
Earth Engine API is switched off for the Cloud project `prayashack`, so every call fails with
"Google Earth Engine API has not been used in project prayashack before or it is disabled".
The results in this repo therefore use Sentinel-2 from Microsoft Planetary Computer, which needs no login.

## 1. Enable the API
1. Sign in to Google with the account that **owns** the `prayashack` project (check the avatar, top right).
2. Open: https://console.developers.google.com/apis/api/earthengine.googleapis.com/overview?project=prayashack
3. Click the blue **ENABLE** button. If it shows **MANAGE**, it is already enabled.

## 2. Register the project for Earth Engine
1. Open: https://code.earthengine.google.com/register
2. Choose **Register a Noncommercial or Commercial Cloud project**. Non-commercial (unpaid) use is free and needs no billing.
3. Select the non-commercial option and the category that fits (for example **Academia & Research**).
4. Choose **an existing Google Cloud project** and select `prayashack`, then continue.
5. Confirm on the summary page. (Button labels can differ slightly from this text.)

## 3. Wait 2-5 minutes
Google needs a few minutes to propagate the change.

## 4. Test it
From the repo folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -c "import ee; ee.Initialize(project='prayashack'); print(ee.Number(1).add(1).getInfo())"
```

It must print `2`. A stronger check that real imagery is reachable:

```powershell
.\.venv\Scripts\python.exe -c "import ee; ee.Initialize(project='prayashack'); print(ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate('2020-01-20','2020-02-05').filterBounds(ee.Geometry.Point(77.67,12.93)).size().getInfo())"
```

## If it fails
| Error | Fix |
|---|---|
| `Caller does not have required permission to use project prayashack` | Cloud console -> **IAM & Admin -> IAM** -> edit your account -> add the role **Service Usage Consumer** |
| `Please authorize access` / expired credentials | `.\.venv\Scripts\earthengine.exe authenticate`, sign in in the browser, then `.\.venv\Scripts\earthengine.exe set_project prayashack` |
| Still "API has not been used" after 5 minutes | Repeat step 1 and make sure the top bar of the console shows project `prayashack`; then step 2 |
| "Project is not registered for Earth Engine" | Redo step 2 |

## After it works
- `python -m scripts.tune_mndwi --start 2024-01-01 --end 2024-06-30` picks per-site water thresholds (writes `config/mndwi_thresholds.yaml`).
- `python -m scripts.run_acquisition ...` exports Sentinel-2 rasters for the configured sites (large sites are tiled automatically).
- The Earth Engine code paths in `src/acquisition/gee.py` have unit tests but have never been run against Earth Engine.
