# OceanWings

OceanWings is a Python/Tkinter tool for Drosophila wing disc nuclei detection, Voronoi cell-area estimation, wing-shape metrics, and optional AP/DV staining intensity profiles.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
python src/oceanwings.py
```

Use the GUI to select the root folder containing disc folders. Each disc folder should contain DAPI TIFF slices and matching Ilastik HDF5 probability maps.

By default, the watershed boundaries are guided by the cell-area/gap probability map, treating high gap probability as watershed ridges between dense nuclei. Disable the checkbox in the peak-detection section to compare against the original flat Voronoi-style watershed.

Optional staining-channel profiles can be exported along the modal-slice AP axis, DV axis, or both.

Whole-disc shape measurements in `_WingDiscShape.csv` use the convex hull around the modal-slice disc mask, so weak internal DAPI regions do not reduce the reported disc area.

## Notes

- HDBSCAN is optional at runtime. If it is not installed, clustering and high-confidence cluster labels are skipped.
- Generated CSV, PNG, TIFF, HDF5, and microscopy data outputs are ignored by Git by default so the repository stays lightweight.
