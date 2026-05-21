# OceanWings

OceanWings is a Python/Tkinter tool for Drosophila wing disc nuclei detection, Voronoi cell-area estimation, wing-shape metrics, and optional AP staining intensity profiles.

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

## Notes

- HDBSCAN is optional at runtime. If it is not installed, clustering and high-confidence cluster labels are skipped.
- Generated CSV, PNG, TIFF, HDF5, and microscopy data outputs are ignored by Git by default so the repository stays lightweight.
