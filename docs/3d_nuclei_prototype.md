# Contact-aware 3D nuclear segmentation prototype

**Status: experimental / NOT validated for biological quantification.**

This branch adds an isolated comparator for the newer HPC script's existing
3D EDT/watershed/DAPI-refinement branch. It does **not** replace the legacy
2D path or change morphology, AP/DV, disc-shape or staining outputs.

## Correct Ilastik semantics

For the current classifier:

- Class 0: background.
- Class 1: bright DAPI-rich **nuclear interiors**.
- Class 2: **contacts between nuclei**, including altered signal where
  adjacent nuclei touch/overlap in the image.

Class 2 is **not** a cell membrane or cell-boundary marker. In this prototype
classes 1+2 define candidate nuclear foreground, and class 2 raises the
watershed elevation at putative contacts. A contact ridge cannot create a
nucleus without class-1 evidence.

**Verify the actual class order in the Ilastik project before running.**
The prototype accepts pre-aligned (Z,Y,X) arrays, not H5 files, and does not
infer whether H5 is channels-first/last or which H5 belongs to which DAPI z.
Numeric z IDs and physical voxel calibration must be verified by the caller.

## Example (comparison run, not a production replacement)

```python
import pandas as pd
import tifffile
from nuclei3d import (
    Nuclei3DConfig,
    segment_nuclei_contact_aware_3d,
    projected_voronoi_area_proxy,
)

# Build these aligned ZYX arrays using the existing HPC image loader:
#   dapi_stack
#   probability_interior_stack  <- Ilastik class 1
#   probability_contact_stack   <- Ilastik class 2
#   disc_proper_roi_yx          <- validated measurement ROI for ONE epithelium
# Physical z-step must come from the acquisition, not assumed from pixel size.
spacing = (z_step_um, pixel_size_y_um, pixel_size_x_um)

result = segment_nuclei_contact_aware_3d(
    probability_interior_stack,
    probability_contact_stack,
    spacing,
    dapi_zyx=dapi_stack,
    roi_mask=disc_proper_roi_yx,
    config=Nuclei3DConfig(),
)
tifffile.imwrite("nuclear_instances_3d_experimental.tif", result.labels_zyx)
pd.DataFrame(result.nuclei).to_csv("nuclei_3d_experimental.csv", index=False)
print(result.summary)

# Optional: an XY *area proxy*, not true cell area.
# Do not use nuclear contact probability as a cell boundary image.
voronoi, areas, summary = projected_voronoi_area_proxy(
    result.nuclei,
    disc_proper_roi_yx,
    (pixel_size_y_um, pixel_size_x_um),
    strict_outside=True,
    strict_collisions=False,  # preserve IDs; inspect flagged collisions
)
pd.DataFrame(areas).to_csv("projected_voronoi_area_proxy.csv", index=False)
tifffile.imwrite("projected_voronoi_proxy.tif", voronoi)
print(summary)
```

## What this deliberately does not claim

1. It does not identify cell membranes, true apical surface areas, or a
   unique apical/basal layer from DAPI alone.
2. Two nuclei may share the same XY position at different z. The code
   preserves both 3D IDs and flags rounded-pixel projection collisions.
   Voronoi areas for colliding seeds are **not** trustworthy cell areas.
3. A nucleus touching the top/bottom of the stack has a truncated volume;
   the recorded volume is observed volume, not reconstructed full volume.
4. The ROI area divided by the number of projected nuclei is only
   interpretable if the ROI represents one epithelial sheet, one nucleus
   per relevant cell, and the z-stack captures the corresponding nuclei.
   A folded disc or peripodial epithelium can violate these assumptions.
5. Thresholds, seed prominence and contact weight are exploratory values.
   Neither accuracy nor improvement over the HPC 3D branch is established.
6. The existing morphology-defined plane must not be replaced with the
   slice containing the most nuclear detections.

## Validation before integration

- Annotate representative dense pouch, hinge, depth-limited, mitotic,
  touching and stack-edge 3D ROIs.
- Compare this prototype with the existing HPC 3D detector on exactly
  the same ROIs and physical voxel spacing.
- Report 3D instance precision/recall/F1, split/merge rates, count bias,
  projected centroid error, edge-truncated fraction and area-proxy bias.
- Inspect XY, XZ and YZ label overlays, including contact-probability ridges.
- Keep the 2D legacy outputs unchanged until the 3D path is demonstrably
  better; add new files with distinct names for side-by-side comparison.

## Smoke tests

```bash
python -m unittest discover -s tests -v
```

These are **synthetic invariants**, not validation on real wing discs.
