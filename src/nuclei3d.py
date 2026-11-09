"""Experimental contact-aware 3D nuclear instance segmentation.

The Ilastik classes used here are:
    class 0: background
    class 1: nuclear interior
    class 2: nuclear contacts / touching nuclei

Class 2 is NOT a cell-membrane or cell-boundary probability.

This module is intentionally independent of the legacy slice-wise pipeline.
It is a benchmark candidate, not a validated production replacement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


@dataclass(frozen=True)
class Nuclei3DConfig:
    """Initial exploratory parameters; calibrate against annotated 3D ROIs."""

    foreground_low: float = 0.45
    core_high: float = 0.60
    seed_min_core: float = 0.50
    seed_prominence: float = 0.10
    smooth_sigma_um: float = 0.20
    seed_contact_penalty: float = 0.50
    watershed_contact_weight: float = 1.00
    watershed_distance_weight: float = 0.25
    min_foreground_component_um3: float = 3.0
    small_instance_flag_um3: float = 5.0


@dataclass
class Nuclei3DResult:
    labels_zyx: np.ndarray
    nuclei: list[dict[str, Any]]
    summary: dict[str, Any]


def _validate_inputs(
    core: np.ndarray,
    contact: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
    dapi: np.ndarray | None,
    roi_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], np.ndarray | None, np.ndarray]:
    core = np.asarray(core, dtype=np.float32)
    contact = np.asarray(contact, dtype=np.float32)
    if core.ndim != 3 or contact.shape != core.shape:
        raise ValueError("Core and contact probability arrays must have matching (Z,Y,X) shapes.")
    if not np.isfinite(core).all() or not np.isfinite(contact).all():
        raise ValueError("Probability arrays contain NaN or infinity.")
    if (np.min(core) < -1e-4 or np.max(core) > 1.0001
            or np.min(contact) < -1e-4 or np.max(contact) > 1.0001):
        raise ValueError("Ilastik probabilities must be in [0,1].")
    core = np.clip(core, 0, 1)
    contact = np.clip(contact, 0, 1)

    spacing = tuple(float(v) for v in spacing_zyx_um)
    if len(spacing) != 3 or not all(np.isfinite(v) and v > 0 for v in spacing):
        raise ValueError("spacing_zyx_um must be three positive, finite values (dz,dy,dx).")

    if dapi is not None:
        dapi = np.asarray(dapi)
        if dapi.shape != core.shape or not np.isfinite(dapi).all():
            raise ValueError("DAPI must match the (Z,Y,X) shape and be finite.")

    if roi_mask is None:
        roi = np.ones(core.shape, dtype=bool)
    else:
        roi = np.asarray(roi_mask, dtype=bool)
        if roi.shape == core.shape[1:]:
            roi = np.broadcast_to(roi[None, :, :], core.shape)
        elif roi.shape != core.shape:
            raise ValueError("roi_mask must be (Y,X) or (Z,Y,X).")
    return core, contact, spacing, dapi, roi


def _hysteresis_foreground(
    core: np.ndarray,
    contact: np.ndarray,
    roi: np.ndarray,
    spacing: tuple[float, float, float],
    cfg: Nuclei3DConfig,
) -> np.ndarray:
    # Contacts belong to the nuclear foreground; they are not empty gaps.
    candidate = (np.minimum(core + contact, 1.0) >= cfg.foreground_low) & roi
    components, n = ndi.label(candidate, structure=ndi.generate_binary_structure(3, 1))
    if n == 0:
        return np.zeros(core.shape, dtype=bool)
    high_components = np.unique(components[(core >= cfg.core_high) & roi])
    high_components = high_components[high_components != 0]
    min_voxels = max(1, int(np.ceil(
        cfg.min_foreground_component_um3 / np.prod(spacing)
    )))
    counts = np.bincount(components.ravel(), minlength=n + 1)
    keep = np.zeros(n + 1, dtype=bool)
    keep[high_components] = counts[high_components] >= min_voxels
    keep[0] = False
    return keep[components]


def _make_markers(
    foreground: np.ndarray,
    smooth_core: np.ndarray,
    smooth_contact: np.ndarray,
    cfg: Nuclei3DConfig,
) -> tuple[np.ndarray, dict[int, str]]:
    # A contact ridge discourages a seed on the interface, but cannot
    # independently create a second seed.
    seed_score = smooth_core - cfg.seed_contact_penalty * smooth_contact
    maxima = h_maxima(seed_score, cfg.seed_prominence)
    maxima &= foreground & (smooth_core >= cfg.seed_min_core)
    maxima_labels, _ = ndi.label(maxima, structure=ndi.generate_binary_structure(3, 1))
    markers = np.zeros(foreground.shape, dtype=np.int32)
    sources: dict[int, str] = {}
    next_id = 1
    for label_id, slc in enumerate(ndi.find_objects(maxima_labels), start=1):
        if slc is None:
            continue
        local_score = np.where(maxima_labels[slc] == label_id, seed_score[slc], -np.inf)
        pos = np.unravel_index(int(np.argmax(local_score)), local_score.shape)
        global_pos = tuple(int(pos[axis] + slc[axis].start) for axis in range(3))
        markers[global_pos] = next_id
        sources[next_id] = "core_maximum"
        next_id += 1

    # Do not silently lose a valid foreground component if h-maxima finds
    # no seed. Fallback is explicitly recorded for QC.
    components, _ = ndi.label(foreground, structure=ndi.generate_binary_structure(3, 1))
    seeded_components = set(np.unique(components[markers > 0]).tolist())
    for comp_id, slc in enumerate(ndi.find_objects(components), start=1):
        if slc is None or comp_id in seeded_components:
            continue
        local_score = np.where(components[slc] == comp_id, smooth_core[slc], -np.inf)
        pos = np.unravel_index(int(np.argmax(local_score)), local_score.shape)
        global_pos = tuple(int(pos[axis] + slc[axis].start) for axis in range(3))
        markers[global_pos] = next_id
        sources[next_id] = "fallback_component"
        next_id += 1
    return markers, sources


def measure_nuclear_instances(
    labels_zyx: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
    dapi_zyx: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
    small_instance_flag_um3: float = 5.0,
    marker_sources: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """One row per persistent 3D instance; no XY/z deduplication."""
    labels = np.asarray(labels_zyx)
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels_zyx must be a 3D integer label volume.")
    spacing = np.asarray(spacing_zyx_um, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError("Invalid physical voxel spacing.")
    if dapi_zyx is not None and np.shape(dapi_zyx) != labels.shape:
        raise ValueError("DAPI shape does not match label volume.")

    roi_boundary_ids: set[int] = set()
    if roi_mask is not None:
        roi = np.asarray(roi_mask, dtype=bool)
        if roi.shape == labels.shape[1:]:
            roi = np.broadcast_to(roi[None], labels.shape)
        if roi.shape != labels.shape:
            raise ValueError("ROI shape does not match labels.")
        # border_value=1 prevents treating the volume's outside as an ROI
        # cut: physical stack-edge truncation is tracked separately.
        interior = ndi.binary_erosion(
            roi, structure=ndi.generate_binary_structure(3, 1), border_value=1
        )
        roi_boundary_ids = set(np.unique(labels[roi & ~interior]).tolist())
        roi_boundary_ids.discard(0)

    z_size, y_size, x_size = labels.shape
    voxel_volume = float(np.prod(spacing))
    nuclei: list[dict[str, Any]] = []
    for prop in regionprops(labels):
        center_px = np.asarray(prop.centroid, dtype=float)
        center_um = center_px * spacing
        z0, y0, x0, z1, y1, x1 = prop.bbox
        y_pixel, x_pixel = np.rint(center_px[1:]).astype(int)
        row: dict[str, Any] = {
            "nucleus_id": int(prop.label),
            "centroid_z_px": float(center_px[0]),
            "centroid_y_px": float(center_px[1]),
            "centroid_x_px": float(center_px[2]),
            "centroid_z_um": float(center_um[0]),
            "centroid_y_um": float(center_um[1]),
            "centroid_x_um": float(center_um[2]),
            "volume_um3": float(prop.area * voxel_volume),
            "touches_z_stack_edge": bool(z0 == 0 or z1 == z_size),
            "touches_xy_stack_edge": bool(
                y0 == 0 or x0 == 0 or y1 == y_size or x1 == x_size
            ),
            "touches_roi_edge": int(prop.label) in roi_boundary_ids,
            "small_instance_flag": bool(prop.area * voxel_volume < small_instance_flag_um3),
            "seed_source": (marker_sources or {}).get(int(prop.label), "unknown"),
            "projected_pixel_collision": False,
        }
        if roi_mask is not None:
            # For a 3D ROI, XY inclusion must be defined by the caller's
            # measurement plane; do not infer it from a 3D ROI here.
            row["centroid_within_image_xy"] = bool(
                0 <= y_pixel < y_size and 0 <= x_pixel < x_size
            )
        if dapi_zyx is not None:
            coords = prop.coords
            intensities = np.maximum(
                np.asarray(dapi_zyx[tuple(coords.T)], dtype=float), 0
            )
            if intensities.sum() > 0:
                weighted_um = np.average(coords, axis=0, weights=intensities) * spacing
                row["intensity_centroid_shift_um"] = float(
                    np.linalg.norm(weighted_um - center_um)
                )
            else:
                row["intensity_centroid_shift_um"] = float("nan")
        nuclei.append(row)

    # Projected collisions are real possibilities in pseudostratified tissue:
    # FLAG them, never merge 3D instances just because their XY seeds overlap.
    by_pixel: dict[tuple[int, int], list[int]] = {}
    for i, row in enumerate(nuclei):
        key = (
            int(np.rint(row["centroid_y_px"])),
            int(np.rint(row["centroid_x_px"])),
        )
        by_pixel.setdefault(key, []).append(i)
    for indices in by_pixel.values():
        if len(indices) > 1:
            for i in indices:
                nuclei[i]["projected_pixel_collision"] = True
    return nuclei


def segment_nuclei_contact_aware_3d(
    nuclear_interior_probability_zyx: np.ndarray,
    nuclear_contact_probability_zyx: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
    *,
    dapi_zyx: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
    config: Nuclei3DConfig | None = None,
) -> Nuclei3DResult:
    """Segment (Z,Y,X) nuclei, preserving separate IDs across z.

    Experimental baseline. Do not use for biological quantification until
    compared with manually labelled nuclei and the existing HPC 3D branch.
    """
    cfg = config or Nuclei3DConfig()
    if not (0 < cfg.foreground_low <= 1 and 0 < cfg.core_high <= 1
            and 0 < cfg.seed_min_core <= 1 and cfg.seed_prominence > 0
            and cfg.smooth_sigma_um >= 0
            and cfg.min_foreground_component_um3 >= 0
            and cfg.small_instance_flag_um3 >= 0
            and cfg.seed_contact_penalty >= 0
            and cfg.watershed_contact_weight >= 0
            and cfg.watershed_distance_weight >= 0):
        raise ValueError("Invalid Nuclei3DConfig parameter.")
    core, contact, spacing, dapi, roi = _validate_inputs(
        nuclear_interior_probability_zyx,
        nuclear_contact_probability_zyx,
        spacing_zyx_um,
        dapi_zyx,
        roi_mask,
    )
    foreground = _hysteresis_foreground(core, contact, roi, spacing, cfg)
    if not foreground.any():
        return Nuclei3DResult(
            labels_zyx=np.zeros(core.shape, dtype=np.uint32),
            nuclei=[],
            summary={"n_instances": 0, "n_fallback_seeds": 0,
                     "n_projected_pixel_collisions": 0,
                     "foreground_voxels": 0},
        )

    sigma_px = tuple(cfg.smooth_sigma_um / s for s in spacing)
    smooth_core = ndi.gaussian_filter(core, sigma=sigma_px, mode="nearest")
    smooth_contact = ndi.gaussian_filter(contact, sigma=sigma_px, mode="nearest")
    markers, sources = _make_markers(foreground, smooth_core, smooth_contact, cfg)

    # Pad with background before EDT so edge-truncated nuclei have explicit
    # exterior distances. sampling is in physical (dz,dy,dx) micrometres.
    padded = np.pad(foreground, 1, mode="constant", constant_values=False)
    distance_um = ndi.distance_transform_edt(padded, sampling=spacing)[1:-1, 1:-1, 1:-1]
    radius_scale = max(float(np.percentile(distance_um[foreground], 90)), min(spacing))
    elevation = (
        -smooth_core
        + cfg.watershed_contact_weight * smooth_contact
        - cfg.watershed_distance_weight * (distance_um / radius_scale)
    )
    labels = watershed(
        elevation, markers=markers, mask=foreground,
        connectivity=ndi.generate_binary_structure(3, 1),
    ).astype(np.uint32, copy=False)
    nuclei = measure_nuclear_instances(
        labels, spacing, dapi_zyx=dapi,
        roi_mask=roi if roi_mask is not None else None,
        small_instance_flag_um3=cfg.small_instance_flag_um3,
        marker_sources=sources,
    )
    return Nuclei3DResult(
        labels_zyx=labels,
        nuclei=nuclei,
        summary={
            "n_instances": len(nuclei),
            "n_fallback_seeds": sum(row["seed_source"] == "fallback_component" for row in nuclei),
            "n_projected_pixel_collisions": sum(row["projected_pixel_collision"] for row in nuclei),
            "n_z_truncated": sum(row["touches_z_stack_edge"] for row in nuclei),
            "n_small_instances": sum(row["small_instance_flag"] for row in nuclei),
            "foreground_voxels": int(foreground.sum()),
        },
    )


def projected_voronoi_area_proxy(
    nuclei: list[dict[str, Any]],
    measurement_mask_yx: np.ndarray,
    spacing_yx_um: tuple[float, float],
    *,
    strict_collisions: bool = False,
    strict_outside: bool = True,
    chunk_size: int = 200_000,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Area *proxy*, NOT membrane-measured cell area.

    Uses one XY-projected 3D centroid per nucleus and a separately chosen 2D
    tissue/epithelial-layer mask. Nuclear-contact probability is not a cell
    boundary map. Count excludes projected centres outside the mask; by
    default such exclusions raise to prevent a biased area/count ratio.
    """
    mask = np.asarray(measurement_mask_yx, dtype=bool)
    spacing = np.asarray(spacing_yx_um, dtype=float)
    if mask.ndim != 2 or spacing.shape != (2,) or not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError("Expected 2D mask and positive (dy,dx) spacing.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not nuclei:
        raise ValueError("No nuclei supplied for cell-area estimation.")

    in_mask = []
    outside_ids = []
    for row in nuclei:
        y, x = float(row["centroid_y_px"]), float(row["centroid_x_px"])
        iy, ix = int(np.rint(y)), int(np.rint(x))
        if (0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]
                and mask[iy, ix]):
            in_mask.append(row)
        else:
            outside_ids.append(int(row["nucleus_id"]))
    if outside_ids and strict_outside:
        raise ValueError(
            f"{len(outside_ids)} projected nuclei fall outside the 2D measurement ROI. "
            "Choose a consistent epithelial-layer ROI or inspect them before estimating area."
        )
    if not in_mask:
        raise ValueError("No projected centroids lie inside the measurement mask.")

    collisions = [row["nucleus_id"] for row in in_mask if row.get("projected_pixel_collision")]
    if collisions and strict_collisions:
        raise ValueError(
            "Projected nuclei collide in XY. Keep their 3D IDs separate; "
            "a 2D Voronoi territory cannot resolve true axial overlap."
        )

    # Continuous physical XY coordinates are used, not rounded pixel markers.
    # Thus close centres remain distinct where sampling permits; exact
    # projected ties can still give zero-area territories and are QC failures.
    centres_um = np.asarray([
        [row["centroid_y_px"] * spacing[0], row["centroid_x_px"] * spacing[1]]
        for row in in_mask
    ], dtype=float)
    ids = np.asarray([row["nucleus_id"] for row in in_mask], dtype=np.uint32)
    tree = cKDTree(centres_um)
    labels = np.zeros(mask.shape, dtype=np.uint32)
    flat = np.flatnonzero(mask)
    for start in range(0, len(flat), chunk_size):
        positions = flat[start:start + chunk_size]
        yy, xx = np.unravel_index(positions, mask.shape)
        query = np.column_stack((yy * spacing[0], xx * spacing[1]))
        _, nearest = tree.query(query, k=1)
        labels.flat[positions] = ids[nearest]

    counts = np.bincount(labels.ravel(), minlength=int(ids.max()) + 1)
    pixel_area = float(np.prod(spacing))
    rows = [
        {
            "nucleus_id": int(row["nucleus_id"]),
            "voronoi_area_um2": float(counts[int(row["nucleus_id"])] * pixel_area),
            "projected_pixel_collision": bool(row.get("projected_pixel_collision", False)),
            "touches_z_stack_edge": bool(row.get("touches_z_stack_edge", False)),
        }
        for row in in_mask
    ]
    summary = {
        "n_nuclei_inside_roi": len(in_mask),
        "n_nuclei_outside_roi": len(outside_ids),
        "n_projected_collision_nuclei": len(collisions),
        "roi_area_um2": float(mask.sum() * pixel_area),
        "mean_area_proxy_um2": float(mask.sum() * pixel_area / len(in_mask)),
        "median_voronoi_area_proxy_um2": float(
            np.median([row["voronoi_area_um2"] for row in rows])
        ),
        "interpretation": "2D projected nuclear Voronoi area proxy; not a measured cell boundary",
    }
    return labels, rows, summary
