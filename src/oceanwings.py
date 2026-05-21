#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OceanWings - Voronoi nuclei detection and cell-area estimation for Drosophila wing discs.
"""

import glob
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import h5py
import matplotlib
import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, cKDTree
from skimage import exposure, filters, measure, morphology
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.measure import profile_line
from skimage.morphology import convex_hull_image, disk
from skimage.segmentation import watershed

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import hdbscan

    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


DEFAULT_CONFIG = {
    "h5_dataset_name": "exported_data",
    "nuclei_class_index": 1,
    "pixel_size_um": 0.3119629,
    "prob_low_threshold": 0.6,
    "use_dapi_for_disc_mask": True,
    "dapi_adaptive": True,
    "dapi_fixed_threshold": 500,
    "dapi_gaussian_sigma_for_mask": 3.0,
    "trachea_opening_radius": 3,
    "raw_peak_min_distance": 2,
    "core_peak_min_distance": 4,
    "gaussian_sigma": 1.0,
    "core_peak_prob_threshold": 0.5,
    "use_cellarea_gap_guided_watershed": True,
    "hdbscan_min_cluster_size": 25,
    "hdbscan_min_samples": 10,
    "hdbscan_nn_cv_max": 0.35,
    "large_output": False,
    "save_voronoi_labels": False,
    "compute_ap_intensity_profile": False,
    "compute_dv_intensity_profile": False,
    "random_seed": 42,
}

np.random.seed(DEFAULT_CONFIG["random_seed"])


def find_disc_folders(root, log_fn=print):
    disc_folders = []
    dir_count = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        dir_count += 1
        tifs = [f for f in filenames if f.lower().endswith((".tif", ".tiff"))]
        if tifs:
            sample = ", ".join(tifs[:3])
            log_fn(f"DEBUG: found {len(tifs)} TIFF(s) in {dirpath}: {sample}")
            disc_folders.append(dirpath)
    disc_folders = sorted(set(disc_folders))
    log_fn(f"DEBUG: scanned {dir_count} directories, found {len(disc_folders)} disc folder(s).")
    return disc_folders


def _parse_z_from_filename(path):
    match = re.search(r"slice_(\d+)", os.path.basename(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _read_first_plane(path, channel_name):
    img = tiff.imread(path)
    if img.ndim == 3:
        if img.shape[-1] <= 4:
            img2d = img[..., 0]
        else:
            img2d = img[0]
    elif img.ndim == 2:
        img2d = img
    else:
        raise ValueError(f"Unexpected {channel_name} image shape {img.shape} in {path}")
    return img2d.astype(np.float32)


def _sort_paths_by_z(paths):
    paths_with_z = [(_parse_z_from_filename(path), path) for path in paths]
    if all(z is not None for z, _path in paths_with_z):
        paths_with_z.sort(key=lambda item: item[0])
    else:
        paths_with_z.sort(key=lambda item: os.path.basename(item[1]))
    return paths_with_z


def load_dapi_and_stain_stacks(folder):
    dapi_paths = glob.glob(os.path.join(folder, "*Dapi.tif")) + glob.glob(
        os.path.join(folder, "*Dapi.tiff")
    )
    if not dapi_paths:
        raise FileNotFoundError(f"No DAPI tiffs (*Dapi.tif/.tiff) found in {folder}")

    dapi_with_z = _sort_paths_by_z(dapi_paths)
    dapi_paths_sorted = [path for _z, path in dapi_with_z]
    dapi_stack = np.stack([_read_first_plane(path, "DAPI") for path in dapi_paths_sorted], axis=0)

    stain_paths = glob.glob(os.path.join(folder, "*Staining.tif")) + glob.glob(
        os.path.join(folder, "*Staining.tiff")
    )
    stain_stack = None
    stain_ok = False

    if stain_paths:
        stain_with_z = _sort_paths_by_z(stain_paths)
        z_to_stain = {z: path for z, path in stain_with_z}
        stain_slices = []
        consistent = True
        for z, _dapi_path in dapi_with_z:
            if z is None or z not in z_to_stain:
                consistent = False
                break
            stain_slices.append(_read_first_plane(z_to_stain[z], "Staining"))
        if consistent and stain_slices:
            stain_stack = np.stack(stain_slices, axis=0)
            stain_ok = True

    return dapi_stack, stain_stack, dapi_paths_sorted, stain_ok


def load_prob_stack_for_tifs(folder, dapi_tif_paths, config):
    preferred_dataset = config["h5_dataset_name"]
    nuclei_class_index = config["nuclei_class_index"]
    cellarea_channel_index = 2

    h5_paths = sorted(glob.glob(os.path.join(folder, "*.h5")) + glob.glob(os.path.join(folder, "*.hdf5")))
    if not h5_paths:
        raise FileNotFoundError(f"No .h5/.hdf5 files found in {folder}")

    def select_dataset(handle, h5_path):
        if preferred_dataset in handle:
            return handle[preferred_dataset]
        for key in handle.keys():
            obj = handle[key]
            if hasattr(obj, "shape") and hasattr(obj, "ndim") and obj.ndim >= 3:
                return obj
        raise KeyError(f"No suitable dataset found in {h5_path}. Tried '{preferred_dataset}', keys: {list(handle.keys())}")

    def load_one_h5(h5_path):
        with h5py.File(h5_path, "r") as handle:
            arr = np.asarray(select_dataset(handle, h5_path)[()])

        def channels_first(a):
            if a.shape[0] < 3:
                raise ValueError(f"Expected at least 3 channels in {h5_path}, got shape {a.shape}")
            return a[nuclei_class_index, ...], a[cellarea_channel_index, ...]

        def channels_last(a):
            if a.shape[-1] < 3:
                raise ValueError(f"Expected at least 3 channels in {h5_path}, got shape {a.shape}")
            return a[..., nuclei_class_index], a[..., cellarea_channel_index]

        if arr.ndim == 3:
            if arr.shape[0] <= 4 and arr.shape[0] <= arr.shape[-1]:
                nuclei_prob, cellarea_prob = channels_first(arr)
            elif arr.shape[-1] <= 4 and arr.shape[-1] <= arr.shape[0]:
                nuclei_prob, cellarea_prob = channels_last(arr)
            else:
                raise ValueError(f"3D Ilastik array shape not clearly channels-first or -last in {h5_path}: {arr.shape}")
        elif arr.ndim == 4:
            if arr.shape[0] != 1:
                raise ValueError(f"Expected first dim==1 in 4D array {h5_path}, got {arr.shape}")
            arr3 = arr[0]
            if arr3.shape[0] <= 4 and arr3.shape[0] <= arr3.shape[-1]:
                nuclei_prob, cellarea_prob = channels_first(arr3)
            elif arr3.shape[-1] <= 4 and arr3.shape[-1] <= arr3.shape[0]:
                nuclei_prob, cellarea_prob = channels_last(arr3)
            else:
                raise ValueError(f"4D Ilastik array not clearly channels-first/last in {h5_path}: {arr3.shape}")
        else:
            raise ValueError(f"Unexpected Ilastik array shape in {h5_path}: {arr.shape}")

        return nuclei_prob.astype(np.float32), cellarea_prob.astype(np.float32)

    nuclei_slices = []
    cellarea_slices = []
    if len(h5_paths) == len(dapi_tif_paths):
        for h5_path in h5_paths:
            nuc, cell = load_one_h5(h5_path)
            nuclei_slices.append(nuc)
            cellarea_slices.append(cell)
    else:
        for tif_path in dapi_tif_paths:
            base = os.path.splitext(os.path.basename(tif_path))[0]
            candidates = [os.path.join(folder, base + ".h5"), os.path.join(folder, base + ".hdf5")]
            if base.lower().endswith("_dapi"):
                base2 = base.rsplit("_", 1)[0]
                candidates.extend([os.path.join(folder, base2 + ".h5"), os.path.join(folder, base2 + ".hdf5")])
            h5_path = next((candidate for candidate in candidates if os.path.exists(candidate)), None)
            if h5_path is None:
                tried = ", ".join(os.path.basename(candidate) for candidate in candidates)
                raise FileNotFoundError(f"Missing probability map for {os.path.basename(tif_path)}. Tried: {tried}")
            nuc, cell = load_one_h5(h5_path)
            nuclei_slices.append(nuc)
            cellarea_slices.append(cell)

    return np.stack(nuclei_slices, axis=0), np.stack(cellarea_slices, axis=0)


def make_disc_mask_from_dapi(dapi_slice, config):
    img = exposure.rescale_intensity(dapi_slice.astype(np.float32), in_range="image", out_range=(0, 1))
    sigma_mask = float(config.get("dapi_gaussian_sigma_for_mask", 0.0))
    img_for_thr = gaussian(img, sigma=sigma_mask, preserve_range=True) if sigma_mask > 0 else img
    thresh = filters.threshold_otsu(img_for_thr) if config["dapi_adaptive"] else config["dapi_fixed_threshold"] / 65535.0
    mask = img_for_thr > thresh
    mask = remove_objects_smaller_than(mask, 100)
    mask = remove_holes_smaller_than(mask, 100)
    return morphology.opening(mask, footprint=disk(3))


def remove_objects_smaller_than(mask, min_size):
    # scikit-image 0.26 renamed min_size to max_size and made the cutoff inclusive.
    return morphology.remove_small_objects(mask, max_size=int(min_size) - 1)


def remove_holes_smaller_than(mask, max_hole_area):
    # Preserve the old strictly-smaller-than behavior by subtracting one pixel.
    return morphology.remove_small_holes(mask, max_size=int(max_hole_area) - 1)


def make_combined_disc_mask(dapi_slice, _cellarea_slice, config):
    disc_mask = np.ones_like(dapi_slice, dtype=bool)
    if config["use_dapi_for_disc_mask"]:
        disc_mask &= make_disc_mask_from_dapi(dapi_slice, config)
    disc_mask = remove_objects_smaller_than(disc_mask, 100)
    disc_mask = remove_holes_smaller_than(disc_mask, 100)
    disc_mask = morphology.opening(disc_mask, footprint=disk(3))

    radius = int(config.get("trachea_opening_radius", 0))
    if radius > 0:
        selem = disk(radius)
        disc_mask = morphology.dilation(morphology.erosion(disc_mask, footprint=selem), footprint=selem)
    return disc_mask


def make_watershed_gradient(cellarea_slice, disc_mask, config):
    if not config.get("use_cellarea_gap_guided_watershed", False):
        return np.zeros_like(cellarea_slice, dtype=float)

    gradient = np.asarray(cellarea_slice, dtype=float)
    finite_vals = gradient[np.isfinite(gradient)]
    fill_low = float(np.min(finite_vals)) if finite_vals.size else 0.0
    fill_high = float(np.max(finite_vals)) if finite_vals.size else 1.0
    gradient = np.nan_to_num(gradient, nan=fill_low, posinf=fill_high, neginf=fill_low)
    gradient = gradient.copy()
    gradient[~disc_mask] = fill_high
    return gradient


def analyze_slice_peaks_voronoi(nuc_prob_slice, dapi_slice, cellarea_slice, config):
    prob = nuc_prob_slice.astype(np.float32)
    disc_mask = make_combined_disc_mask(dapi_slice, cellarea_slice, config)
    prob_disc = prob * disc_mask

    raw_peaks = peak_local_max(
        prob_disc,
        min_distance=int(config["raw_peak_min_distance"]),
        threshold_abs=float(config["prob_low_threshold"]),
        exclude_border=False,
    )
    if raw_peaks.size == 0:
        return disc_mask, np.empty((0, 2), dtype=int), np.zeros_like(prob, dtype=int)

    prob_blur = gaussian(prob_disc, sigma=float(config["gaussian_sigma"]), preserve_range=True)
    core_peaks = peak_local_max(
        prob_blur,
        min_distance=int(config["core_peak_min_distance"]),
        threshold_abs=float(config["core_peak_prob_threshold"]),
        exclude_border=False,
    )

    if core_peaks.size == 0:
        final_peaks = raw_peaks
    else:
        tree = cKDTree(core_peaks.astype(float))
        _dist, idx_core = tree.query(raw_peaks.astype(float), k=1)
        final_peak_list = []
        core_to_raw_indices = {}
        for raw_idx, core_idx in enumerate(idx_core):
            core_to_raw_indices.setdefault(int(core_idx), []).append(int(raw_idx))
        for raw_indices in core_to_raw_indices.values():
            coords = raw_peaks[raw_indices]
            vals = prob[coords[:, 0], coords[:, 1]]
            final_peak_list.append(coords[int(np.argmax(vals))])
        for i_core, (r_core, c_core) in enumerate(core_peaks):
            if i_core not in core_to_raw_indices:
                final_peak_list.append(np.array([r_core, c_core]))
        final_peaks = np.unique(np.array(final_peak_list, dtype=int), axis=0)

    if final_peaks.size == 0:
        return disc_mask, np.empty((0, 2), dtype=int), np.zeros_like(prob, dtype=int)

    seeds = np.zeros_like(prob, dtype=int)
    for i, (r, c) in enumerate(final_peaks, start=1):
        seeds[r, c] = i
    gradient = make_watershed_gradient(cellarea_slice, disc_mask, config)
    voronoi_labels = watershed(gradient, markers=seeds, mask=disc_mask)
    return disc_mask, final_peaks, voronoi_labels


def cluster_peaks_hdbscan(peaks_slices, config, log_fn=print):
    cluster_summaries = []
    peak_cluster_labels = {}
    if not HAS_HDBSCAN:
        log_fn("HDBSCAN not installed; skipping clustering/cell-area estimation.")
        return cluster_summaries, peak_cluster_labels

    min_cluster_size = int(config["hdbscan_min_cluster_size"])
    min_samples = int(config["hdbscan_min_samples"])
    nn_cv_max = float(config["hdbscan_nn_cv_max"])

    for slice_data in peaks_slices:
        z = slice_data["z"]
        coords = slice_data["coords"]
        nn_dist = slice_data["nn_dist"]
        varea = slice_data["varea"]
        if coords.shape[0] < min_cluster_size:
            continue

        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        ).fit_predict(coords[:, ::-1].astype(float))

        for cluster_label in np.unique(labels):
            if cluster_label < 0:
                continue
            mask = labels == cluster_label
            n = int(np.sum(mask))
            if n < min_cluster_size:
                continue
            cluster_nn = nn_dist[mask]
            cluster_nn = cluster_nn[~np.isnan(cluster_nn)]
            if cluster_nn.size < 3:
                continue
            mean_d = float(np.mean(cluster_nn))
            sd_d = float(np.std(cluster_nn))
            cv = sd_d / mean_d if mean_d > 0 else np.inf
            if cv > nn_cv_max:
                continue
            median_d = float(np.median(cluster_nn))
            cluster_varea = varea[mask]
            cluster_varea = cluster_varea[~np.isnan(cluster_varea)]
            median_varea = float(np.median(cluster_varea)) if cluster_varea.size else float("nan")
            est_cell_area_hex = (np.sqrt(3.0) / 2.0) * (median_d**2)
            idxs = np.where(mask)[0]
            for idx in idxs:
                peak_cluster_labels[(z, int(idx))] = int(cluster_label)
            cluster_summaries.append(
                (int(z), int(cluster_label), n, mean_d, sd_d, cv, median_d, median_varea, est_cell_area_hex, median_varea)
            )

    return cluster_summaries, peak_cluster_labels


def compute_whole_disc_localisation_metrics(intensities_1d):
    vals = np.asarray(intensities_1d, dtype=float)
    vals = vals[np.isfinite(vals) & (vals >= 0)]
    if vals.size == 0:
        return np.nan, np.nan
    total = np.sum(vals)
    if total <= 0:
        return np.nan, np.nan
    p = vals / total
    entropy = -np.sum(p * np.log(p + 1e-12))
    loc_entropy = float(np.clip(1.0 - entropy / np.log(p.size), 0.0, 1.0)) if p.size > 1 else np.nan
    vals_sorted = np.sort(vals)[::-1]
    k = int(np.searchsorted(np.cumsum(vals_sorted), 0.8 * total, side="left")) + 1
    loc_area80 = float(np.clip(1.0 - max(1, min(k, vals.size)) / float(vals.size), 0.0, 1.0))
    return loc_entropy, loc_area80


def compute_ap_dv_chords_from_props(props, full_shape):
    H, W = full_shape
    coords = props.coords.astype(float)
    if coords.shape[0] < 2:
        return (np.nan, np.nan, None, None, None, None, None, None, None, None)

    try:
        hull_vertices = coords[ConvexHull(coords).vertices]
    except Exception:
        hull_vertices = coords

    if hull_vertices.shape[0] < 2:
        return (np.nan, np.nan, None, None, None, None, None, None, None, None)

    diffs = hull_vertices[:, None, :] - hull_vertices[None, :, :]
    i, j = np.unravel_index(np.argmax(np.sum(diffs**2, axis=2)), (hull_vertices.shape[0], hull_vertices.shape[0]))
    dv_p0 = hull_vertices[i]
    dv_p1 = hull_vertices[j]

    def clamp_round(point):
        r = max(0, min(H - 1, int(np.round(point[0]))))
        c = max(0, min(W - 1, int(np.round(point[1]))))
        return r, c

    dv_r0, dv_c0 = clamp_round(dv_p0)
    dv_r1, dv_c1 = clamp_round(dv_p1)
    DV_axis_px = float(np.hypot(dv_r1 - dv_r0, dv_c1 - dv_c0))
    if DV_axis_px <= 0:
        return (np.nan, np.nan, None, None, None, None, None, None, None, None)

    dv_vec = np.array([dv_r1 - dv_r0, dv_c1 - dv_c0], dtype=float)
    dv_dir = dv_vec / np.linalg.norm(dv_vec)
    perp0 = np.arctan2(dv_dir[0], dv_dir[1]) - np.pi / 2.0
    angles = np.linspace(perp0 - np.deg2rad(10.0), perp0 + np.deg2rad(10.0), num=21)
    center = np.array(props.centroid, dtype=float)
    best_ap_len_global = -1.0
    best_ap_endpoints = None

    for angle in angles:
        ap_dir = np.array([np.sin(angle), np.cos(angle)], dtype=float)
        ap_dir /= np.linalg.norm(ap_dir)
        ap_perp_dir = np.array([-ap_dir[1], ap_dir[0]], dtype=float)
        q = coords - center
        s = q @ ap_dir
        t_idx = np.round(q @ ap_perp_dir).astype(int)
        chords = {}
        for si, ti in zip(s, t_idx):
            if ti in chords:
                smin, smax = chords[ti]
                chords[ti] = (min(smin, si), max(smax, si))
            else:
                chords[ti] = (si, si)
        if not chords:
            continue
        best_ti, (smin, smax) = max(chords.items(), key=lambda item: item[1][1] - item[1][0])
        best_len_local = smax - smin
        if best_len_local > best_ap_len_global:
            best_ap_len_global = best_len_local
            t_best = float(best_ti)
            best_ap_endpoints = (
                center + t_best * ap_perp_dir + smin * ap_dir,
                center + t_best * ap_perp_dir + smax * ap_dir,
            )

    if best_ap_endpoints is None or best_ap_len_global <= 0:
        return (np.nan, DV_axis_px, None, None, None, None, dv_r0, dv_c0, dv_r1, dv_c1)

    ap_r0, ap_c0 = clamp_round(best_ap_endpoints[0])
    ap_r1, ap_c1 = clamp_round(best_ap_endpoints[1])
    AP_axis_px = float(np.hypot(ap_r1 - ap_r0, ap_c1 - ap_c0))
    return (AP_axis_px, DV_axis_px, ap_r0, ap_c0, ap_r1, ap_c1, dv_r0, dv_c0, dv_r1, dv_c1)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def choose_modal_z_from_peaks(peak_counts):
    if not peak_counts:
        return 0
    zs = np.array(list(peak_counts.keys()), dtype=int)
    counts = np.array(list(peak_counts.values()), dtype=int)
    return int(zs[int(np.argmax(counts))])


def _fmt_nan(value, nd=6):
    return "" if np.isnan(value) else f"{value:.{nd}f}"


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(header + "\n")
        for row in rows:
            handle.write(",".join(str(value) for value in row) + "\n")


def _axis_is_available(axis_px, r0, c0, r1, c1):
    return not np.isnan(axis_px) and None not in (r0, c0, r1, c1)


def _write_intensity_profile_csv(folder, axis_name, stain_image, start_rc, end_rc, axis_px, pixel_size_um, disc_name, modal_z):
    profile = profile_line(stain_image, start_rc, end_rc, mode="constant", order=1)
    if profile.size == 0:
        return None

    dist_px = np.linspace(0.0, axis_px, profile.size, dtype=float)
    max_intensity = np.max(profile)
    profile_norm = profile / max_intensity if max_intensity > 0 else np.full_like(profile, np.nan, dtype=float)
    profile_csv_path = os.path.join(folder, f"_{axis_name}IntensityProfile.csv")

    rows = [
        [
            disc_name,
            modal_z,
            i,
            f"{dist_px[i]:.6f}",
            f"{dist_px[i] * pixel_size_um:.6f}",
            f"{profile[i]:.6f}",
            _fmt_nan(profile_norm[i]),
        ]
        for i in range(profile.size)
    ]
    _write_csv(
        profile_csv_path,
        "disc_name,modal_z_index,index,distance_pixels,distance_um,intensity_raw,intensity_norm",
        rows,
    )
    return profile_csv_path


def _save_axis_on_staining(out_dir, disc_name, modal_z, stain_slice, axis_name, color, start_rc, end_rc):
    stain_png_path = os.path.join(out_dir, f"_modal_{axis_name}_on_Staining.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(exposure.rescale_intensity(stain_slice, in_range="image", out_range=(0, 1)), cmap="gray")
    ax.plot([start_rc[1], end_rc[1]], [start_rc[0], end_rc[0]], color=color, linewidth=2, label=f"{axis_name} sampling line")
    ax.set_title(f"{disc_name} - {axis_name} intensity line (modal z={modal_z})")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    fig.savefig(stain_png_path, dpi=200)
    plt.close(fig)
    return stain_png_path


def _save_modal_images(
    out_dir,
    disc_name,
    modal_z,
    modal_dapi,
    modal_mask,
    modal_labels,
    modal_coords,
    hull,
    axes,
    stain_slice=None,
    profile_axes=None,
):
    profile_axes = profile_axes or []
    modal_overlay = label2rgb(
        modal_labels,
        image=exposure.rescale_intensity(modal_dapi, in_range="image", out_range=(0, 1)),
        bg_label=0,
        alpha=0.4,
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(modal_overlay)
    if modal_coords.shape[0] > 0:
        ax.plot(modal_coords[:, 1], modal_coords[:, 0], marker="o", linestyle="", markersize=3, markeredgewidth=0.7, markerfacecolor="none", markeredgecolor="white")
    ax.set_title(f"{disc_name} - modal z={modal_z} | peaks: {modal_coords.shape[0]}")
    ax.axis("off")
    fig.tight_layout()
    modal_png_path = os.path.join(out_dir, "_modal_z_overlay.png")
    fig.savefig(modal_png_path, dpi=200)
    plt.close(fig)

    discmask_png_path = os.path.join(out_dir, "_modal_discmask.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(exposure.rescale_intensity(modal_dapi, in_range="image", out_range=(0, 1)), cmap="gray")
    ax.contour(modal_mask, levels=[0.5], colors="red", linewidths=1.5)
    if hull is not None:
        ax.contour(hull, levels=[0.5], colors="blue", linewidths=1.0)
    ax.set_title(f"{disc_name} - modal disc mask & hull")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(discmask_png_path, dpi=200)
    plt.close(fig)

    AP_axis_px, DV_axis_px, ap_r0, ap_c0, ap_r1, ap_c1, dv_r0, dv_c0, dv_r1, dv_c1 = axes
    axes_dapi_png_path = os.path.join(out_dir, "_modal_axes_on_DAPI.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(exposure.rescale_intensity(modal_dapi, in_range="image", out_range=(0, 1)), cmap="gray")
    if not np.isnan(AP_axis_px) and None not in (ap_r0, ap_c0, ap_r1, ap_c1):
        ax.plot([ap_c0, ap_c1], [ap_r0, ap_r1], color="red", linewidth=2, label="AP (widest chord)")
    if not np.isnan(DV_axis_px) and None not in (dv_r0, dv_c0, dv_r1, dv_c1):
        ax.plot([dv_c0, dv_c1], [dv_r0, dv_r1], color="blue", linewidth=2, label="DV (longest chord)")
    ax.set_title(f"{disc_name} - modal z={modal_z} AP/DV chords")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    fig.savefig(axes_dapi_png_path, dpi=200)
    plt.close(fig)

    stain_png_paths = {}
    if stain_slice is not None:
        for axis_name, color, start_rc, end_rc in profile_axes:
            stain_png_paths[axis_name] = _save_axis_on_staining(
                out_dir,
                disc_name,
                modal_z,
                stain_slice,
                axis_name,
                color,
                start_rc,
                end_rc,
            )

    return modal_png_path, discmask_png_path, axes_dapi_png_path, stain_png_paths


def process_disc_folder(folder, config, log_fn=print):
    disc_name = os.path.basename(folder.rstrip(os.sep))
    log_fn(f"\nProcessing disc folder: {disc_name}")
    out_dir = os.path.join(folder, "_nuclei_output")
    ensure_dir(out_dir)

    pixel_size_um = float(config["pixel_size_um"])
    area_factor_um2 = pixel_size_um**2
    compute_ap_profile = bool(config.get("compute_ap_intensity_profile", False))
    compute_dv_profile = bool(config.get("compute_dv_intensity_profile", False))
    compute_any_axis_profile = compute_ap_profile or compute_dv_profile

    dapi_stack, stain_stack, dapi_paths, stain_ok = load_dapi_and_stain_stacks(folder)
    if compute_any_axis_profile and (stain_stack is None or not stain_ok):
        log_fn(f"  WARNING: axis intensity profile requested but cannot align staining stack for disc '{disc_name}'.")
        stain_stack = None

    nuc_prob_stack, cellarea_stack = load_prob_stack_for_tifs(folder, dapi_paths, config)
    Z = min(dapi_stack.shape[0], nuc_prob_stack.shape[0], cellarea_stack.shape[0])
    if stain_stack is not None:
        Z = min(Z, stain_stack.shape[0])
    if not (dapi_stack.shape[0] == nuc_prob_stack.shape[0] == cellarea_stack.shape[0] and (stain_stack is None or stain_stack.shape[0] == dapi_stack.shape[0])):
        log_fn(f"  WARNING: Z mismatch. DAPI={dapi_stack.shape[0]}, Nuc={nuc_prob_stack.shape[0]}, CellArea={cellarea_stack.shape[0]}, Stain={stain_stack.shape[0] if stain_stack is not None else 'None'}. Using first {Z}.")

    dapi_stack = dapi_stack[:Z]
    nuc_prob_stack = nuc_prob_stack[:Z]
    cellarea_stack = cellarea_stack[:Z]
    if stain_stack is not None:
        stain_stack = stain_stack[:Z]
    stain_maxproj = np.max(stain_stack, axis=0) if stain_stack is not None else None

    disc_masks = []
    voronoi_labels_list = []
    peaks_slices = []
    peak_counts = {}

    for z in range(Z):
        dapi_slice = dapi_stack[z]
        nuc_prob_slice = nuc_prob_stack[z]
        cellarea_slice = cellarea_stack[z]
        disc_mask, final_peaks, voronoi_labels = analyze_slice_peaks_voronoi(nuc_prob_slice, dapi_slice, cellarea_slice, config)
        disc_masks.append(disc_mask)
        voronoi_labels_list.append(voronoi_labels)
        num_peaks = final_peaks.shape[0]
        peak_counts[z] = num_peaks

        if num_peaks == 0:
            peaks_slices.append({"z": z, "coords": np.empty((0, 2), dtype=float), "nn_dist": np.array([], dtype=float), "varea": np.array([], dtype=float), "prob": np.array([], dtype=float)})
        else:
            counts = np.bincount(voronoi_labels.ravel(), minlength=voronoi_labels.max() + 1) if voronoi_labels.max() >= 1 else np.array([])
            vareas = counts[1 : num_peaks + 1].astype(float) if counts.size else np.full(num_peaks, np.nan, dtype=float)
            probs_at_peaks = nuc_prob_slice[final_peaks[:, 0], final_peaks[:, 1]]
            if num_peaks > 1:
                dists, _idx = cKDTree(final_peaks.astype(float)).query(final_peaks.astype(float), k=2)
                nn_dist = dists[:, 1]
            else:
                nn_dist = np.array([np.nan], dtype=float)
            peaks_slices.append({"z": z, "coords": final_peaks.astype(float), "nn_dist": nn_dist.astype(float), "varea": vareas, "prob": probs_at_peaks.astype(float)})

        if config["large_output"]:
            rng = np.random.default_rng(42)
            random_colors = rng.random((int(voronoi_labels.max()) + 1, 3))
            random_colors[0] = [0, 0, 0]
            overlay = label2rgb(voronoi_labels, image=exposure.rescale_intensity(dapi_slice, in_range="image", out_range=(0, 1)), bg_label=0, alpha=0.4, colors=random_colors)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(overlay)
            if num_peaks > 0:
                ax.plot(final_peaks[:, 1], final_peaks[:, 0], marker="o", linestyle="", markersize=3, markeredgewidth=0.7, markerfacecolor="none", markeredgecolor="white")
            ax.set_title(f"{disc_name} - z={z} | peaks: {num_peaks}")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"z{z:03d}_overlay.png"), dpi=200)
            plt.close(fig)

    disc_mask_stack = np.stack(disc_masks, axis=0)
    voronoi_labels_stack = np.stack(voronoi_labels_list, axis=0)
    if config["save_voronoi_labels"]:
        tiff.imwrite(os.path.join(out_dir, "_VoronoiLabels.tif"), voronoi_labels_stack.astype(np.uint16))

    cellareas_rows = []
    for slice_data, disc_mask in zip(peaks_slices, disc_masks):
        z = slice_data["z"]
        coords = slice_data["coords"]
        varea_non_nan = slice_data["varea"][~np.isnan(slice_data["varea"])]
        mean_va = float(np.mean(varea_non_nan)) if coords.shape[0] and varea_non_nan.size else np.nan
        median_va = float(np.median(varea_non_nan)) if coords.shape[0] and varea_non_nan.size else np.nan
        cellareas_rows.append([z, coords.shape[0], int(np.sum(disc_mask)), _fmt_nan(mean_va), _fmt_nan(mean_va * area_factor_um2), _fmt_nan(median_va), _fmt_nan(median_va * area_factor_um2)])
    cellareas_csv_path = os.path.join(folder, "_CellAreas.csv")
    _write_csv(cellareas_csv_path, "z_index,num_nuclei,disc_area_pixels,mean_voronoi_area_pixels,mean_voronoi_area_um2,median_voronoi_area_pixels,median_voronoi_area_um2", cellareas_rows)

    cluster_summaries, peak_cluster_labels = cluster_peaks_hdbscan(peaks_slices, config, log_fn=log_fn)

    peaks_rows = []
    for slice_data in peaks_slices:
        z = slice_data["z"]
        for idx in range(slice_data["coords"].shape[0]):
            r, c = slice_data["coords"][idx]
            d_px = slice_data["nn_dist"][idx] if idx < slice_data["nn_dist"].size else np.nan
            va_px2 = slice_data["varea"][idx] if idx < slice_data["varea"].size else np.nan
            cl_label = peak_cluster_labels.get((z, idx), -1)
            peaks_rows.append([z, idx, int(r), int(c), f"{slice_data['prob'][idx]:.6f}", _fmt_nan(d_px), _fmt_nan(d_px * pixel_size_um), _fmt_nan(va_px2), _fmt_nan(va_px2 * area_factor_um2), cl_label, 1 if cl_label >= 0 else 0])
    peaks_csv_path = os.path.join(folder, "_Peaks.csv")
    _write_csv(peaks_csv_path, "z_index,idx_in_slice,y,x,nuclei_prob,nn_distance_pixels,nn_distance_um,voronoi_area_pixels,voronoi_area_um2,cluster_label,high_conf", peaks_rows)

    clusters_rows = []
    for z, cl, n, mean_d, sd_d, cv, median_d, median_varea, est_area_hex_px2, est_area_vor_px2 in cluster_summaries:
        clusters_rows.append([z, cl, n, _fmt_nan(mean_d), _fmt_nan(mean_d * pixel_size_um), _fmt_nan(sd_d), _fmt_nan(cv), _fmt_nan(median_d), _fmt_nan(median_d * pixel_size_um), _fmt_nan(median_varea), _fmt_nan(est_area_hex_px2), _fmt_nan(est_area_hex_px2 * area_factor_um2), _fmt_nan(est_area_vor_px2), _fmt_nan(est_area_vor_px2 * area_factor_um2)])
    clusters_csv_path = os.path.join(folder, "_Clusters.csv")
    _write_csv(clusters_csv_path, "z_index,cluster_label,num_nuclei,mean_nn,mean_nn_um,sd_nn,cv_nn,median_nn,median_nn_um,median_voronoi_area_pixels,est_cell_area_hex_pixels2,est_cell_area_hex_um2,est_cell_area_voronoi_pixels2,est_cell_area_voronoi_um2", clusters_rows)

    modal_z = choose_modal_z_from_peaks(peak_counts)
    modal_mask = disc_mask_stack[modal_z]
    mask_filled = ndi.binary_fill_holes(modal_mask)
    labeled = measure.label(mask_filled, connectivity=1)
    props = measure.regionprops(labeled)

    wing_area_px2 = wing_perimeter_px = cy = cx = np.nan
    axes = (np.nan, np.nan, None, None, None, None, None, None, None, None)
    hull = None
    if props:
        largest = max(props, key=lambda prop: prop.area)
        hull = convex_hull_image(labeled == largest.label)
        hull_props = measure.regionprops(measure.label(hull, connectivity=1))
        if hull_props:
            hull_region = hull_props[0]
            wing_area_px2 = float(hull_region.area)
            wing_perimeter_px = float(hull_region.perimeter)
            cy, cx = hull_region.centroid
            axes = compute_ap_dv_chords_from_props(hull_region, modal_mask.shape)
        else:
            wing_area_px2 = float(largest.area)
            wing_perimeter_px = float(largest.perimeter)
            cy, cx = largest.centroid

    AP_axis_px, DV_axis_px, ap_r0, ap_c0, ap_r1, ap_c1, dv_r0, dv_c0, dv_r1, dv_c1 = axes
    stain_mean_raw = stain_median_raw = stain_integrated_raw = stain_mean_bgsub = stain_integrated_bgsub = loc_entropy = loc_area80 = np.nan
    if stain_maxproj is not None and hull is not None:
        vals_disc = stain_maxproj[hull].astype(float)
        if vals_disc.size > 0:
            stain_mean_raw = float(np.mean(vals_disc))
            stain_median_raw = float(np.median(vals_disc))
            stain_integrated_raw = float(np.sum(vals_disc))
            vals_bgsub = vals_disc - np.percentile(vals_disc, 10)
            vals_bgsub[vals_bgsub < 0] = 0
            stain_mean_bgsub = float(np.mean(vals_bgsub))
            stain_integrated_bgsub = float(np.sum(vals_bgsub))
            loc_entropy, loc_area80 = compute_whole_disc_localisation_metrics(vals_bgsub)

    shape_csv_path = os.path.join(folder, "_WingDiscShape.csv")
    aspect_ratio = DV_axis_px / AP_axis_px if (not np.isnan(DV_axis_px) and AP_axis_px > 0) else np.nan
    shape_row = [
        disc_name, modal_z, _fmt_nan(wing_area_px2, 3), _fmt_nan(wing_area_px2 * area_factor_um2, 3),
        _fmt_nan(wing_perimeter_px, 3), _fmt_nan(wing_perimeter_px * pixel_size_um, 3),
        _fmt_nan(AP_axis_px, 3), _fmt_nan(AP_axis_px * pixel_size_um, 3),
        _fmt_nan(DV_axis_px, 3), _fmt_nan(DV_axis_px * pixel_size_um, 3),
        _fmt_nan(aspect_ratio, 4), _fmt_nan(cy, 3), _fmt_nan(cx, 3),
        _fmt_nan(stain_mean_raw, 3), _fmt_nan(stain_median_raw, 3), _fmt_nan(stain_integrated_raw, 3),
        _fmt_nan(stain_mean_bgsub, 3), _fmt_nan(stain_integrated_bgsub, 3),
        _fmt_nan(loc_entropy, 4), _fmt_nan(loc_area80, 4),
    ]
    _write_csv(shape_csv_path, "disc_name,modal_z_index,area_pixels,area_um2,perimeter_pixels,perimeter_um,AP_axis_length_pixels,AP_axis_length_um,DV_axis_length_pixels,DV_axis_length_um,aspect_ratio_DV_over_AP,centroid_y,centroid_x,stain_mean_raw,stain_median_raw,stain_integrated_raw,stain_mean_bgsub,stain_integrated_bgsub,loc_entropy,loc_area80", [shape_row])

    profile_csv_paths = {}
    profile_axes = []
    if compute_ap_profile and stain_maxproj is not None and _axis_is_available(AP_axis_px, ap_r0, ap_c0, ap_r1, ap_c1):
        profile_csv_paths["AP"] = _write_intensity_profile_csv(
            folder,
            "AP",
            stain_maxproj,
            (ap_r0, ap_c0),
            (ap_r1, ap_c1),
            AP_axis_px,
            pixel_size_um,
            disc_name,
            modal_z,
        )
        profile_axes.append(("AP", "red", (ap_r0, ap_c0), (ap_r1, ap_c1)))
    if compute_dv_profile and stain_maxproj is not None and _axis_is_available(DV_axis_px, dv_r0, dv_c0, dv_r1, dv_c1):
        profile_csv_paths["DV"] = _write_intensity_profile_csv(
            folder,
            "DV",
            stain_maxproj,
            (dv_r0, dv_c0),
            (dv_r1, dv_c1),
            DV_axis_px,
            pixel_size_um,
            disc_name,
            modal_z,
        )
        profile_axes.append(("DV", "blue", (dv_r0, dv_c0), (dv_r1, dv_c1)))

    modal_slice_data = [slice_data for slice_data in peaks_slices if slice_data["z"] == modal_z][0]
    modal_png_path, discmask_png_path, axes_dapi_png_path, stain_png_paths = _save_modal_images(
        out_dir,
        disc_name,
        modal_z,
        dapi_stack[modal_z],
        modal_mask,
        voronoi_labels_stack[modal_z],
        modal_slice_data["coords"],
        hull,
        axes,
        stain_stack[modal_z] if profile_axes and stain_stack is not None else None,
        profile_axes,
    )

    log_fn(f"  -> CellAreas CSV         : {cellareas_csv_path}")
    log_fn(f"  -> Peaks CSV             : {peaks_csv_path}")
    log_fn(f"  -> Clusters CSV          : {clusters_csv_path}")
    log_fn(f"  -> WingDiscShape CSV     : {shape_csv_path}")
    for axis_name, profile_csv_path in profile_csv_paths.items():
        if profile_csv_path is not None:
            log_fn(f"  -> {axis_name}IntensityProfile CSV: {profile_csv_path}")
    log_fn(f"  -> Modal overlay PNG     : {modal_png_path}")
    log_fn(f"  -> Disc mask PNG         : {discmask_png_path}")
    log_fn(f"  -> Axes on DAPI PNG      : {axes_dapi_png_path}")
    for axis_name, stain_png_path in stain_png_paths.items():
        log_fn(f"  -> {axis_name} on Staining PNG    : {stain_png_path}")
    log_fn(f"  -> Modal z slice         : {modal_z} (max peaks)")


def run_pipeline(root_dir, config, log_fn=print):
    log_fn(f"Using ROOT folder: {root_dir}")
    log_fn(f"Pixel size: {config['pixel_size_um']} um/px")
    log_fn(f"Gap-guided watershed: {config.get('use_cellarea_gap_guided_watershed', False)}")
    log_fn(f"AP intensity profile: {config.get('compute_ap_intensity_profile', False)}")
    log_fn(f"DV intensity profile: {config.get('compute_dv_intensity_profile', False)}")
    disc_folders = find_disc_folders(root_dir, log_fn=log_fn)
    if not disc_folders:
        log_fn("No disc folders found (no directories with .tif/.tiff). Check that you selected the correct folder.")
        return
    log_fn(f"Found {len(disc_folders)} disc folder(s).")
    for folder in disc_folders:
        try:
            process_disc_folder(folder, config, log_fn=log_fn)
        except Exception as exc:
            log_fn(f"ERROR processing {folder}: {exc}")


class OceanWingsGUI:
    def __init__(self, master):
        self.master = master
        master.title("OceanWings - Voronoi nuclei & cell-area estimator")
        master.geometry("880x740")
        self.root_dir_var = tk.StringVar()
        self.prob_low_var = tk.StringVar(value=str(DEFAULT_CONFIG["prob_low_threshold"]))
        self.use_dapi_var = tk.BooleanVar(value=DEFAULT_CONFIG["use_dapi_for_disc_mask"])
        self.dapi_adaptive_var = tk.BooleanVar(value=DEFAULT_CONFIG["dapi_adaptive"])
        self.dapi_fixed_thr_var = tk.StringVar(value=str(DEFAULT_CONFIG["dapi_fixed_threshold"]))
        self.dapi_mask_sigma_var = tk.StringVar(value=str(DEFAULT_CONFIG["dapi_gaussian_sigma_for_mask"]))
        self.trachea_radius_var = tk.StringVar(value=str(DEFAULT_CONFIG["trachea_opening_radius"]))
        self.pixel_size_var = tk.StringVar(value=str(DEFAULT_CONFIG["pixel_size_um"]))
        self.raw_peak_dist_var = tk.StringVar(value=str(DEFAULT_CONFIG["raw_peak_min_distance"]))
        self.core_peak_dist_var = tk.StringVar(value=str(DEFAULT_CONFIG["core_peak_min_distance"]))
        self.gaussian_sigma_var = tk.StringVar(value=str(DEFAULT_CONFIG["gaussian_sigma"]))
        self.core_peak_thr_var = tk.StringVar(value=str(DEFAULT_CONFIG["core_peak_prob_threshold"]))
        self.gap_guided_watershed_var = tk.BooleanVar(value=DEFAULT_CONFIG["use_cellarea_gap_guided_watershed"])
        self.hdb_min_cluster_var = tk.StringVar(value=str(DEFAULT_CONFIG["hdbscan_min_cluster_size"]))
        self.hdb_min_samples_var = tk.StringVar(value=str(DEFAULT_CONFIG["hdbscan_min_samples"]))
        self.hdb_cv_max_var = tk.StringVar(value=str(DEFAULT_CONFIG["hdbscan_nn_cv_max"]))
        self.large_output_var = tk.BooleanVar(value=DEFAULT_CONFIG["large_output"])
        self.save_voronoi_var = tk.BooleanVar(value=DEFAULT_CONFIG["save_voronoi_labels"])
        self.compute_ap_profile_var = tk.BooleanVar(value=DEFAULT_CONFIG["compute_ap_intensity_profile"])
        self.compute_dv_profile_var = tk.BooleanVar(value=DEFAULT_CONFIG["compute_dv_intensity_profile"])
        self._build_widgets()

    def _build_widgets(self):
        frame_root = ttk.LabelFrame(self.master, text="1. Root folder")
        frame_root.pack(fill="x", padx=10, pady=5)
        ttk.Label(frame_root, text="Root folder:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(frame_root, textvariable=self.root_dir_var, width=60).grid(row=0, column=1, padx=5, pady=5, sticky="we")
        ttk.Button(frame_root, text="Browse...", command=self.browse_root).grid(row=0, column=2, padx=5, pady=5)
        frame_root.columnconfigure(1, weight=1)

        frame_prob = ttk.LabelFrame(self.master, text="2. Probability / disc mask & pixel size")
        frame_prob.pack(fill="x", padx=10, pady=5)
        ttk.Label(frame_prob, text="Nuclei prob floor (prob_low_threshold):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(frame_prob, textvariable=self.prob_low_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        ttk.Checkbutton(frame_prob, text="Use DAPI to define disc mask", variable=self.use_dapi_var).grid(row=1, column=0, columnspan=4, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(frame_prob, text="DAPI: adaptive threshold (Otsu)", variable=self.dapi_adaptive_var, command=self._update_dapi_state).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        ttk.Label(frame_prob, text="DAPI fixed threshold (16-bit):").grid(row=2, column=2, sticky="e", padx=5, pady=2)
        self.dapi_fixed_entry = ttk.Entry(frame_prob, textvariable=self.dapi_fixed_thr_var, width=10)
        self.dapi_fixed_entry.grid(row=2, column=3, padx=5, pady=2)
        ttk.Label(frame_prob, text="DAPI mask Gaussian sigma (px):").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(frame_prob, textvariable=self.dapi_mask_sigma_var, width=10).grid(row=3, column=1, padx=5, pady=2)
        ttk.Label(frame_prob, text="Trachea opening radius (px, 0=off):").grid(row=3, column=2, sticky="e", padx=5, pady=2)
        ttk.Entry(frame_prob, textvariable=self.trachea_radius_var, width=10).grid(row=3, column=3, padx=5, pady=2)
        ttk.Label(frame_prob, text="Pixel size (um per pixel):").grid(row=4, column=2, sticky="e", padx=5, pady=2)
        ttk.Entry(frame_prob, textvariable=self.pixel_size_var, width=10).grid(row=4, column=3, padx=5, pady=2)
        self._update_dapi_state()

        frame_peaks = ttk.LabelFrame(self.master, text="3. Peak detection & Gaussian blur")
        frame_peaks.pack(fill="x", padx=10, pady=5)
        ttk.Label(frame_peaks, text="Raw peak min distance (px):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(frame_peaks, textvariable=self.raw_peak_dist_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(frame_peaks, text="Core peak min distance (px):").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        ttk.Entry(frame_peaks, textvariable=self.core_peak_dist_var, width=10).grid(row=0, column=3, padx=5, pady=2)
        ttk.Label(frame_peaks, text="Gaussian sigma (prob map):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(frame_peaks, textvariable=self.gaussian_sigma_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(frame_peaks, text="Core peak prob threshold:").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        ttk.Entry(frame_peaks, textvariable=self.core_peak_thr_var, width=10).grid(row=1, column=3, padx=5, pady=2)
        ttk.Checkbutton(
            frame_peaks,
            text="Use cell-area/gap probability to guide watershed boundaries",
            variable=self.gap_guided_watershed_var,
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=5, pady=2)

        frame_hdb = ttk.LabelFrame(self.master, text="4. HDBSCAN clustering (per slice)")
        frame_hdb.pack(fill="x", padx=10, pady=5)
        ttk.Label(frame_hdb, text="Min cluster size:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(frame_hdb, textvariable=self.hdb_min_cluster_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(frame_hdb, text="Min samples:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        ttk.Entry(frame_hdb, textvariable=self.hdb_min_samples_var, width=10).grid(row=0, column=3, padx=5, pady=2)
        ttk.Label(frame_hdb, text="Max NN CV in cluster:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(frame_hdb, textvariable=self.hdb_cv_max_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        frame_out = ttk.LabelFrame(self.master, text="5. Output options")
        frame_out.pack(fill="x", padx=10, pady=5)
        ttk.Checkbutton(frame_out, text="Generate BIG outputs (Voronoi overlay for EVERY z-slice)", variable=self.large_output_var).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(frame_out, text="Save Voronoi label stacks (TIFF)", variable=self.save_voronoi_var).grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(frame_out, text="Compute AP intensity profile along staining channel (modal z)", variable=self.compute_ap_profile_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(frame_out, text="Compute DV intensity profile along staining channel (modal z)", variable=self.compute_dv_profile_var).grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        frame_run = ttk.Frame(self.master)
        frame_run.pack(fill="x", padx=10, pady=5)
        ttk.Button(frame_run, text="Run OceanWings", command=self.run_pipeline_gui).pack(side="right")
        frame_log = ttk.LabelFrame(self.master, text="Log")
        frame_log.pack(fill="both", expand=True, padx=10, pady=5)
        self.text_log = tk.Text(frame_log, height=10, wrap="word")
        self.text_log.pack(fill="both", expand=True)
        self.text_log.insert("end", "Ready.\n")
        if not HAS_HDBSCAN:
            self.text_log.insert("end", "Note: hdbscan is not installed; clustering & cell-area clustering will be skipped.\n")

    def _update_dapi_state(self):
        self.dapi_fixed_entry.configure(state="disabled" if self.dapi_adaptive_var.get() else "normal")

    def browse_root(self):
        folder = filedialog.askdirectory(title="Select ROOT folder containing disc folders")
        if folder:
            self.root_dir_var.set(folder)

    def log(self, msg):
        self.text_log.insert("end", msg + "\n")
        self.text_log.see("end")
        self.master.update_idletasks()

    def build_config_from_gui(self):
        cfg = dict(DEFAULT_CONFIG)
        try:
            cfg["prob_low_threshold"] = float(self.prob_low_var.get())
            cfg["dapi_fixed_threshold"] = int(float(self.dapi_fixed_thr_var.get()))
            cfg["dapi_gaussian_sigma_for_mask"] = float(self.dapi_mask_sigma_var.get())
            cfg["trachea_opening_radius"] = int(float(self.trachea_radius_var.get()))
            cfg["pixel_size_um"] = float(self.pixel_size_var.get())
            cfg["raw_peak_min_distance"] = int(float(self.raw_peak_dist_var.get()))
            cfg["core_peak_min_distance"] = int(float(self.core_peak_dist_var.get()))
            cfg["gaussian_sigma"] = float(self.gaussian_sigma_var.get())
            cfg["core_peak_prob_threshold"] = float(self.core_peak_thr_var.get())
            cfg["hdbscan_min_cluster_size"] = int(float(self.hdb_min_cluster_var.get()))
            cfg["hdbscan_min_samples"] = int(float(self.hdb_min_samples_var.get()))
            cfg["hdbscan_nn_cv_max"] = float(self.hdb_cv_max_var.get())
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value in GUI: {exc}") from exc

        cfg["use_dapi_for_disc_mask"] = bool(self.use_dapi_var.get())
        cfg["dapi_adaptive"] = bool(self.dapi_adaptive_var.get())
        cfg["use_cellarea_gap_guided_watershed"] = bool(self.gap_guided_watershed_var.get())
        cfg["large_output"] = bool(self.large_output_var.get())
        cfg["save_voronoi_labels"] = bool(self.save_voronoi_var.get())
        cfg["compute_ap_intensity_profile"] = bool(self.compute_ap_profile_var.get())
        cfg["compute_dv_intensity_profile"] = bool(self.compute_dv_profile_var.get())
        return cfg

    def run_pipeline_gui(self):
        root_dir = self.root_dir_var.get().strip()
        if not root_dir:
            messagebox.showerror("Error", "Please select a root folder.")
            return
        if not os.path.isdir(root_dir):
            messagebox.showerror("Error", f"Root folder does not exist:\n{root_dir}")
            return
        try:
            cfg = self.build_config_from_gui()
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            return
        self.log("====================================================")
        self.log("Starting OceanWings pipeline...")
        self.log(f"Root folder: {root_dir}")
        self.log(f"Pixel size: {cfg['pixel_size_um']} um/px")
        self.log(f"Gap-guided watershed: {cfg['use_cellarea_gap_guided_watershed']}")
        self.log(f"AP intensity profile: {cfg['compute_ap_intensity_profile']}")
        self.log(f"DV intensity profile: {cfg['compute_dv_intensity_profile']}")
        try:
            run_pipeline(root_dir, cfg, log_fn=self.log)
            self.log("Done.")
            messagebox.showinfo("OceanWings", "Processing complete.")
        except Exception as exc:
            self.log(f"FATAL ERROR: {exc}")
            messagebox.showerror("Error", f"An error occurred:\n{exc}")


def main():
    root = tk.Tk()
    OceanWingsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
