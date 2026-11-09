"""Synthetic smoke tests for the experimental 3D nuclear detector.

Run: python -m unittest discover -s tests -v
These tests verify invariants, not accuracy on real wing discs.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nuclei3d import (
    Nuclei3DConfig,
    measure_nuclear_instances,
    projected_voronoi_area_proxy,
    segment_nuclei_contact_aware_3d,
)


class TestNuclei3D(unittest.TestCase):
    def test_contact_ridge_can_separate_two_connected_nuclei(self):
        z, y, x = np.indices((11, 25, 31))
        a = np.exp(-(((z - 5) / 2.0) ** 2
                     + ((y - 12) / 4.0) ** 2
                     + ((x - 11) / 4.0) ** 2) / 2.0)
        b = np.exp(-(((z - 5) / 2.0) ** 2
                     + ((y - 12) / 4.0) ** 2
                     + ((x - 17) / 4.0) ** 2) / 2.0)
        core = (0.95 * np.maximum(a, b)).astype(np.float32)
        contact = (0.85 * np.exp(-(((z - 5) / 2.0) ** 2
                                   + ((y - 12) / 4.0) ** 2
                                   + ((x - 14) / 1.5) ** 2) / 2.0)).astype(np.float32)
        result = segment_nuclei_contact_aware_3d(
            core, contact, (0.8, 0.3, 0.3),
            config=Nuclei3DConfig(
                foreground_low=0.35,
                core_high=0.55,
                seed_min_core=0.50,
                seed_prominence=0.10,
                min_foreground_component_um3=1.0,
            ),
        )
        self.assertEqual(result.summary["n_instances"], 2)
        self.assertNotEqual(result.labels_zyx[5, 12, 11], result.labels_zyx[5, 12, 17])
        self.assertTrue(all(row["nucleus_id"] > 0 for row in result.nuclei))

    def test_contact_alone_cannot_create_a_nucleus(self):
        core = np.zeros((5, 11, 11), dtype=np.float32)
        contact = np.ones_like(core)
        result = segment_nuclei_contact_aware_3d(core, contact, (1.0, 0.3, 0.3))
        self.assertEqual(result.summary["n_instances"], 0)

    def test_distinct_z_instances_remain_distinct_after_xy_projection(self):
        labels = np.zeros((8, 9, 9), dtype=np.uint32)
        labels[1:3, 4, 4] = 1
        labels[5:7, 4, 4] = 2
        rows = measure_nuclear_instances(labels, (0.8, 0.3, 0.3))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["projected_pixel_collision"] for row in rows))
        self.assertNotEqual(rows[0]["centroid_z_um"], rows[1]["centroid_z_um"])
        with self.assertRaisesRegex(ValueError, "collide"):
            projected_voronoi_area_proxy(
                rows, np.ones((9, 9), dtype=bool), (0.3, 0.3),
                strict_collisions=True,
            )

    def test_edge_flags_and_physical_volume(self):
        labels = np.zeros((6, 10, 10), dtype=np.uint32)
        labels[0:2, 2:4, 2:4] = 1
        row = measure_nuclear_instances(labels, (0.8, 0.3, 0.3))[0]
        self.assertTrue(row["touches_z_stack_edge"])
        self.assertFalse(row["touches_xy_stack_edge"])
        self.assertAlmostEqual(row["volume_um3"], 8 * 0.8 * 0.3 * 0.3)
        self.assertAlmostEqual(row["centroid_z_um"], 0.4)

    def test_area_proxy_is_mask_area_divided_by_unique_nuclei(self):
        rows = [
            {"nucleus_id": 1, "centroid_y_px": 5.0, "centroid_x_px": 2.0},
            {"nucleus_id": 2, "centroid_y_px": 5.0, "centroid_x_px": 7.0},
        ]
        labels, areas, summary = projected_voronoi_area_proxy(
            rows, np.ones((10, 10), dtype=bool), (0.5, 0.5)
        )
        self.assertEqual(summary["n_nuclei_inside_roi"], 2)
        self.assertAlmostEqual(summary["mean_area_proxy_um2"], 12.5)
        self.assertAlmostEqual(sum(r["voronoi_area_um2"] for r in areas), 25.0)
        self.assertEqual(set(np.unique(labels)), {1, 2})

    def test_outside_roi_is_not_silently_excluded(self):
        rows = [
            {"nucleus_id": 1, "centroid_y_px": 5.0, "centroid_x_px": 2.0},
            {"nucleus_id": 2, "centroid_y_px": 5.0, "centroid_x_px": 20.0},
        ]
        with self.assertRaisesRegex(ValueError, "outside"):
            projected_voronoi_area_proxy(
                rows, np.ones((10, 10), dtype=bool), (0.3, 0.3)
            )

    def test_invalid_spacing_is_rejected(self):
        core = np.ones((5, 5, 5), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "spacing"):
            segment_nuclei_contact_aware_3d(core, core, (0.0, 0.3, 0.3))


if __name__ == "__main__":
    unittest.main()
