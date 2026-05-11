import unittest

import numpy as np
import cv2

from subsection_utils import (
    get_subsection_yranges,
    filter_spans_for_yrange,
    make_feather_weights_1d,
    generate_coord_maps,
    build_composite_coord_map,
    SUBSECTION_GRID_ROWS,
    SUBSECTION_OVERLAP,
    MIN_SPANS_PER_SUBSECTION,
)


class TestGetSubsectionYranges(unittest.TestCase):

    def test_two_rows_cover_full_page(self):
        page_height = 1.0
        ranges = get_subsection_yranges(page_height, grid_rows=2, overlap_frac=0.0)
        self.assertEqual(len(ranges), 2)
        self.assertAlmostEqual(ranges[0][0], 0.0)
        self.assertAlmostEqual(ranges[-1][1], page_height)

    def test_three_rows_cover_full_page(self):
        ranges = get_subsection_yranges(3.0, grid_rows=3, overlap_frac=0.0)
        self.assertEqual(len(ranges), 3)
        self.assertAlmostEqual(ranges[0][0], 0.0)
        self.assertAlmostEqual(ranges[-1][1], 3.0)

    def test_overlap_extends_range(self):
        ranges = get_subsection_yranges(1.0, grid_rows=2, overlap_frac=0.25)
        # First row should start at 0 and extend past 0.5
        self.assertAlmostEqual(ranges[0][0], 0.0)
        self.assertGreater(ranges[0][1], 0.5)
        # Last row should end at 1.0 and start before 0.5
        self.assertLess(ranges[1][0], 0.5)
        self.assertAlmostEqual(ranges[1][1], 1.0)

    def test_overlap_clamps_to_page_bounds(self):
        ranges = get_subsection_yranges(1.0, grid_rows=2, overlap_frac=1.0)
        for y_start, y_end in ranges:
            self.assertGreaterEqual(y_start, 0.0)
            self.assertLessEqual(y_end, 1.0)

    def test_single_row_returns_full_range(self):
        ranges = get_subsection_yranges(2.5, grid_rows=1, overlap_frac=0.0)
        self.assertEqual(len(ranges), 1)
        self.assertAlmostEqual(ranges[0][0], 0.0)
        self.assertAlmostEqual(ranges[0][1], 2.5)

    def test_default_grid_rows_constant_used(self):
        # smoke test: use the module constant
        ranges = get_subsection_yranges(1.0, SUBSECTION_GRID_ROWS, SUBSECTION_OVERLAP)
        self.assertEqual(len(ranges), SUBSECTION_GRID_ROWS)


class TestFilterSpansForYrange(unittest.TestCase):

    def _make_span_point(self, y):
        return np.array([[[0.0, y]]], dtype=np.float32)

    def test_filters_spans_by_ycoord(self):
        span_points = [self._make_span_point(y) for y in [0.1, 0.5, 0.9]]
        ycoords = np.array([0.1, 0.5, 0.9])
        xcoords = [np.array([0.0]), np.array([0.0]), np.array([0.0])]

        indices, sp, yc, xc = filter_spans_for_yrange(
            span_points, ycoords, xcoords, 0.0, 0.6)

        self.assertEqual(indices, [0, 1])
        self.assertEqual(len(sp), 2)
        np.testing.assert_array_equal(yc, np.array([0.1, 0.5]))

    def test_all_in_range(self):
        span_points = [self._make_span_point(y) for y in [0.2, 0.8]]
        ycoords = np.array([0.2, 0.8])
        xcoords = [np.array([0.0]), np.array([0.0])]

        indices, sp, yc, xc = filter_spans_for_yrange(
            span_points, ycoords, xcoords, 0.0, 1.0)

        self.assertEqual(len(indices), 2)

    def test_none_in_range_returns_empty(self):
        span_points = [np.array([[[0.0, 0.9]]], dtype=np.float32)]
        ycoords = np.array([0.9])
        xcoords = [np.array([0.0])]

        indices, sp, yc, xc = filter_spans_for_yrange(
            span_points, ycoords, xcoords, 0.0, 0.5)

        self.assertEqual(indices, [])
        self.assertEqual(len(sp), 0)
        self.assertEqual(len(yc), 0)

    def test_boundary_values_included(self):
        span_points = [np.array([[[0.0, y]]], dtype=np.float32) for y in [0.0, 1.0]]
        ycoords = np.array([0.0, 1.0])
        xcoords = [np.array([0.0]), np.array([0.0])]

        indices, sp, yc, xc = filter_spans_for_yrange(
            span_points, ycoords, xcoords, 0.0, 1.0)

        self.assertEqual(len(indices), 2)


class TestMakeFeatherWeights1d(unittest.TestCase):

    def test_peak_at_centre(self):
        weights = make_feather_weights_1d(100, 20, 80)
        centre = (20 + 80) // 2
        # centre weight should be ~1.0
        self.assertAlmostEqual(float(weights[centre]), 1.0, places=5)

    def test_zero_outside_range(self):
        weights = make_feather_weights_1d(100, 30, 70)
        self.assertEqual(float(weights[0]), 0.0)
        self.assertEqual(float(weights[99]), 0.0)

    def test_nonzero_inside_range(self):
        weights = make_feather_weights_1d(100, 20, 80)
        self.assertGreater(float(weights[50]), 0.0)

    def test_zero_height_returns_all_zeros(self):
        weights = make_feather_weights_1d(100, 50, 50)
        self.assertTrue(np.all(weights == 0.0))

    def test_output_length(self):
        weights = make_feather_weights_1d(200, 10, 190)
        self.assertEqual(len(weights), 200)

    def test_weights_in_valid_range(self):
        weights = make_feather_weights_1d(100, 10, 90)
        self.assertGreaterEqual(float(weights.min()), 0.0)
        self.assertLessEqual(float(weights.max()), 1.0 + 1e-6)


class TestGenerateCoordMaps(unittest.TestCase):

    def _dummy_project_xy(self, xy_coords, pvec):
        # identity-like projection: image coords == page coords (scaled)
        n = xy_coords.shape[0]
        pts = xy_coords[:, :2].reshape((n, 1, 2)).astype(np.float32)
        return pts

    def _dummy_norm2pix(self, shape, pts, as_integer):
        h, w = shape[:2]
        scale = max(h, w) * 0.5
        result = pts * scale + np.array([w * 0.5, h * 0.5]).reshape((-1, 1, 2))
        return result

    def test_output_shape(self):
        img_shape = (400, 300, 3)
        page_dims = (1.0, 1.0)
        # minimal params – only first 8 entries matter
        params = np.zeros(8)
        remap_decimate = 8
        output_h, output_w = 64, 48

        ix, iy = generate_coord_maps(
            img_shape, page_dims, params, output_h, output_w,
            remap_decimate, self._dummy_project_xy, self._dummy_norm2pix)

        self.assertEqual(ix.shape, (output_h, output_w))
        self.assertEqual(iy.shape, (output_h, output_w))
        self.assertEqual(ix.dtype, np.float32)
        self.assertEqual(iy.dtype, np.float32)


class TestBuildCompositeCoordMap(unittest.TestCase):

    def _dummy_project_xy(self, xy_coords, pvec):
        n = xy_coords.shape[0]
        # scale by cubic offset stored in pvec[6] (just a marker value)
        pts = xy_coords[:, :2].copy().reshape((n, 1, 2)).astype(np.float32)
        return pts

    def _dummy_norm2pix(self, shape, pts, as_integer):
        return pts

    def _make_params(self, marker=0.0):
        p = np.zeros(10, dtype=np.float64)
        p[6] = marker
        return p

    def test_output_shape(self):
        page_dims = (1.0, 1.0)
        params_list = [self._make_params(0.0), self._make_params(0.5)]
        y_ranges = [(0.0, 0.6), (0.4, 1.0)]
        output_h, output_w = 32, 24
        img_shape = (32, 24, 3)

        cx, cy = build_composite_coord_map(
            img_shape, page_dims, params_list, y_ranges,
            output_h, output_w, 4,
            self._dummy_project_xy, self._dummy_norm2pix)

        self.assertEqual(cx.shape, (output_h, output_w))
        self.assertEqual(cy.shape, (output_h, output_w))

    def test_single_params_matches_generate_coord_maps(self):
        from subsection_utils import generate_coord_maps

        page_dims = (1.0, 1.0)
        params = self._make_params(0.0)
        y_ranges = [(0.0, 1.0)]
        output_h, output_w = 16, 12
        img_shape = (16, 12, 3)

        cx, cy = build_composite_coord_map(
            img_shape, page_dims, [params], y_ranges,
            output_h, output_w, 4,
            self._dummy_project_xy, self._dummy_norm2pix)

        ix, iy = generate_coord_maps(
            img_shape, page_dims, params,
            output_h, output_w, 4,
            self._dummy_project_xy, self._dummy_norm2pix)

        np.testing.assert_allclose(cx, ix, atol=1e-4)
        np.testing.assert_allclose(cy, iy, atol=1e-4)

    def test_fallback_on_empty_params_list(self):
        # Edge-case: empty params_list should not crash (returns zeros)
        page_dims = (1.0, 1.0)
        output_h, output_w = 8, 8
        img_shape = (8, 8, 3)

        cx, cy = build_composite_coord_map(
            img_shape, page_dims, [], [],
            output_h, output_w, 2,
            self._dummy_project_xy, self._dummy_norm2pix)

        self.assertEqual(cx.shape, (output_h, output_w))

    def test_no_seam_artifacts_in_blend_region(self):
        # Two identical params should produce a smooth blend with no jumps.
        page_dims = (1.0, 1.0)
        params = self._make_params(0.0)
        y_ranges = [(0.0, 0.6), (0.4, 1.0)]
        output_h, output_w = 32, 24
        img_shape = (32, 24, 3)

        cx, cy = build_composite_coord_map(
            img_shape, page_dims, [params, params], y_ranges,
            output_h, output_w, 4,
            self._dummy_project_xy, self._dummy_norm2pix)

        # Consecutive row differences should be small
        row_diff = np.abs(np.diff(cx, axis=0)).max()
        self.assertLess(float(row_diff), 0.1)


if __name__ == '__main__':
    unittest.main()
