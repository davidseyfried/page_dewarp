"""subsection_utils.py - Helper functions for local warping-based subsection dewarping.

This module provides utilities to:
1. Partition a page into overlapping vertical subsections by y-range.
2. Filter text spans to those within a given y-range.
3. Generate per-subsection coordinate maps using local cubic sheet models.
4. Blend the per-subsection coordinate maps with cosine feathering into a
   seamless composite map that is applied via cv2.remap.
"""

import numpy as np
import cv2  # type: ignore

# Number of vertical subsection rows used for local model fitting.
SUBSECTION_GRID_ROWS = 2

# Fractional overlap between adjacent subsection y-ranges.
# Each subsection extends this fraction of a row-step beyond its nominal boundary
# so that spans near row boundaries are included in both adjacent subsections.
SUBSECTION_OVERLAP = 0.25

# Minimum number of spans required inside a subsection to attempt local
# optimization; subsections with fewer spans fall back to the global model.
MIN_SPANS_PER_SUBSECTION = 2


def get_subsection_yranges(page_height, grid_rows, overlap_frac=SUBSECTION_OVERLAP):
    """Compute overlapping y-ranges in page space for each subsection row.

    Args:
        page_height: total page height in page-space units.
        grid_rows:   number of vertical subsection rows.
        overlap_frac: fraction of a row step to extend each range on each side.

    Returns:
        List of ``(y_start, y_end)`` tuples (in page-space units), one per row.
    """
    step = page_height / grid_rows
    half_overlap = overlap_frac * step / 2.0

    ranges = []
    for row in range(grid_rows):
        y_start = max(0.0, row * step - half_overlap)
        y_end = min(page_height, (row + 1) * step + half_overlap)
        ranges.append((y_start, y_end))

    return ranges


def filter_spans_for_yrange(span_points, ycoords, xcoords, y_start, y_end):
    """Filter spans to those whose y-coordinate falls within ``[y_start, y_end]``.

    Args:
        span_points: list of (N_i, 1, 2) float32 arrays, one per span.
        ycoords:     1-D array of per-span y offsets in page-space units.
        xcoords:     list of 1-D arrays of per-span x offsets.
        y_start:     lower y boundary in page-space units (inclusive).
        y_end:       upper y boundary in page-space units (inclusive).

    Returns:
        ``(indices, filtered_span_points, filtered_ycoords, filtered_xcoords)``
        where *indices* are the original positions of the kept spans.
    """
    indices = [i for i, yc in enumerate(ycoords) if y_start <= yc <= y_end]

    filtered_span_points = [span_points[i] for i in indices]
    filtered_ycoords = (ycoords[np.array(indices, dtype=int)]
                        if indices else np.array([], dtype=np.float64))
    filtered_xcoords = [xcoords[i] for i in indices]

    return indices, filtered_span_points, filtered_ycoords, filtered_xcoords


def make_feather_weights_1d(total_rows, sub_y0_pix, sub_y1_pix):
    """Create a 1-D cosine feathering weight array of length *total_rows*.

    Weights are ``1.0`` at the centre of ``[sub_y0_pix, sub_y1_pix)`` and taper
    smoothly to ``0.0`` at the boundary pixels, following a raised-cosine shape.

    Args:
        total_rows:  length of the output weight array (output image height).
        sub_y0_pix:  first pixel row of the subsection (inclusive).
        sub_y1_pix:  last pixel row of the subsection (exclusive).

    Returns:
        Float32 ndarray of shape ``(total_rows,)``.
    """
    weights = np.zeros(total_rows, dtype=np.float32)
    sub_height = sub_y1_pix - sub_y0_pix

    if sub_height <= 0:
        return weights

    center = (sub_y0_pix + sub_y1_pix) / 2.0
    half = sub_height / 2.0

    ys = np.arange(sub_y0_pix, sub_y1_pix, dtype=np.float32)
    dist = np.abs(ys - center) / half   # 0 at centre → 1 at boundaries
    weights[sub_y0_pix:sub_y1_pix] = 0.5 * (1.0 + np.cos(np.pi * dist))

    return weights


def generate_coord_maps(img_shape, page_dims, params, output_h, output_w,
                        remap_decimate, project_xy_fn, norm2pix_fn):
    """Generate full-resolution source-coordinate maps for a dewarping model.

    Builds a coarse coordinate grid (downscaled by *remap_decimate*), projects
    each page-space point through *project_xy_fn*, converts to pixel space via
    *norm2pix_fn*, then upscales to ``(output_h, output_w)`` with bicubic
    interpolation.

    Args:
        img_shape:      shape of the *source* (original) image, used by norm2pix.
        page_dims:      ``(page_width, page_height)`` in page-space units.
        params:         optimized parameter vector; only the first 8 elements
                        (rvec, tvec, cubic slopes) are used.
        output_h:       height of the output (dewarped) image in pixels.
        output_w:       width of the output (dewarped) image in pixels.
        remap_decimate: downscale factor for the coarse grid.
        project_xy_fn:  callable ``project_xy(xy_coords, pvec) -> image_points``.
        norm2pix_fn:    callable ``norm2pix(shape, pts, as_integer) -> pts``.

    Returns:
        ``(image_x, image_y)`` – two float32 arrays of shape
        ``(output_h, output_w)`` giving the source pixel coordinates for
        ``cv2.remap``.
    """
    height_small = output_h // remap_decimate
    width_small = output_w // remap_decimate

    page_x_range = np.linspace(0, page_dims[0], width_small)
    page_y_range = np.linspace(0, page_dims[1], height_small)

    page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)

    page_xy_coords = np.hstack((
        page_x_coords.flatten().reshape((-1, 1)),
        page_y_coords.flatten().reshape((-1, 1))
    )).astype(np.float32)

    image_points = project_xy_fn(page_xy_coords, params)
    image_points = norm2pix_fn(img_shape, image_points, False)

    ix_small = image_points[:, 0, 0].reshape(height_small, width_small)
    iy_small = image_points[:, 0, 1].reshape(height_small, width_small)

    ix = cv2.resize(ix_small, (output_w, output_h), interpolation=cv2.INTER_CUBIC)
    iy = cv2.resize(iy_small, (output_w, output_h), interpolation=cv2.INTER_CUBIC)

    return ix.astype(np.float32), iy.astype(np.float32)


def build_composite_coord_map(img_shape, page_dims, params_list, y_ranges,
                               output_h, output_w, remap_decimate,
                               project_xy_fn, norm2pix_fn):
    """Build a seamless composite coordinate map by feather-blending local models.

    For each subsection row *i*:

    * A coordinate map is generated from ``params_list[i]``.
    * A 1-D cosine weight array is computed for the subsection's pixel rows.
    * The weighted maps are accumulated and finally normalised by the total
      weight at each row to produce the composite.

    Areas not covered by any subsection (weight sum ≈ 0) are filled from the
    first entry in *params_list*.

    Args:
        img_shape:      shape of the *source* image (used by norm2pix).
        page_dims:      ``(page_width, page_height)`` in page-space units.
        params_list:    list of optimized params vectors, one per subsection row.
        y_ranges:       list of ``(y_start, y_end)`` in page-space units.
        output_h:       output image height in pixels.
        output_w:       output image width in pixels.
        remap_decimate: coarse-grid downscale factor.
        project_xy_fn:  callable – see :func:`generate_coord_maps`.
        norm2pix_fn:    callable – see :func:`generate_coord_maps`.

    Returns:
        ``(composite_x, composite_y)`` – float32 arrays of shape
        ``(output_h, output_w)`` suitable for use with ``cv2.remap``.
    """
    # Convert page y-ranges to pixel rows in the output image.
    pix_ranges = []
    for y_start, y_end in y_ranges:
        y0 = max(0, int(round(y_start / page_dims[1] * output_h)))
        y1 = min(output_h, int(round(y_end / page_dims[1] * output_h)))
        pix_ranges.append((y0, y1))

    weight_sum = np.zeros(output_h, dtype=np.float32)
    composite_x = np.zeros((output_h, output_w), dtype=np.float32)
    composite_y = np.zeros((output_h, output_w), dtype=np.float32)

    for params, (y0_pix, y1_pix) in zip(params_list, pix_ranges):
        ix, iy = generate_coord_maps(img_shape, page_dims, params,
                                     output_h, output_w, remap_decimate,
                                     project_xy_fn, norm2pix_fn)

        weights = make_feather_weights_1d(output_h, y0_pix, y1_pix)
        weight_sum += weights

        w2d = weights.reshape(-1, 1)    # broadcast over output columns
        composite_x += ix * w2d
        composite_y += iy * w2d

    # Normalise by accumulated weight; any uncovered pixels use the first model.
    zero_mask = weight_sum < 1e-8
    if zero_mask.any() and params_list:
        fallback_x, fallback_y = generate_coord_maps(
            img_shape, page_dims, params_list[0],
            output_h, output_w, remap_decimate,
            project_xy_fn, norm2pix_fn)
        composite_x[zero_mask] = fallback_x[zero_mask]
        composite_y[zero_mask] = fallback_y[zero_mask]

    # Prevent division by zero regardless of params_list emptiness.
    weight_sum = np.where(zero_mask, 1.0, weight_sum)

    norm = weight_sum.reshape(-1, 1)
    composite_x /= norm
    composite_y /= norm

    return composite_x, composite_y
