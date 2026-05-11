## Plan: Local Warping-Based Subsection Dewarping

Enhance the current dewarping pipeline to extract image subsections based on local warping similarity, apply per-subsection dewarping transforms, and merge the results into a seamless output. This approach addresses limitations of a single global model, enabling better handling of local warps, creases, or folds.

**Steps**

### Phase 1: Subsection Segmentation
1. Analyze the image to identify regions with similar local warping (e.g., using tangent angle, keypoint density, or cubic coefficient similarity).
2. Partition the image into overlapping or adaptive grid subsections, with finer granularity in areas of high warping variation.
3. Ensure each subsection contains enough keypoints for stable optimization.

### Phase 2: Per-Subsection Model Fitting
4. For each subsection, extract relevant keypoints (text/line contours).
5. Fit a local cubic sheet model (or reduced-parameter variant) to the keypoints in each subsection using Powell optimization.
6. Optionally, regularize or smooth parameters across subsection boundaries to avoid discontinuities.

### Phase 3: Dewarping and Merging
7. For each subsection, generate a local coordinate map for remapping.
8. Blend coordinate maps in overlapping regions using feathering, Gaussian, or content-aware blending to ensure seamless transitions.
9. Assemble a unified coordinate map for the entire output image.

### Phase 4: Output Generation
10. Apply the composite coordinate map to the original image using `cv2.remap` to produce the dewarped output.
11. Optionally, apply adaptive thresholding per subsection for optimal binarization.

**Relevant files**
- dewarp.py — Main logic for segmentation, optimization, remapping, and blending
- (Potentially new) subsection_utils.py — Helper functions for segmentation, blending, and parameter smoothing

**Verification**
1. Visual inspection: Check for visible seams or artifacts at subsection boundaries.
2. Quantitative: Compare local straightness of text lines before/after dewarping.
3. Regression: Ensure global dewarping still works as a fallback.
4. Test on images with creases, folds, and strong local warping.

**Decisions**
- Use adaptive or fixed grid for subsectioning, depending on warping complexity.
- Blend coordinate maps in overlap regions to avoid seams.
- Regularize parameters to ensure smooth transitions.
- Reuse as much of the current contour/keypoint extraction and optimization code as possible.

**Further Considerations**
1. Subsection size: Recommend starting with 2×2 or 3×3 grid, then adapt based on results.
2. Performance: Parallelize per-subsection optimization and remapping for speed.
3. Advanced: Consider thin-plate spline or TPS for parameter interpolation if needed.
