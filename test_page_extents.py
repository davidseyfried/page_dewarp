import unittest

import cv2
import numpy as np

from dewarp import PAGE_MARGIN_X, PAGE_MARGIN_Y, get_page_extents

MAX_PAGE_MASK_RATIO = 0.85
MIN_PAGE_MASK_RATIO = 0.15


class GetPageExtentsTests(unittest.TestCase):

    def test_detects_document_contour(self):
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        contour = np.array([[55, 25], [255, 35], [245, 175], [45, 165]], dtype=np.int32)
        cv2.fillConvexPoly(image, contour, (255, 255, 255))

        page_mask, outline = get_page_extents(image)

        self.assertEqual(page_mask.shape, image.shape[:2])
        self.assertEqual(outline.shape, (4, 2))
        self.assertLess(np.count_nonzero(page_mask), int(MAX_PAGE_MASK_RATIO * image.shape[0] * image.shape[1]))
        self.assertGreater(np.count_nonzero(page_mask), int(MIN_PAGE_MASK_RATIO * image.shape[0] * image.shape[1]))

    def test_falls_back_to_margin_rectangle_when_detection_fails(self):
        image = np.zeros((120, 180, 3), dtype=np.uint8)
        page_mask, outline = get_page_extents(image)

        expected_outline = np.array([
            [PAGE_MARGIN_X, PAGE_MARGIN_Y],
            [PAGE_MARGIN_X, image.shape[0] - PAGE_MARGIN_Y],
            [image.shape[1] - PAGE_MARGIN_X, image.shape[0] - PAGE_MARGIN_Y],
            [image.shape[1] - PAGE_MARGIN_X, PAGE_MARGIN_Y],
        ], dtype=np.int32)

        self.assertTrue(np.array_equal(outline, expected_outline))
        self.assertEqual(page_mask[PAGE_MARGIN_Y, PAGE_MARGIN_X], 255)


if __name__ == '__main__':
    unittest.main()
