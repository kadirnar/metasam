from typing import Tuple

import cv2
import numpy as np


def apply_color_mask(
        image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.5) -> np.ndarray:
    """
    Apply a colored mask to an image.

    Args:
        image (np.ndarray): The original image.
        mask (np.ndarray): The binary mask to apply.
        color (Tuple[int, int, int]): The RGB color of the mask.
        alpha (float): The transparency of the mask (0-1).

    Returns:
        np.ndarray: The image with the colored mask applied.
    """
    mask = mask.astype(bool)
    colored_mask = np.zeros_like(image)
    colored_mask[mask] = color
    masked_image = np.where(mask[:, :, None], colored_mask, image)
    return cv2.addWeighted(image, 1 - alpha, masked_image, alpha, 0)


def draw_points_on_image(
        image: np.ndarray,
        coords: np.ndarray,
        labels: np.ndarray,
        color_positive: Tuple[int, int, int] = (0, 255, 0),
        color_negative: Tuple[int, int, int] = (0, 0, 255),
        radius: int = 5) -> np.ndarray:
    """
    Draw points on an image.

    Args:
        image (np.ndarray): The image to draw on.
        coords (np.ndarray): Array of point coordinates.
        labels (np.ndarray): Array of point labels (1 for positive, 0 for negative).
        color_positive (Tuple[int, int, int]): Color for positive points.
        color_negative (Tuple[int, int, int]): Color for negative points.
        radius (int): Radius of the points.

    Returns:
        np.ndarray: The image with points drawn.
    """
    image_copy = image.copy()
    for coord, label in zip(coords, labels):
        color = color_positive if label == 1 else color_negative
        cv2.circle(image_copy, tuple(coord.astype(int)), radius, color, -1)
    return image_copy


def draw_bounding_box(
        image: np.ndarray,
        box: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2) -> np.ndarray:
    """
    Draw a bounding box on an image.

    Args:
        image (np.ndarray): The image to draw on.
        box (Tuple[int, int, int, int]): The bounding box coordinates (x1, y1, x2, y2).
        color (Tuple[int, int, int]): The color of the bounding box.
        thickness (int): The thickness of the bounding box lines.

    Returns:
        np.ndarray: The image with the bounding box drawn.
    """
    image_copy = image.copy()
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)
    return image_copy
