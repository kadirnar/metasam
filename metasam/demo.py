import cv2
import numpy as np
import torch
from PIL import Image

from metasam.sam2.build_sam import build_sam2
from metasam.sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2Wrapper:

    def __init__(self, checkpoint_path, model_cfg_path, device="cuda"):
        # Initialize CUDA settings
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load the SAM2 model
        self.sam2_model = build_sam2(model_cfg_path, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

    def set_image(self, image_path):
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.image)

    def predict(self, point_coords, point_labels, multimask_output=True):
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )
        sorted_ind = np.argsort(scores)[::-1]
        return masks[sorted_ind], scores[sorted_ind], logits[sorted_ind]

    @staticmethod
    def apply_mask(image, mask, color, alpha=0.5):
        """Apply a colored mask to an image without changing the original image colors."""
        mask = mask.astype(bool)
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = color
        masked_image = np.where(mask[:, :, None], colored_mask, image)
        return cv2.addWeighted(image, 1 - alpha, masked_image, alpha, 0)

    @staticmethod
    def draw_points(image, coords, labels, color_positive=(0, 255, 0), color_negative=(0, 0, 255), radius=5):
        for coord, label in zip(coords, labels):
            color = color_positive if label == 1 else color_negative
            cv2.circle(image, tuple(coord.astype(int)), radius, color, -1)

    @staticmethod
    def draw_box(image, box, color=(0, 255, 0), thickness=2):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    def show_masks(self, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
        if len(masks) == 0:
            print("No masks to display.")
            return None

        # Find the index of the mask with the highest score
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]

        # Create a copy of the original image
        result = self.image.copy()

        # Apply the mask
        mask_color = [30, 144, 255]  # RGB color
        result = self.apply_mask(result, best_mask, mask_color)

        # If borders are requested, draw them
        if borders:
            contours, _ = cv2.findContours((best_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (255, 255, 255), 2)

        # Draw points if provided
        if point_coords is not None and input_labels is not None:
            self.draw_points(result, point_coords, input_labels)

        # Draw box if provided
        if box_coords is not None:
            self.draw_box(result, box_coords)

        array = result.astype(np.uint8)
        pil_image = Image.fromarray(array)
        return pil_image, masks[best_mask_idx], scores[best_mask_idx], result
