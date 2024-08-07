import os
import shutil
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from metasam.sam2.build_sam import build_sam2, build_sam2_video_predictor
from metasam.sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2Inference:

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


class VideoSegmentationModel:

    def __init__(self, checkpoint_path, model_cfg_path, device="cuda"):
        # Initialize CUDA settings
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load the SAM2 video predictor
        self.predictor = build_sam2_video_predictor(model_cfg_path, checkpoint_path)
        self.inference_state = None
        self.video_segments = {}
        self.video_capture = None
        self.total_frames = 0
        self.temp_dir = None

    def set_video(self, video_path):
        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(video_path)
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a temporary directory to store frames
        self.temp_dir = tempfile.mkdtemp()

        # Extract frames
        self.frame_names = []
        for i in range(self.total_frames):
            ret, frame = self.video_capture.read()
            if ret:
                frame_name = f"{i:05d}.jpg"
                frame_path = os.path.join(self.temp_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                self.frame_names.append(frame_name)

        if not self.frame_names:
            raise RuntimeError(f"No frames could be extracted from the video: {video_path}")

        print(f"Extracted {len(self.frame_names)} frames to {self.temp_dir}")

        # Initialize the predictor with the temporary directory
        self.inference_state = self.predictor.init_state(video_path=self.temp_dir)
        self.predictor.reset_state(self.inference_state)

    def add_points(self, frame_idx, obj_id, points, labels):
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        return out_obj_ids, out_mask_logits

    def propagate_video(self):
        self.video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    @staticmethod
    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=200):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(
            pos_points[:, 0],
            pos_points[:, 1],
            color='green',
            marker='*',
            s=marker_size,
            edgecolor='white',
            linewidth=1.25)
        ax.scatter(
            neg_points[:, 0],
            neg_points[:, 1],
            color='red',
            marker='*',
            s=marker_size,
            edgecolor='white',
            linewidth=1.25)

    def visualize_frame(self, frame_idx, points=None, labels=None):
        plt.figure(figsize=(12, 8))
        plt.title(f"frame {frame_idx}")
        frame_path = os.path.join(self.temp_dir, self.frame_names[frame_idx])
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame)

        if points is not None and labels is not None:
            self.show_points(points, labels, plt.gca())

        if frame_idx in self.video_segments:
            for obj_id, mask in self.video_segments[frame_idx].items():
                self.show_mask(mask, plt.gca(), obj_id=obj_id)

        plt.show()

    def visualize_video(self, stride=15):
        plt.close("all")
        for out_frame_idx in range(0, self.total_frames, stride):
            self.visualize_frame(out_frame_idx)
