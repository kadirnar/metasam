import os
import shutil
import tempfile
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from metasam.sam2.build_sam import build_sam2, build_sam2_video_predictor
from metasam.sam2.sam2_image_predictor import SAM2ImagePredictor
from metasam.utils.image_utils import apply_color_mask, draw_bounding_box, draw_points_on_image


class SAM2ImageInference:
    """Class for performing inference on images using SAM2 model."""

    def __init__(self, checkpoint_path: str, model_cfg_path: str, device: str = "cuda"):
        """
        Initialize the SAM2ImageInference class.

        Args:
            checkpoint_path (str): Path to the model checkpoint.
            model_cfg_path (str): Path to the model configuration file.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        # Initialize CUDA settings
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load the SAM2 model
        self.sam2_model = build_sam2(model_cfg_path, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        self.image = None

    def load_image(self, image_path: str) -> None:
        """
        Load an image for inference.

        Args:
            image_path (str): Path to the image file.
        """
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.image)

    def predict_masks(
            self,
            point_coords: np.ndarray,
            point_labels: np.ndarray,
            multimask_output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks based on input points.

        Args:
            point_coords (np.ndarray): Coordinates of input points.
            point_labels (np.ndarray): Labels of input points.
            multimask_output (bool): Whether to output multiple masks.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Masks, scores, and logits.
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )
        sorted_ind = np.argsort(scores)[::-1]
        return masks[sorted_ind], scores[sorted_ind], logits[sorted_ind]

    def visualize_prediction(
            self,
            masks: np.ndarray,
            scores: np.ndarray,
            point_coords: Optional[np.ndarray] = None,
            box_coords: Optional[Tuple[int, int, int, int]] = None,
            input_labels: Optional[np.ndarray] = None,
            draw_borders: bool = True) -> Tuple[Image.Image, np.ndarray, float, np.ndarray]:
        """
        Visualize the prediction results.

        Args:
            masks (np.ndarray): Predicted masks.
            scores (np.ndarray): Scores for each mask.
            point_coords (np.ndarray, optional): Coordinates of input points.
            box_coords (Tuple[int, int, int, int], optional): Coordinates of bounding box.
            input_labels (np.ndarray, optional): Labels of input points.
            draw_borders (bool): Whether to draw borders around the mask.

        Returns:
            Tuple[Image.Image, np.ndarray, float, np.ndarray]:
                PIL Image, best mask, best score, and result as numpy array.
        """
        if len(masks) == 0:
            print("No masks to display.")
            return None

        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]

        result = self.image.copy()
        mask_color = [30, 144, 255]  # RGB color
        result = apply_color_mask(result, best_mask, mask_color)

        if draw_borders:
            contours, _ = cv2.findContours((best_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (255, 255, 255), 2)

        if point_coords is not None and input_labels is not None:
            result = draw_points_on_image(result, point_coords, input_labels)

        if box_coords is not None:
            result = draw_bounding_box(result, box_coords)

        pil_image = Image.fromarray(result.astype(np.uint8))
        return pil_image, masks[best_mask_idx], scores[best_mask_idx], result


class SAM2VideoInference:
    """Class for performing inference on videos using SAM2 model."""

    def __init__(self, checkpoint_path: str, model_cfg_path: str, device: str = "cuda"):
        """
        Initialize the SAM2VideoInference class.

        Args:
            checkpoint_path (str): Path to the model checkpoint.
            model_cfg_path (str): Path to the model configuration file.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
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
        self.frame_names = []

    def load_video(self, video_path: str) -> None:
        """
        Load a video for inference and extract its frames.

        Args:
            video_path (str): Path to the input video file.
        """
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

        # Initialize the predictor with the temporary directory
        self.inference_state = self.predictor.init_state(video_path=self.temp_dir)
        self.predictor.reset_state(self.inference_state)

    def add_points(self, frame_idx: int, obj_id: int, points: np.ndarray,
                   labels: np.ndarray) -> Tuple[List[int], torch.Tensor]:
        """
        Add points for object segmentation.

        Args:
            frame_idx (int): Index of the frame to add points to.
            obj_id (int): ID of the object to segment.
            points (np.ndarray): Array of point coordinates.
            labels (np.ndarray): Array of point labels.

        Returns:
            Tuple[List[int], torch.Tensor]: Object IDs and mask logits.
        """
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        return out_obj_ids, out_mask_logits

    def propagate_masks(self) -> None:
        """Propagate segmentation through the video frames."""
        self.video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    def save_segmented_video(self, output_path: str, alpha: float = 0.5) -> None:
        """
        Save the segmented video with colored masks.

        Args:
            output_path (str): Path to save the output video.
            alpha (float): Transparency of the mask (0-1).
        """
        if not self.video_segments:
            raise RuntimeError("No segmentation data available. Run propagate_masks() first.")

        # Get video properties
        frame_path = os.path.join(self.temp_dir, self.frame_names[0])
        frame = cv2.imread(frame_path)
        if frame is None:
            raise RuntimeError(f"Could not read frame from {frame_path}")

        height, width = frame.shape[:2]
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default to 30 fps if unable to get from video capture

        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise RuntimeError(f"Could not open output video file: {output_path}")

        # Create color map for masks
        cmap = plt.get_cmap("tab10")

        for frame_idx in range(self.total_frames):
            frame_path = os.path.join(self.temp_dir, self.frame_names[frame_idx])
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Warning: Could not read frame {frame_idx} from {frame_path}")
                continue

            if frame_idx in self.video_segments:
                for obj_id, mask in self.video_segments[frame_idx].items():
                    # Remove the extra dimension if present
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        mask = mask.squeeze(0)

                    if mask.shape != frame.shape[:2]:
                        mask = cv2.resize(
                            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

                    # Create colored mask
                    color = np.array(cmap(obj_id % 10)[:3]) * 255
                    frame = apply_color_mask(frame, mask, color, alpha)

            # Write the frame
            out.write(frame)

        # Release the VideoWriter
        out.release()

    def cleanup(self) -> None:
        """Clean up resources used by the model."""
        if self.video_capture:
            self.video_capture.release()
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
