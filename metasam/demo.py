import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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
        image = Image.open(image_path)
        self.image = np.array(image.convert("RGB"))
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
    def show_mask(mask, ax, random_color=False, borders=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=375):
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

    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def show_masks(self, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(self.image)
            self.show_mask(mask, plt.gca(), borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                self.show_points(point_coords, input_labels, plt.gca())
            if box_coords is not None:
                self.show_box(box_coords, plt.gca())
            if len(scores) > 1:
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()


# Usage example:
if __name__ == "__main__":
    sam2_wrapper = SAM2Wrapper("checkpoints/sam2_hiera_large.pt", "sam2_hiera_l.yaml")
    sam2_wrapper.set_image('test.png')

    input_point = np.array([[500, 640]])
    input_label = np.array([1])

    masks, scores, logits = sam2_wrapper.predict(input_point, input_label)
    sam2_wrapper.show_masks(masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
