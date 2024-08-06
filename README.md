# MetaSAM 🎭

## Segment Anything Model (SAM) Inference Made Easy! 🚀

### What is MetaSAM? 🤔

MetaSAM is a Python package that simplifies the process of running inference with the Segment Anything Model (SAM). It's designed to make image segmentation tasks a breeze! 🌟

### Features 🌈

- 🖼️ Easy image loading and preprocessing
- 🧠 Simplified SAM model inference
- 🎨 Visualize segmentation results
- 🔧 Customizable segmentation parameters
- 🚀 CUDA-accelerated for lightning-fast performance

### Installation 📦

```bash
pip install metasam
```

### Quick Start 🏃‍♂️

```python
from metasam import SAM2Wrapper

# Initialize SAM2Wrapper
sam = SAM2Wrapper("path/to/checkpoint", "path/to/config")

# Load an image
sam.set_image("path/to/your/image.jpg")

# Predict segmentation
masks, scores, logits = sam.predict(point_coords=[[500, 640]], point_labels=[1])

# Visualize results
sam.show_masks(masks, scores)
```
