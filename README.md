<div align="center">
<h2>
    MetaSam: Packaged version of the Segment Anything 2 Model
</h2>
<div>
    <img width="500" alt="teaser" src="doc/assets/logo.png">
</div>
<div>
    <a href="https://pypi.org/project/metasam" target="_blank">
        <img src="https://img.shields.io/pypi/pyversions/metasam.svg?color=%2334D058" alt="Supported Python versions">
    </a>
    <a href="https://badge.fury.io/py/metasam"><img src="https://badge.fury.io/py/metasam.svg" alt="pypi version"></a>
</div>
</div>

## 🛠️ Installation

```bash
pip install metasam
```

## 🤗 Model Hub

```bash
bash script/download_model.sh
```

## ⭐ Usage

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

## 😍 Contributing

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## 📜 License

This project is licensed under the terms of the Apache License 2.0.

## 🤗 Citation

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```
