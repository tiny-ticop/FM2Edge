# Third-party software and data

This inventory separates source code, pretrained weights, and datasets. A
license for source code must not be assumed to cover weights or training data.

| Item | Source | License | Use in FM2Edge | Notes |
|---|---|---|---|---|
| PyTorch | https://github.com/pytorch/pytorch | BSD-style | Training/runtime | Preserve notices when redistributing. |
| Pillow | https://github.com/python-pillow/Pillow | HPND | Image I/O | Runtime dependency. |
| PyYAML | https://github.com/yaml/pyyaml | MIT | Configuration | Runtime dependency. |
| NumPy | https://github.com/numpy/numpy | BSD-3-Clause | Metrics/data conversion | Runtime dependency. |
| pandas | https://github.com/pandas-dev/pandas | BSD-3-Clause | Optional factor-analysis tables | Installed through the `analysis` extra. |
| Matplotlib | https://github.com/matplotlib/matplotlib | PSF-based | Optional headless report figures | Installed through the `analysis` extra. |
| scikit-learn | https://github.com/scikit-learn/scikit-learn | BSD-3-Clause | Optional PCA and feature-domain metrics | Installed through the `analysis` extra. |
| PIDNet source | https://github.com/XuJiacong/PIDNet | MIT | PIDNet-S architecture | `models/pidnet.py` is an adapted implementation. Keep attribution. |
| PIDNet weights | Official repository download links | Verify per artifact | Optional initialization | Not bundled. Record URL and SHA-256 before use. Do not use Cityscapes segmentation weights in city-held-out experiments. |
| PaddleSeg PP-LiteSeg / STDC1 source | https://github.com/PaddlePaddle/PaddleSeg | Apache-2.0 | PyTorch-adapted PP-LiteSeg-STDC1 architecture | License copy is retained. The official Paddle weight format is not loaded or bundled. |
| Torchvision MobileNetV3 / LR-ASPP source | https://github.com/pytorch/vision | BSD-3-Clause | Dependency-free PyTorch adaptation | License copy is retained. Torchvision and pretrained weights are not bundled. |
| Cityscapes data | https://www.cityscapes-dataset.com/ | Cityscapes terms; non-commercial | Home PoC only | Data is not bundled or redistributed. Do not carry dataset-derived artifacts into a commercial product without review. |
| Oxford-IIIT Pet data | https://www.robots.ox.ac.uk/~vgg/data/pets/ | CC BY-SA 4.0; image copyrights remain with original owners | Disposable public PoC | Data is downloaded from the official host and is not bundled. Preserve attribution and review ShareAlike obligations before redistributing derivatives. |

## Optional foundation-model integrations

| Item | Source | Use in FM2Edge | License status |
|---|---|---|---|
| DINOv2 | https://github.com/facebookresearch/dinov2 | Frozen backbone loaded from a separately cloned official repository | No code or weights are bundled. Review the exact repository revision and weight artifact before company use; upstream also contains separately licensed additions. |
| DINOv3 | https://github.com/facebookresearch/dinov3 | Frozen backbone after access approval | No code or weights are bundled. DINOv3 uses a custom license and access agreement; company legal review is required. |

This document is an engineering inventory, not legal advice.
