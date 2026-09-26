# Third-party software and data

This inventory separates source code, pretrained weights, and datasets. A
license for source code must not be assumed to cover weights or training data.

| Item | Source | License | Use in FM2Edge | Notes |
|---|---|---|---|---|
| PyTorch | https://github.com/pytorch/pytorch | BSD-style | Training/runtime | Preserve notices when redistributing. |
| Pillow | https://github.com/python-pillow/Pillow | HPND | Image I/O | Runtime dependency. |
| PyYAML | https://github.com/yaml/pyyaml | MIT | Configuration | Runtime dependency. |
| NumPy | https://github.com/numpy/numpy | BSD-3-Clause | Metrics/data conversion | Runtime dependency. |
| PIDNet source | https://github.com/XuJiacong/PIDNet | MIT | PIDNet-S architecture | `models/pidnet.py` is an adapted implementation. Keep attribution. |
| PIDNet weights | Official repository download links | Verify per artifact | Optional initialization | Not bundled. Record URL and SHA-256 before use. Do not use Cityscapes segmentation weights in city-held-out experiments. |
| PaddleSeg PP-LiteSeg / STDC1 source | https://github.com/PaddlePaddle/PaddleSeg | Apache-2.0 | PyTorch-adapted PP-LiteSeg-STDC1 architecture | License copy is retained. The official Paddle weight format is not loaded or bundled. |
| Cityscapes data | https://www.cityscapes-dataset.com/ | Cityscapes terms; non-commercial | Home PoC only | Data is not bundled or redistributed. Do not carry dataset-derived artifacts into a commercial product without review. |
| Oxford-IIIT Pet data | https://www.robots.ox.ac.uk/~vgg/data/pets/ | CC BY-SA 4.0; image copyrights remain with original owners | Disposable public PoC | Data is downloaded from the official host and is not bundled. Preserve attribution and review ShareAlike obligations before redistributing derivatives. |

## Planned, not yet included in Phase 1

| Item | Source | License status to verify before integration |
|---|---|---|
| Torchvision LR-ASPP | https://github.com/pytorch/vision | BSD-3-Clause source; weights/training-data provenance recorded separately |
| DINOv2 | https://github.com/facebookresearch/dinov2 | Check the exact model artifact; repository contains separately licensed additions |
| DINOv3 | https://github.com/facebookresearch/dinov3 | Custom DINOv3 license and access agreement; company legal review required |

This document is an engineering inventory, not legal advice.
