# KD implementations and deliberate adaptations

Japanese meeting slides: [three illustrated KD methods](assets/knowledge_distillation/README.md).

These objectives are implemented locally from published mathematical descriptions.
No external KD repository source code is vendored. Historical model code and licenses
remain unchanged. This is not a claim of reproducing the papers' benchmark numbers.

| Config method | Reference | Implementation |
| --- | --- | --- |
| `mgd` | [Yang et al., ECCV 2022](https://arxiv.org/abs/2205.01529) | 1x1 Student alignment, spatial Bernoulli mask, two 3x3 convolutions, Teacher feature reconstruction. Mean valid-element MSE instead of the paper's sum. Final DINO patch layer, no logits required. |
| `heteroakd` | [Huang et al., AAAI 2025](https://arxiv.org/abs/2504.07691) | DINO-probe adaptation: KMM uses sigmoid one-vs-rest BCE reliability and hybrid logits (Eq. 5-7); KEM uses positive reliability discrepancy and class-softmax weights (Eq. 8-9); weighted soft-target CE (Eq. 10). GT-supervised warmup, auxiliary projection supervision and final-output logit KD. |
| `gkd_cnn_source_only` | [Lv et al., GKD](https://arxiv.org/abs/2603.02554) | CNN adaptation of query/Student-value reconstruction (Eq. 7-9), source-only feature learning followed by frozen-representation task learning. No proxy dataset, CLS or masked-image loss. Optional spatial pooling limits quadratic attention memory. |
| `logit_kd` | [Hinton et al.](https://arxiv.org/abs/1503.02531) | Temperature-scaled Teacher-to-Student KL, valid-pixel mean. Control for the DINO probe adaptation. |
| `none` | Existing FM2Edge segmentation objective | Same CE + Dice and PIDNet auxiliary coefficient as the historical Student trainer. |

## HeteroAKD differences

The frozen, fold-specific DINO segmentation probe replaces the paper's trained
segmentation Teacher/intermediate projection. Student projection is a linear 1x1
classifier (not the paper's BN/ReLU projection), supervised with existing CE+Dice.
Hybrid targets and reliability are detached. Logit KD uses KL(Teacher || Student)
with temperature-squared scaling rather than the direction shown in the paper's
Eq. 1. Eq. 10 uses class softmax and a mean reduction. The class-weighted CE is not
silently replaced by a simple confidence gate. These differences are recorded as
`FM2Edge DINO adaptations` and must be considered when interpreting performance.

## GKD partition

- MobileNet: representation = the unchanged backbone; task = existing LR-ASPP head.
  Distillation sees raw high-level backbone features.
- PIDNet: representation = all existing feature modules including P/I/D coupling,
  SPP and DFM; task = existing `final_layer`, `seghead_p`, `seghead_d`.
  Distillation sees DFM output. This is a feature-extractor/head partition, not a
  claim that PIDNet has the same encoder/decoder architecture as the paper's ViT.
- Representation stage freezes task heads and does not optimize segmentation GT.
  Ignore masks only define which spatial cells are usable.
- Task stage freezes representation parameters **and BatchNorm running statistics**.
  `freeze_representation: false` provides a separate fine-tuning ablation.

## Shared spatial handling

Teacher patch features are expanded in padded image coordinates, cropped to the
original input extent and pooled to a common spatial grid. Student features are
sampled from the same original extent. Fractional valid-area masks exclude ignored
regions. This is a documented alignment approximation, not a native DINO token
reconstruction. DINOv2 uses patch 14/right-bottom padding; DINOv3 uses patch 16.
All KD losses run in float32 even when Student forward uses AMP.

## Scope

No target-machine training, external proxy, image augmentation, DINO fine-tuning,
new Student inference layers or guaranteed domain-generalization gains are implied.
Teacher cache half precision can produce small differences from online float32
features; cache/online comparisons use tolerances, not bitwise equality.
