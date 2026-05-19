# Similarity-Aware Adaptive Loss for Fine-Grained Flower Classification

Research-oriented PyTorch implementation for fine-grained flower recognition on **Oxford Flowers-102** using **InceptionV3** and a custom **similarity-aware adaptive loss** designed to penalize structured confusions more effectively than standard Cross Entropy.

## Motivation
Fine-grained visual classification is difficult because many categories are visually adjacent: subtle petal geometry, texture, and color variations separate classes, while viewpoint, lighting, and background clutter increase ambiguity. Standard classification objectives treat all mistakes uniformly, even though some misclassifications reflect meaningful semantic overlap and should receive stronger corrective pressure.

This repository studies loss-function design as the main research contribution rather than only architectural scaling. The goal is to improve discrimination among confusing flower classes by introducing structured, confusion-aware supervision into the optimization objective.

## Problem Statement
Given an input flower image $x$ and a ground-truth label $y \in \{1,\dots,102\}$, learn a classifier that not only maximizes the probability of the true class, but also pays special attention to **historically confusing class pairs**. The project asks the following question:

> Can a confusion-structured adaptive loss improve fine-grained recognition beyond standard Cross Entropy and a generic entropy-aware adaptive objective?

## Research Evolution
This repository reflects the evolution of the research idea:

1. **Cross Entropy baseline** established a strong transfer-learning reference.
2. **Entropy-aware adaptive loss** was explored to emphasize uncertain predictions.
3. A key weakness was identified: **entropy can increase because of noise-induced uncertainty**, not only semantic difficulty.
4. Based on this observation and supervisor feedback, the formulation was refined.
5. The project shifted toward **structured similarity-aware penalization** using confusion-derived class relationships.
6. The final similarity-aware loss was implemented and evaluated against both prior stages.

## Dataset Overview
- **Dataset:** Oxford Flowers-102
- **Task:** Fine-grained flower species classification
- **Classes:** 102
- **Backbone:** InceptionV3 with ImageNet transfer learning
- **Hardware used in experiments:** NVIDIA GeForce RTX 4060 Laptop GPU

Archived project split sizes:

| Split | Images |
|---|---:|
| Train | 1,020 |
| Validation | 1,020 |
| Test | 6,149 |

## Model Architecture
The experiments use **InceptionV3** as the backbone, chosen for its strong multi-scale feature extraction behavior and suitability for high-resolution visual recognition. The classifier head is adapted for 102-way flower classification and trained with transfer learning.

Core pipeline:
- InceptionV3 pretrained on ImageNet
- custom classification head for 102 classes
- data augmentation for train split
- mixed precision on CUDA for adaptive experiments
- checkpointing, logging, and confusion-matrix-based analysis

## Proposed Loss Function
The final contribution of this repository is the following similarity-aware adaptive loss:

$$
\mathcal{L} = -\log(p_t)\left(1 + \lambda \, S(y,\hat{y}) \, (1-p_t) \, \mathbb{I}(\hat{y}\neq y)\right)
$$

where:

- $p_t$ is the predicted probability of the true class
- $\hat{y}$ is the predicted class from $\arg\max$
- $S(y,\hat{y})$ is the confusion-derived similarity score between the true and predicted classes
- $(1-p_t)$ is a difficulty factor
- $\mathbb{I}(\hat{y}\neq y)$ activates the adaptive penalty only for incorrect predictions
- $\lambda$ controls the strength of the structured adaptive term

### Intuition
This loss behaves like standard Cross Entropy on correct predictions, but increases the penalty when:

1. the sample is misclassified,
2. the predicted wrong class is historically confusable with the true class,
3. the model is not confident in the true label.

That makes the loss especially suitable for fine-grained recognition, where visually similar classes dominate the error distribution.

## Mathematical Formulation
Standard multi-class softmax posterior:

$$
p_k = \frac{\exp(z_k)}{\sum_{j=1}^{K}\exp(z_j)}
$$

True-class confidence:

$$
p_t = p_y
$$

Cross Entropy baseline:

$$
\mathcal{L}_{CE} = -\log(p_t)
$$

Difficulty factor:

$$
D = 1 - p_t
$$

Adaptive weight:

$$
W = 1 + \lambda \, S(y,\hat{y}) \, (1-p_t) \, \mathbb{I}(\hat{y}\neq y)
$$

Final similarity-aware objective:

$$
\mathcal{L} = \mathcal{L}_{CE} \cdot W
$$

## Similarity-Aware Learning
The similarity matrix is not handcrafted and is not derived from the proposed loss itself. Instead, it is constructed from the **existing Cross Entropy baseline confusion matrix**:

1. load the CE confusion matrix,
2. zero the diagonal to exclude correct predictions,
3. normalize off-diagonal confusion counts into $[0,1]$,
4. use the result as a structured class-pair similarity prior.

This design ensures that the adaptive penalty is driven by **empirical confusion structure** rather than global uncertainty.

## Why Similarity-Aware Instead of Entropy-Aware?
The entropy-aware formulation improved performance, but entropy penalization has a limitation: uncertainty may arise from

- genuine class similarity,
- poor image quality,
- background clutter,
- atypical samples,
- or annotation noise.

Similarity-aware penalization is more targeted. It does not treat all uncertainty as equally undesirable; instead, it emphasizes those mistakes that correspond to historically difficult class boundaries.

## Experimental Results

### Main Comparison

| Method | Validation Accuracy | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Cross Entropy baseline | Not archived in saved baseline artifact | 88.03% | 87.24% | 87.97% |
| Adaptive entropy-aware loss | 94.12% | 90.40% | 90.17% | 90.36% |
| Similarity-aware adaptive loss | 92.75% | 90.45% | 90.49% | 90.43% |

### Key Takeaways
- The proposed similarity-aware loss improves the CE baseline by **+2.42 test accuracy points**.
- It also improves macro F1 over the CE baseline by **+3.25 points**.
- Compared with the entropy-aware adaptive formulation, it provides a **small but meaningful gain on held-out test performance**, suggesting slightly better generalization.

## Visualizations

### Cross Entropy Confusion Matrix
![Cross Entropy Confusion Matrix](figures/ce_confusion_matrix.png)

### Adaptive Entropy-Aware Training Curves
![Adaptive Training Curves](figures/adaptive_training_curves.png)

### Adaptive Entropy-Aware Confusion Matrix
![Adaptive Confusion Matrix](figures/adaptive_confusion_matrix.png)

### Similarity-Aware Training Curves
![Similarity-Aware Training Curves](figures/similarity_training_curves.png)

### Similarity-Aware Confusion Matrix
![Similarity-Aware Confusion Matrix](figures/similarity_confusion_matrix.png)

### Similarity Matrix Heatmap
![Similarity Matrix Heatmap](figures/similarity_matrix_heatmap.png)

### Grad-CAM Placeholder
Future releases may include Grad-CAM visualizations to show which flower regions are most influential for discrimination under each loss design.

## Installation
Clone the repository and install the dependencies:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
pip install -r requirements.txt
```

## Training

### 1. Cross Entropy Baseline
```bash
python main.py --mode train_eval
```

### 2. Adaptive Entropy-Aware Loss
Quick profile:

```bash
python train_adaptive_loss.py
```

Full profile:

```bash
python train_adaptive_loss.py --profile full
```

### 3. Similarity-Aware Adaptive Loss
Quick profile:

```bash
python train_similarity_aware_loss.py
```

Full profile:

```bash
python train_similarity_aware_loss.py --profile full
```

## Evaluation

### Evaluate Cross Entropy Baseline
```bash
python main.py --mode eval
```

### Similarity-Aware Evaluation Outputs
The similarity-aware pipeline saves:
- best model checkpoint
- final metrics JSON
- classification report
- test confusion matrix
- training curves
- similarity matrix heatmap
- Optuna study outputs

## Public Repository Structure

```text
.
├── figures/
│   ├── adaptive_confusion_matrix.png
│   ├── adaptive_training_curves.png
│   ├── ce_confusion_matrix.png
│   ├── similarity_confusion_matrix.png
│   ├── similarity_matrix_heatmap.png
│   └── similarity_training_curves.png
├── reports/
├── LICENSE
├── README.md
├── main.py
├── requirements.txt
├── train_adaptive_loss.py
├── train_similarity_aware_loss.py
└── .gitignore
```

Local-only folders such as dataset copies, checkpoints, logs, and caches are intentionally excluded from version control for a cleaner public release.

## Recommended Pre-Upload Cleanup
Before pushing publicly, keep the repository lightweight and research-focused:

### Exclude from Git
- `flowers102/`
- `checkpoints/`
- `checkpoints_adaptive_loss/`
- `checkpoints_similarity_loss/`
- `__pycache__/`
- large `.pth` files
- intermediate `.npy` artifacts used only for local experimentation
- training logs and temporary outputs

### Keep in Git
- core training scripts
- curated figures
- README
- license
- requirements
- selected report assets and publication-ready documentation

## Future Work
- feature-embedding-based similarity estimation
- dynamic similarity matrices updated during training
- cosine-similarity class prototypes
- attention-aware or region-aware loss formulations
- transformer backbones for fine-grained classification
- self-supervised similarity estimation
- Grad-CAM and interpretability analysis

## References
1. S. Nilsback and A. Zisserman, “Automated Flower Classification over a Large Number of Classes,” 2008.
2. C. Szegedy et al., “Rethinking the Inception Architecture for Computer Vision,” CVPR 2016.
3. T.-Y. Lin et al., “Focal Loss for Dense Object Detection,” ICCV 2017.
4. C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.

## License
This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

## Citation
If you use this repository in academic or technical work, please cite it as:

```bibtex
@misc{theegala2026similarityawareflowers,
  title        = {Similarity-Aware Adaptive Loss for Fine-Grained Flower Classification},
  author       = {Yeseswini Theegala and Aditya Chauhan and Harshit Patidhar},
  year         = {2026},
  note         = {PyTorch implementation and research repository},
  howpublished = {\url{https://github.com/<your-username>/<your-repository-name>}}
}
```

## Repository Positioning
This repository is intended to present:
- a serious fine-grained visual classification study,
- a custom research-motivated loss design,
- reproducible experimentation in PyTorch,
- and a portfolio-ready machine learning project with publication-style documentation.
