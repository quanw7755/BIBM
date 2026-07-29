[README.md](https://github.com/user-attachments/files/30509980/README.md)
# BIBM# FSI-CLIP

PyTorch implementation of **FSI-CLIP: Frequency-Structure Interaction for Zero-Shot Medical Anomaly Detection**.

FSI-CLIP addresses the frequency-structure under-modeling problem in CLIP-based medical anomaly detection through two complementary modules:

- **Frequency Token Enhancement (FTE):** enhances subtle frequency-sensitive abnormal cues in CLIP patch tokens.
- **Hierarchical Region Multi-scale Interaction (HRMI):** models cross-scale and region-level relationships to improve anomaly-map consistency.

The CLIP visual and text encoders remain frozen, while the learnable prompts, FTE, and HRMI are optimized on an auxiliary anomaly-detection source dataset and directly evaluated on unseen medical datasets.

<p align="center">
  <img src="./pic/model.png" width="95%" alt="FSI-CLIP framework">
</p>

## Highlights

- Zero-shot transfer from industrial source datasets to unseen medical targets.
- Frequency-aware enhancement in the CLIP token space rather than on raw images.
- Hierarchical region interaction for reducing scattered and fragmented anomaly responses.
- Joint support for image-level anomaly classification and pixel-level anomaly localization.
- A single configuration is retained across target datasets without target-domain training.

## Experimental Results

FSI-CLIP is evaluated on nine medical anomaly-detection benchmarks:

- **Image-level and/or pixel-level:** BrainMRI, Br35H, LiverCT, and RESC.
- **Image-level:** OCT17.
- **Pixel-level:** ISIC, ClinicDB, ColonDB, and Kvasir.

The paper reports overall average scores of **87.0%** on the first benchmark group and **87.1%** on the LiverCT/RESC/OCT17 group.

## Environment

The code is implemented with Python and PyTorch. A typical environment can be created as follows:

```bash
conda create -n fsi_clip python=3.10 -y
conda activate fsi_clip
pip install -r requirements.txt
```

The main dependencies include:

- Python 3.10
- PyTorch
- torchvision
- OpenAI CLIP
- NumPy
- SciPy
- scikit-learn
- Pillow
- OpenCV
- tqdm
- PyYAML

Please install the CUDA-compatible PyTorch version for your system.

## Dataset Preparation

Please organize the datasets under a common root directory. The exact inner structure should follow the corresponding dataset loader in the repository.

A recommended layout is:

```text
datasets/
├── mvtec/
│   ├── bottle/
│   │   ├── train/
│   │   │   └── good/
│   │   ├── test/
│   │   │   ├── good/
│   │   │   └── ...
│   │   └── ground_truth/
│   │       └── ...
│   └── ...
├── visa/
│   ├── candle/
│   │   └── Data/
│   │       ├── Images/
│   │       │   ├── Normal/
│   │       │   └── Anomaly/
│   │       └── Masks/
│   │           └── Anomaly/
│   ├── ...
│   └── split_csv/
├── BrainMRI/
│   ├── no/
│   └── yes/
├── Br35H/
│   ├── no/
│   └── yes/
├── ISIC2016/
│   ├── ISBI2016_ISIC_Part1_Test_Data/
│   └── ISBI2016_ISIC_Part1_Test_GroundTruth/
├── CVC-ClinicDB/
│   ├── images/
│   └── masks/
├── CVC-ColonDB/
│   ├── images/
│   └── masks/
├── Kvasir/
│   ├── images/
│   └── masks/
├── LiverCT/
├── RESC/
└── OCT17/
```

For **LiverCT**, **RESC**, and **OCT17**, keep the original downloaded data and configure their image, label, and mask paths according to the corresponding loader.

The source-domain training protocol is:

- Use **MVTec AD** or **VisA** as the auxiliary anomaly-detection source.
- Evaluate directly on unseen medical target datasets.
- Do not use target-domain images for training or parameter optimization.

## Configuration

Set the dataset root and experiment options in `train.sh`, `test.sh`, or the corresponding configuration file.

For example:

```bash
DATA_DIR=/path/to/datasets
```

The default settings reported in the paper are:

| Setting | Value |
|---|---|
| Backbone | CLIP ViT-L/14 |
| Input resolution | 518 × 518 |
| Feature layers | 6, 12, 18, 24 |
| Prompt length | 12 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Training epochs | 2 |
| Segmentation-loss weight `lambda_1` | 1.0 |
| Patch-alignment weight `lambda_2` | 0.1 |

All target datasets use the same predefined configuration.

## Training

After setting the dataset and output paths, train FSI-CLIP with:

```bash
bash train.sh
```

The training stage optimizes the learnable prompts, FTE, and HRMI while keeping the CLIP visual and text encoders frozen.

## Evaluation

Evaluate the trained model with:

```bash
bash test.sh
```

The evaluation reports:

- **I-AUROC:** image-level area under the ROC curve.
- **I-AP:** image-level average precision.
- **P-AUROC:** pixel-level area under the ROC curve.
- **P-PRO:** pixel-level per-region overlap.

Only the metrics applicable to each dataset are reported.

## Reproducibility Notes

- Use the same source split, random seed, checkpoint, and preprocessing configuration when comparing model variants.
- Do not tune hyperparameters on target-domain annotations.
- Keep the CLIP encoders frozen for all experiments.
- For random-seed analysis, run the complete training and evaluation pipeline independently for each seed.
- Dataset paths and class names are case-sensitive on Linux systems.

## Model Components

### Frequency Token Enhancement

FTE reshapes CLIP patch tokens into two-dimensional feature maps, performs low- and high-frequency decomposition in the token-space Fourier domain, and adaptively fuses the two components through learnable gates and residual refinement.

### Hierarchical Region Multi-scale Interaction

HRMI combines:

- cross-layer scale weighting,
- multi-grid regional pooling,
- discrepancy-aware feature interaction,
- adaptive region gating, and
- gated residual enhancement.

These operations encourage spatially coherent anomaly responses while preserving the original CLIP representation.

## Citation

For anonymous review, please use the following temporary entry:

```bibtex
@inproceedings{anonymous2026fsiclip,
  title     = {FSI-CLIP: Frequency-Structure Interaction for Zero-Shot Medical Anomaly Detection},
  author    = {Anonymous},
  booktitle = {IEEE International Conference on Bioinformatics and Biomedicine},
  year      = {2026}
}
```

The citation information will be updated after the review process.

## Acknowledgements

This implementation builds upon publicly available CLIP-based anomaly-detection codebases, including CLIP and AF-CLIP. We thank the authors of the corresponding projects and datasets for making their work publicly available.

## Contact

For anonymous review, please use the repository issue tracker for questions related to installation, dataset preparation, training, or evaluation.
