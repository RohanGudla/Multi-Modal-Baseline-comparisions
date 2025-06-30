# Multi-Modal Baseline Comparisons for Behavioral Annotation

Comprehensive F1 score benchmarks for video-based facial expression and behavioral marker prediction using multi-modal machine learning approaches.

## 🎯 Overview

This repository provides baseline F1 performance metrics for automated behavioral annotation prediction across 4 experimental paradigms at two temporal sampling rates (0.2 FPS and 1 FPS).

**Dataset**: 81 participants across Big Interview and GENEX datasets  
**Markers**: 47 facial expressions and behavioral annotations  
**Architectures**: Individual classifiers, multi-label, and cross-modal fusion  

## 📊 F1 Score Results Summary

### 0.2 FPS Performance (Efficient Baseline)

| Experiment | Type | Best F1 Score | Top Performers |
|------------|------|---------------|----------------|
| **Experiment 1** | Individual Markers | **0.9737** | Attention, Head Leaning Backward (0.9731) |
| **Experiment 2** | Multi-Label | **0.7803** (Micro) | Macro F1: 0.5400 |
| **Experiment 3** | Cross-Modal | **0.9292** | Joy ↔ Smile bidirectional prediction |

### 1 FPS Performance (High Temporal Resolution)

| Experiment | Type | Best F1 Score | Top Performers |
|------------|------|---------------|----------------|
| **Experiment 1** | Individual Markers | **0.9878** | Attention, Head Leaning Backward (0.9879) |
| **Experiment 2** | Multi-Label | **0.6854** (Micro) | Macro F1: 0.6051 |
| **Experiment 3** | Cross-Modal | **0.8561** | Smile → Joy prediction |
| **Experiment 4** | Multi-Output | **0.6434** | 14 emotions → 33 behaviors |

## 🏆 Top F1 Performance Champions

### Individual Marker Detection (Experiment 1)
**0.2 FPS Champions:**
- **Attention**: 0.9737 F1
- **Head Leaning Backward**: 0.9731 F1  
- **Head Turned Forward**: 0.9684 F1
- **Head Not Tilted**: 0.9644 F1
- **Head Pointing Up**: 0.9631 F1

**1 FPS Champions:**
- **Attention**: 0.9878 F1 (+0.0141 improvement)
- **Head Leaning Backward**: 0.9879 F1 (+0.0148 improvement)
- **Head Not Tilted**: 0.9807 F1
- **Head Pointing Up**: 0.9809 F1

### Cross-Modal Prediction (Experiment 3)
**0.2 FPS Champions:**
- **Joy ↔ Smile**: 0.9292 F1 (bidirectional)
- **Fear → Eye Widen**: 0.8244 F1
- **Surprise → Brow Raise**: 0.8252 F1
- **Brow Raise → Surprise**: 0.8108 F1

**1 FPS Champions:**
- **Smile → Joy**: 0.8561 F1
- **Joy → Smile**: 0.8034 F1
- **Disgust → Nose Wrinkle**: 0.7485 F1

## 📈 0.2 FPS vs 1 FPS Comparison

### Performance Improvements at 1 FPS
- **Emotion Detection**: +0.1535 to +0.2278 F1 improvement
  - Disgust: 0.6530 → 0.8066 (+0.1535)
  - Sadness: 0.5064 → 0.6964 (+0.1900)
  - Confusion: 0.5772 → 0.8050 (+0.2278)
- **Head Pose**: Consistent 0.97+ F1 scores at both rates

### Performance Degradations at 1 FPS  
- **Cross-Modal**: Joy-Smile 0.9292 → 0.8561 (-0.0731)
- **Multi-Label Micro F1**: 0.7803 → 0.6854 (-0.0949)
- **Fine Motor**: Eye/Lip movements show decreased F1

### Key Finding
**0.2 FPS optimal for cross-modal tasks, 1 FPS optimal for emotion detection**

## 📁 Repository Structure

```
Multi-Modal-Baseline-comparisions/
├── experiments_0.2fps/           # 0.2 FPS sampling experiments
│   ├── experiment_1/             # Individual marker prediction
│   │   ├── models/               # Best trained models (.pth)
│   │   ├── results/              # F1 scores and metrics (.json)
│   │   └── logs/                 # Training logs (.csv)
│   ├── experiment_2/             # Multi-label prediction
│   ├── experiment_3/             # Cross-modal prediction
│   └── *.py, *.log              # Scripts and training logs
│
├── experiments_1fps/             # 1 FPS sampling experiments
│   └── 1frame_per_second/        # High temporal resolution
│       ├── experiment_1_1fps/    # Individual marker (1 FPS)
│       ├── experiment_2_1fps/    # Multi-label (1 FPS)
│       ├── experiment_3_1fps/    # Cross-modal (1 FPS)
│       ├── experiment_4_1fps/    # Multi-output (1 FPS)
│       └── *.py, *.log          # Scripts and training logs
│
├── datasets/                     # Data and annotations
│   ├── annotations/              # Manual annotation files (.csv, .xlsx)
│   ├── preprocessed_data/        # 0.2 FPS preprocessed (.npz)
│   └── model_outputs/            # Prediction results (.npz)
│
└── scripts/                      # Utilities
    ├── markers_47.txt           # List of 47 behavioral markers
    └── run_experiment.sh        # Automated execution script
```

## 🔬 Experiment Details

### Experiment 1: Individual Marker Prediction
- **Approach**: 47 separate binary classifiers
- **Architecture**: SmallFastCNN (300K parameters each)
- **Best Performance**: Attention marker (F1: 0.9878 at 1 FPS)
- **Strengths**: Highest F1 scores, modular approach
- **Applications**: Real-time single-marker detection

### Experiment 2: Multi-Label Prediction  
- **Approach**: Single model predicting all 47 markers
- **Architecture**: SmallFastCNN + multi-label heads (4.8M parameters)
- **Best Performance**: Micro F1: 0.7803 at 0.2 FPS
- **Strengths**: Single efficient model
- **Applications**: Simultaneous multi-marker prediction

### Experiment 3: Cross-Modal Prediction
- **Approach**: Bidirectional emotion ↔ behavior mapping
- **Architecture**: Cross-modal fusion networks
- **Best Performance**: Joy ↔ Smile (F1: 0.9292 at 0.2 FPS)
- **Strengths**: Demonstrates emotion-behavior relationships
- **Applications**: Emotion inference from facial behavior

### Experiment 4: Multi-Output Domain Mapping (1 FPS only)
- **Approach**: 14 emotions → 33 behaviors comprehensive mapping  
- **Architecture**: Multi-modal fusion (4.9M parameters)
- **Best Performance**: Average F1: 0.6434
- **Strengths**: Most comprehensive behavioral prediction
- **Applications**: Complete facial expression analysis

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision numpy pandas scikit-learn opencv-python
```

### Running Experiments
```bash
# 0.2 FPS experiments
cd experiments_0.2fps
python experiment_1_individual_markers.py  # Individual markers
python experiment_2_multi_label.py         # Multi-label  
python experiment_3_cross_modal.py         # Cross-modal

# 1 FPS experiments  
cd experiments_1fps/1frame_per_second
python experiment_1_individual_markers.py  # Individual markers (1 FPS)
python experiment_4_multi_modal.py         # Multi-output (1 FPS)
```

### Automated Execution
```bash
./scripts/run_experiment.sh               # Run all experiments
./scripts/run_experiment.sh 1             # Run only experiment 1
```

## 📊 Key Technical Specifications

- **Frame Sampling**: 0.2 FPS (every 5th frame) vs 1 FPS  
- **Image Resolution**: 224x224 pixels
- **Training**: 12-20 epochs with early stopping
- **Hardware**: CUDA-enabled GPU recommended
- **Memory**: 8-24GB GPU memory for largest models

## 🎯 Baseline Achievements

This repository establishes:
- **Highest Individual F1**: 0.9878 (Attention at 1 FPS)
- **Highest Cross-Modal F1**: 0.9292 (Joy ↔ Smile at 0.2 FPS)  
- **Optimal Sampling Rate**: Task-dependent (0.2 FPS for cross-modal, 1 FPS for emotions)
- **Production Readiness**: 0.2 FPS models suitable for real-time applications

## 📄 Citation

```bibtex
@misc{multimodal_baselines_2024,
  title={Multi-Modal Baseline Comparisons for Behavioral Annotation},
  author={Research Team},
  year={2024},
  note={F1 score benchmarks for video-based facial expression prediction}
}
```

---

**Repository Status**: Complete F1 benchmarks established  
**Last Updated**: December 2024  
**Performance**: Production-ready baselines with 0.97+ F1 scores