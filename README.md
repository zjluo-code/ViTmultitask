# ViTmultitask

ViTMultiTask: Mitigating Variance Shrinkage in Multi-task Galaxy Property Inference from Imaging Data via Physical Manifold Constraints

Official implementation for the paper: " Mitigating Variance Shrinkage in Multi-task Galaxy Property Inference from Imaging Data via Physical Manifold Constraints".

1.	Overview

ViTMultiTask is a deep learning framework designed to simultaneously estimate high-dimensional galaxy physical parameters—redshift ($z$), stellar mass ($\log M_*/\mathrm{M}_\odot$), $D_n 4000$, and $\mathrm{H}\delta_A$ equivalent width—from $64 \times 64$ five-band Hyper Suprime-Cam (HSC) optical cutouts.By integrating a Vision Transformer (ViT) backbone with a Maximum Mean Discrepancy (MMD) regularization penalty, the network preserves intrinsic multi-parameter physical manifolds (e.g., the $D_n 4000$--$\mathrm{H}\delta_A$ evolutionary sequence) and mitigates standard MSE-induced variance shrinkage.

2.	Repository Structure

├── hsc_mmd_256bs.py          # Main script for ViTMultiTask model training and evaluation

├── plot_scatter.py           # Plotting script: physical parameter scatter & residual evaluation

├── plot_1D_kde.py            # Plotting script: 1D KDE density distribution comparison

├── plot_2D_kde.py            # Plotting script: 2D KDE contour overlays on the Dn4000-HdeltaA manifold

├── mmd_pred_256bs.dat        # Model predictions on the test set (ViTMultiTask with MMD)

├── nommd_pred_256bs.dat      # Baseline predictions on the test set (ViTMultiTask without MMD)

├── vit_multitask_arch.png    # Model architecture diagram

├── desi_hsc_mmd_256bs/       # Pre-trained checkpoint directory for ViTMultiTask (with MMD)

├── desi_hsc_nommd_256bs/     # Pre-trained checkpoint directory for baseline ViTMultiTask (without MMD)

└── README.md                 # Documentation

3.	Environment Setup

Ensure you have Python 3.8+ and standard scientific/deep learning packages installed:
pip install tensorflow numpy scipy matplotlib seaborn scikit-learn

4.	Usage

1） Training & Inference
To train the ViTMultiTask network or run inference on the HSC test set:

python hsc_mmd_256bs.py

Pre-trained weights are automatically loaded from desi_hsc_mmd_256bs/ and desi_hsc_nommd_256bs/.

The evaluation outputs prediction files (mmd_pred_256bs.dat and nommd_pred_256bs.dat).

2.）Reproducing Paper Figures
Run the provided evaluation scripts to regenerate the diagnostic figures from the prediction data:
Scatter & Residual Analysis:   
            python plot_scatter.py       # Generates: physical_parameters_evaluation.eps


1D Distribution Profile Consistency:
            python plot_1D_kde.py        # Generates: kde_density_distribution_comparison.eps


2D Astrophysical Manifold Reconstruction ($D_n 4000$ vs $\mathrm{H}\delta_A$):
            python plot_2D_kde.py        # Generates: 2D_KDE_overlay_Dn4000_vs_HdeltaA.eps

5.	Pre-trained Checkpoints & Data

The pre-trained network weights and dataset identifiers are archived and publicly accessible:
Zenodo Archive: 10.5281/zenodo.xxxxxxxx

6.	Contact & Citation

For queries regarding code or dataset cross-matching, please open an Issue or contact Zhijian Luo.

If you find this repository useful for your research, please cite our paper and dataset:
@article{Luo2026ViTMultiTask,
  author    = {Zhijian Luo & Jianzhen Chen},
  title     = { Mitigating Variance Shrinkage in Multi-task Galaxy Property Inference from Imaging Data via Physical Manifold Constraints},
  journal   = {xxxxxxxxxxxxxxxxxxxxxxxxx},
  year      = {2026},
  doi       = {10.5281/zenodo.xxxxxxxx}
}
