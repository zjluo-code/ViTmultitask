import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. Set plot style
# -----------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# 2. Read data files (nommd_pred_256bs.dat & mmd_pred_256bs.dat)
# -----------------------------------------------------------------------------
path_no_mmd = 'nommd_pred_256bs.dat'
path_mmd = 'mmd_pred_256bs.dat'

try:
    # Read predictions without MMD
    df_no_mmd = pd.read_csv(path_no_mmd, sep=r'\s+', header=0, engine='python')
    y_true_no_mmd = df_no_mmd.iloc[:, 1:5].values.astype(float)
    y_pred_no_mmd = df_no_mmd.iloc[:, 5:9].values.astype(float)

    # Read predictions with MMD
    df_mmd = pd.read_csv(path_mmd, sep=r'\s+', header=0, engine='python')
    y_true_mmd = df_mmd.iloc[:, 1:5].values.astype(float)
    y_pred_mmd = df_mmd.iloc[:, 5:9].values.astype(float)

    print("Successfully loaded nommd_pred_256bs.dat and mmd_pred_256bs.dat data files!\n")

except Exception as e:
    print(f"Data reading failed, please check file path or format: {e}")
    exit()

# -----------------------------------------------------------------------------
# 3. Task and physical quantity label definitions (order: redshift, mass, Dn4000, HdeltaA)
# -----------------------------------------------------------------------------
tasks = [
    {"data_idx": 2, "key": "z", "name": r"Redshift ($z$)", "unit": ""},
    {"data_idx": 1, "key": "mass", "name": r"Stellar Mass ($\log M_*$)", "unit": r"$\log M_{\odot}$"},
    {"data_idx": 0, "key": "Dn4000", "name": r"$D_n4000$", "unit": ""},
    {"data_idx": 3, "key": "ew_hd", "name": r"$\mathrm{H}\delta_A$", "unit": r"$\mathrm{\AA}$"}
]

subplot_labels = ['(a)', '(b)', '(c)', '(d)']

# -----------------------------------------------------------------------------
# 4. Plot 2x2 KDE probability density distribution comparison
# -----------------------------------------------------------------------------
# Slightly widen the canvas to 10.0 to give enough space for left and right columns
fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), dpi=100)
axes = axes.flatten()

for i, task in enumerate(tasks):
    ax = axes[i]
    idx = task["data_idx"] 
    
    yt = y_true_mmd[:, idx]          
    yp_no_mmd = y_pred_no_mmd[:, idx] 
    yp_mmd = y_pred_mmd[:, idx]      
    
    # 1. Ground Truth
    sns.kdeplot(yt, ax=ax, color='black', facecolor='#e5e5e5', edgecolor='black',
                linewidth=1.6, label='Ground Truth', fill=True, zorder=1)
    
    # 2. Baseline Model (w/o MMD)
    sns.kdeplot(yp_no_mmd, ax=ax, color='#1f77b4', linestyle='--', linewidth=1.8, 
                label='w/o MMD', zorder=2)
    
    # 3. Proposed Model (w/ MMD)
    sns.kdeplot(yp_mmd, ax=ax, color='#d62728', linestyle='-', linewidth=1.8, 
                label='w/ MMD', zorder=3)
    
    # Axis tick and border settings (thickened)
    ax.tick_params(direction='in', which='both', top=True, right=True, 
                    width=1.2, length=5, labelsize=11)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        
    title_str = task['name'] + (f" [{task['unit']}]" if task['unit'] else "")
    ax.set_title(title_str, fontsize=12, pad=6)
    ax.set_xlabel(f"{task['name']}", fontsize=11, labelpad=4)
    ax.set_ylabel("Probability Density", fontsize=11, labelpad=4)
    
    # Subplot label
    ax.text(0.04, 0.95, subplot_labels[i], transform=ax.transAxes, 
            fontsize=12, fontweight='bold', ha='left', va='top')
            
    if i == 0:
        ax.legend(frameon=False, fontsize=11, loc='upper right')

# [Key modification]: increase w_pad (horizontal spacing) and h_pad (vertical spacing) to completely avoid overlap
plt.tight_layout(pad=1.2, w_pad=3.0, h_pad=2.5)

# -----------------------------------------------------------------------------
# 5. Export figures
# -----------------------------------------------------------------------------
plt.savefig('kde_density_distribution_comparison.eps', format='eps', bbox_inches='tight')
plt.savefig('kde_density_distribution_comparison.png', dpi=300, bbox_inches='tight')
print("Figures successfully exported as kde_density_distribution_comparison.eps and kde_density_distribution_comparison.png!")

plt.show()
