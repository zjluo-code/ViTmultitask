import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

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
    df_no_mmd = pd.read_csv(path_no_mmd, sep=r'\s+', header=0, engine='python')
    y_true = df_no_mmd.iloc[:, 1:5].values.astype(float)
    y_pred_no_mmd = df_no_mmd.iloc[:, 5:9].values.astype(float)

    df_mmd = pd.read_csv(path_mmd, sep=r'\s+', header=0, engine='python')
    y_pred_mmd = df_mmd.iloc[:, 5:9].values.astype(float)

    print("Successfully loaded nommd_pred_256bs.dat and mmd_pred_256bs.dat data files!\n")

except Exception as e:
    print(f"Data reading failed, please check file path or format: {e}")
    exit()

# Extract Dn4000 (index 0) and HdeltaA (index 3)
x_true, y_true_hd = y_true[:, 0], y_true[:, 3]
x_nommd, y_nommd = y_pred_no_mmd[:, 0], y_pred_no_mmd[:, 3]
x_mmd, y_mmd = y_pred_mmd[:, 0], y_pred_mmd[:, 3]

# -----------------------------------------------------------------------------
# 3. Create a light-grey colormap suitable for EPS/PDF vector graphics
# -----------------------------------------------------------------------------
# Transition from pure white (#ffffff) to light grey (#b0b0b0)
custom_light_greys = LinearSegmentedColormap.from_list(
    'light_greys', ['#ffffff', '#e0e0e0', '#cccccc', '#b0b0b0']
)

# Contour levels for prediction models (4 lines)
pred_levels = [0.25, 0.5, 0.75, 0.9]
pred_kde_kwargs = {
    'bw_adjust': 0.8,
    'levels': pred_levels,
    'thresh': 0.05
}

# Contour levels for Ground Truth shading
gt_levels = [0.25, 0.5, 0.75, 0.9, 1.0]
gt_kde_kwargs = {
    'bw_adjust': 0.8,
    'levels': gt_levels,
    'thresh': 0.05
}

# -----------------------------------------------------------------------------
# 4. Single-panel overlay 2D KDE (shading + 4-level contours)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5), dpi=100)

# 1. Ground Truth: using custom light-grey colormap
# rasterized=True
sns.kdeplot(x=x_true, y=y_true_hd, ax=ax, cmap=custom_light_greys,
            fill=True, alpha=1.0, rasterized=True, **gt_kde_kwargs)

# 2. Baseline Model (w/o MMD): blue dashed 4-level contours
sns.kdeplot(x=x_nommd, y=y_nommd, ax=ax, color='#1f77b4',
            fill=False, linewidths=1.8, linestyles='--', **pred_kde_kwargs)

# 3. Proposed Model (w/ MMD): red solid 4-level contours
sns.kdeplot(x=x_mmd, y=y_mmd, ax=ax, color='#d62728',
            fill=False, linewidths=1.8, linestyles='-', **pred_kde_kwargs)

# Set axis limits
ax.set_xlim(left=0.9, right=2.1)
ax.set_ylim(-4, 7)

# Axis tick and border settings
ax.tick_params(direction='in', which='both', top=True, right=True,
                width=1.2, length=5, labelsize=13)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# Set axis labels and title
ax.set_xlabel(r"$D_n4000$", fontsize=13)
ax.set_ylabel(r"$\mathrm{H}\delta_A \ [\mathrm{\AA}]$", fontsize=13)
fig.suptitle(r"2D Distribution: $D_n4000$ vs $\mathrm{H}\delta_A$", fontsize=13, y=0.96)

# Custom legend handles (Patch color matches the core light grey)
legend_elements = [
    Patch(facecolor='#cccccc', edgecolor='none', label='Ground Truth'),
    Line2D([0], [0], color='#1f77b4', lw=1.8, ls='--', label='w/o MMD'),
    Line2D([0], [0], color='#d62728', lw=1.8, ls='-', label='w/ MMD')
]

ax.legend(handles=legend_elements, frameon=False, fontsize=13, loc='upper right', bbox_to_anchor=(0.98, 0.98))

# Adjust layout
fig.subplots_adjust(top=0.92, bottom=0.12, left=0.12, right=0.96, hspace=0, wspace=0)

# Export to EPS, PDF and PNG
plt.savefig('2D_KDE_overlay_Dn4000_vs_HdeltaA.eps', format='eps', dpi=300, bbox_inches='tight', pad_inches=0.01)
#plt.savefig('2D_KDE_overlay_Dn4000_vs_HdeltaA.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.savefig('2D_KDE_overlay_Dn4000_vs_HdeltaA.png', dpi=300, bbox_inches='tight', pad_inches=0.01)

plt.show()
