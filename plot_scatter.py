import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Define evaluation metric calculation functions
# -----------------------------------------------------------------------------
def calculate_metrics(y_true, y_pred, task_key):
    """
    Calculate evaluation metrics for a single physical quantity:
    - Redshift (z): uses normalised residual \Delta z_i = (\hat{z}_i - z_i) / (1 + z_i)
    - Other physical quantities: uses physical residual \Delta y_{i,k} = \hat{y}_{i,k} - y_{i,k}
    - Statistical metrics: RMSE, MAE, sigma_NMAD, Outlier Rate (eta)
    """
    N = len(y_true)
    
    # Compute the defined residual \Delta y_{i,k} according to quantity type
    if task_key == 'z':
        # \Delta z_i = (\hat{z}_i - z_i) / (1 + z_i)
        delta_y = (y_pred - y_true) / (1.0 + y_true)
    else:
        # \Delta y_{i,k} = \hat{y}_{i,k} - y_{i,k}
        delta_y = y_pred - y_true
    
    # Point-to-point statistical performance (RMSE and MAE)
    rmse = np.sqrt(np.mean(delta_y ** 2))
    mae = np.mean(np.abs(delta_y))
    
    # Normalised median absolute deviation (\sigma_{\mathrm{NMAD}, k})
    median_delta = np.median(delta_y)
    nmad = 1.4826 * np.median(np.abs(delta_y - median_delta))
    
    # Outlier indicator factor (C_{i,k}) determination
    if task_key == 'z':
        # |\Delta z_i| > 0.15 (k = z)
        outlier_mask = np.abs(delta_y) > 0.15
    else:
        # |\Delta y_{i,k}| > 3 * \sigma_{\mathrm{NMAD}, k} (k \neq z)
        outlier_mask = np.abs(delta_y) > (3.0 * nmad)
        
    eta = (np.sum(outlier_mask) / N) * 100.0  # Outlier percentage \eta_k
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "sigma_NMAD": nmad,
        "Eta (%)": eta
    }, outlier_mask

# -----------------------------------------------------------------------------
# 2. Load data and handle file header
# -----------------------------------------------------------------------------
data_path = 'mmd_pred_256bs.dat'

try:
    df = pd.read_csv(data_path, sep=r'\s+', header=0, engine='python')
    
    # Original data column order (assumed: Dn4000, mass, z, ew_hd)
    raw_y_true = df.iloc[:, 1:5].values.astype(float)
    raw_y_pred = df.iloc[:, 5:9].values.astype(float)
    
    print(f"Successfully loaded {len(raw_y_true)} valid test samples!\n")

except Exception as e:
    print(f"Data reading failed, please check file path and format: {e}")
    exit()

# -----------------------------------------------------------------------------
# 3. Configure subplot arrangement order and label identifiers
# Subplot layout order: (a) Redshift, (b) Mass, (c) Dn4000, (d) Hdelta_A
# -----------------------------------------------------------------------------
tasks = [
    {"key": "z",      "raw_idx": 2, "label": "(a)", "name": r"Redshift ($z$)",              "unit": ""},
    {"key": "mass",   "raw_idx": 1, "label": "(b)", "name": r"Stellar Mass ($\log M_*$)", "unit": r"$\log M_{\odot}$"},
    {"key": "Dn4000", "raw_idx": 0, "label": "(c)", "name": r"$D_n4000$",                   "unit": ""},
    {"key": "ew_hd",  "raw_idx": 3, "label": "(d)", "name": r"$\mathrm{H}\delta_A$",         "unit": r"$\mathrm{\AA}$"}
]

metrics_results = []
outlier_masks = []
y_true_ordered = []
y_pred_ordered = []

print("=" * 68)
print(f"{'Task':<25} | {'RMSE':<8} | {'MAE':<8} | {'σ_NMAD':<8} | {'η (%)':<8}")
print("-" * 68)

for task in tasks:
    idx = task["raw_idx"]
    yt = raw_y_true[:, idx]
    yp = raw_y_pred[:, idx]
    
    y_true_ordered.append(yt)
    y_pred_ordered.append(yp)
    
    res, out_mask = calculate_metrics(yt, yp, task["key"])
    metrics_results.append(res)
    outlier_masks.append(out_mask)
    
    # Print plain text name without LaTeX characters
    clean_name = task['name'].replace('$', '').replace(r'\mathrm', '')
    print(f"{clean_name:<25} | {res['RMSE']:<8.4f} | {res['MAE']:<8.4f} | {res['sigma_NMAD']:<8.4f} | {res['Eta (%)']:<8.2f}%")

print("=" * 68)

# -----------------------------------------------------------------------------
# 4. Plot 2x2 scatter comparison figure
# -----------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(9, 9), dpi=100)
axes = axes.flatten()

for i, task in enumerate(tasks):
    ax = axes[i]
    yt = y_true_ordered[i]
    yp = y_pred_ordered[i]
    res = metrics_results[i]
    out_mask = outlier_masks[i]
    
    # Set axis limits
    if task["key"] == "z":
        lims = [0.0, 0.8]          # Subplot (a) z limits
    elif task["key"] == "Dn4000":
        lims = [0.9, 2.2]          # Subplot (c) Dn4000 limits
    elif task["key"] == "ew_hd":
        lims = [-5.0, 8.0]         # Subplot (d) Hdelta_A limits
    else:
        # Subplot (b) Stellar Mass: adaptive limits
        min_val = min(np.min(yt), np.min(yp))
        max_val = max(np.max(yt), np.max(yp))
        margin = (max_val - min_val) * 0.05
        lims = [min_val - margin, max_val + margin]
    
    # Axis ticks inward and thickened
    ax.tick_params(direction='in', which='both', top=True, right=True, 
                    width=1.2, length=5, labelsize=12)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # Thicken border lines
    
    # 1:1 ideal reference line
    ax.plot(lims, lims, color='black', linestyle='--', linewidth=1.8, zorder=1, label='1:1 Line')
    
    # Scatter points: normal points (blue dots) and outliers (red crosses)
    ax.scatter(yt[~out_mask], yp[~out_mask], c='#1f77b4', alpha=0.35, s=8, edgecolors='none', label='Inliers', zorder=2)
    ax.scatter(yt[out_mask], yp[out_mask], c='#d62728', alpha=0.6, s=14, marker='x', linewidths=1.2, label='Outliers', zorder=3)
    
    # Title (without subplot label) and axis labels
    title_str = task['name'] + (f" [{task['unit']}]" if task['unit'] else "")
    ax.set_title(title_str, fontsize=12, pad=4)
    ax.set_xlabel(f"Spectroscopic True {task['name']}", fontsize=13)
    ax.set_ylabel(f"Predicted {task['name']}", fontsize=13)
    
    # In-plot statistics (placed in upper left corner)
    stats_text = (
        f"RMSE = {res['RMSE']:.4f}\n"
        f"MAE = {res['MAE']:.4f}\n"
        f"$\\sigma_{{\\mathrm{{NMAD}}}} = {res['sigma_NMAD']:.4f}$\n"
        f"$\\eta = {res['Eta (%)']:.2f}\\%$"
    )
    ax.text(
        0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', horizontalalignment='left'
    )
    
    # Subplot label (a), (b), (c), (d) (placed in lower right corner)
    ax.text(
        0.95, 0.05, task['label'], transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='bottom', horizontalalignment='right'
    )
    
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.6)

# Save high-resolution 300 DPI figure for publication
plt.savefig('physical_parameters_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()
