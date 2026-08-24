import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

# ==========================================
# 1. Global style settings
# ==========================================
try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    try:
        plt.style.use("seaborn-paper")
    except OSError:
        plt.style.use("default")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 5.5,
        "axes.labelsize": 6.0,
        "axes.titlesize": 6.0,
        "xtick.labelsize": 5.0,
        "ytick.labelsize": 5.0,
        "legend.fontsize": 5.0,
        "figure.titlesize": 7.0,
        "mathtext.fontset": "cm",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.minor.size": 1.0,
        "ytick.minor.size": 1.0,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "axes.linewidth": 0.6,
        "axes.grid": False,
    }
)

# ==========================================
# 2. Read HDF5 data and apply physical cuts
# ==========================================
h5_path = "./DESI_HSC_BGS_MDnHdA_Matched.h5"

with h5py.File(h5_path, "r") as f:
    z_raw = np.array(f["DESI_redshift"][:], dtype=np.float64)
    mstar_raw = np.array(f["FIT_RES_M"][:], dtype=np.float64)
    d4000_raw = np.array(f["FIT_RES_D4000_n"][:], dtype=np.float64)
    hda_raw = np.array(f["FIT_RES_EW_hd_a"][:], dtype=np.float64)

with np.errstate(invalid="ignore", divide="ignore"):
    log_mstar_raw = np.where(mstar_raw > 0, np.log10(mstar_raw), np.nan)

y_raw = np.column_stack([d4000_raw, mstar_raw, z_raw, hda_raw])

valid_base = (
    ~np.isnan(y_raw).any(axis=1)
    & ~np.isinf(y_raw).any(axis=1)
    & (y_raw > -900).all(axis=1)
)

valid_physical = (
    (z_raw <= 0.8)
    & (mstar_raw >= 1e7)
    & (mstar_raw <= 1e13)
    & (d4000_raw >= 0.8)
    & (d4000_raw <= 2.5)
    & (hda_raw >= -5.0)
    & (hda_raw <= 10.0)
)

valid_mask = valid_base & valid_physical

z_clean = z_raw[valid_mask]
log_mstar_clean = log_mstar_raw[valid_mask]
d4000_clean = d4000_raw[valid_mask]
hda_clean = hda_raw[valid_mask]

# ==========================================
# 3. Plot 2x2 multi-panel figure (EPS‑compatible settings)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(3.3, 3.0), dpi=300)

# [Key modification]: explicitly specify light solid fill colours (facecolor) for EPS, avoiding alpha transparency
plot_config = [
    {
        "data": z_clean,
        "xlabel": r"Redshift $z$",
        "edgecolor": "#1f77b4",
        "facecolor": "#c6dbef",  # light blue solid colour, no alpha
        "panel": "(a)",
        "bins": 40,
        "legend_loc": (0.57,0.88),
    },
    {
        "data": log_mstar_clean,
        "xlabel": r"$\log(M_* / M_\odot)$",
        "edgecolor": "#ff7f0e",
        "facecolor": "#fdd0a2",  # light orange solid colour
        "panel": "(b)",
        "bins": 40,
        "legend_loc": (0.20, 0.88),
    },
    {
        "data": d4000_clean,
        "xlabel": r"$D_n4000$",
        "edgecolor": "#2ca02c",
        "facecolor": "#c7e9c0",  # light green solid colour
        "panel": "(c)",
        "bins": 40,
        "legend_loc": (0.57,0.88),
    },
    {
        "data": hda_clean,
        "xlabel": r"$\mathrm{H}\delta_A\ (\mathrm{\AA})$",
        "edgecolor": "#d62728",
        "facecolor": "#fcbba1",  # light red solid colour
        "panel": "(d)",
        "bins": 40,
        "legend_loc": (0.57,0.88),
    },
]

for idx, (ax, config) in enumerate(zip(axes.flatten(), plot_config)):
    data = config["data"]
    med_val = np.median(data)

    ax.minorticks_on()

    # [Key modification]: explicitly separate color (edge) and facecolor (opaque fill), and remove any alpha parameter
    sns.histplot(
        data=data,
        ax=ax,
        bins=config["bins"],
        color=config["edgecolor"],
        facecolor=config["facecolor"],
        element="step",
        fill=True,
        kde=False,
        stat="count",
        linewidth=0.6,
    )

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_fontsize(4.8)

    ax.axvline(
        med_val,
        color="black",
        linestyle="--",
        linewidth=0.6,
        label=f"Med: {med_val:.2f}",
    )

    ax.text(
        0.03,
        0.90,
        config["panel"],
        transform=ax.transAxes,
        fontsize=7.0,
        fontweight="bold",
        va="top",
    )

    ax.set_xlabel(config["xlabel"], labelpad=1.5)

    if idx % 2 == 0:
        ax.set_ylabel("Count", labelpad=1.5)
    else:
        ax.set_ylabel("")

    ax.legend(
        loc=config["legend_loc"],
        frameon=False,
        handlelength=2.1,
        borderpad=0.15,
        borderaxespad=0.15,
    )

plt.tight_layout(pad= 1.0, h_pad=0.6, w_pad=0.8)

# Export to EPS, PDF and PNG simultaneously
plt.savefig("figures_parameter_distributions.eps", format="eps", bbox_inches="tight")
#plt.savefig("figures_parameter_distributions.pdf", bbox_inches="tight")
plt.savefig("figures_parameter_distributions.png", bbox_inches="tight", dpi=300)

plt.show()
