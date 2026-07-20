import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------

df_xlstm_individual = pd.read_csv(
    'results/xLSTM_results_20260608_090620.csv'
)

df_xlstm_large = pd.read_csv(
    'results/xLSTM_LARGE_results_20260612_065528.csv'
)

df_xlstm_transfer = pd.read_csv(
    'results/xLSTM_TRANSFER_results_20260715_153842.csv'
)


# Springs used for transfer evaluation
transfer_springs = sorted(df_xlstm_transfer['spring_id'].unique())


# Keep only relevant springs
df_xlstm_individual = df_xlstm_individual[
    df_xlstm_individual['spring_id'].isin(transfer_springs)
]

df_xlstm_large = df_xlstm_large[
    df_xlstm_large['spring_id'].isin(transfer_springs)
]

df_xlstm_transfer = df_xlstm_transfer[
    df_xlstm_transfer['spring_id'].isin(transfer_springs)
]


# Only use horizon 1
df_xlstm_individual = df_xlstm_individual[
    df_xlstm_individual['horizon'] == 1
]

df_xlstm_large = df_xlstm_large[
    df_xlstm_large['horizon'] == 1
]

df_xlstm_transfer = df_xlstm_transfer[
    df_xlstm_transfer['horizon'] == 1
]


# ------------------------------------------------------------------
# Plot setup
# ------------------------------------------------------------------

variables = [
    {
        "column": "emissions training [kg CO₂]",
        "title": "xLSTM Training CO$_2$",
        "ylabel": "g CO$_2$",
        "models": ["individual", "transfer"],
        "scale": 1000
    },
    {
        "column": "energy training [kWh]",
        "title": "xLSTM Training Energy",
        "ylabel": "Wh",
        "models": ["individual", "transfer"],
        "scale": 1000
    },
    {
        "column": "emissions inference [kg CO₂]",
        "title": "xLSTM Inference CO$_2$",
        "ylabel": "g CO$_2$",
        "models": ["individual", "large", "transfer"],
        "scale": 1000
    },
    {
        "column": "energy inference [kWh]",
        "title": "xLSTM Inference Energy",
        "ylabel": "Wh",
        "models": ["individual", "large", "transfer"],
        "scale": 1000
    },
]


fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 12)
)

axes = axes.flatten()


x = np.arange(len(transfer_springs))

bar_width = 0.25

bar_color = "#0072B2"
edge_color = "black"


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

for idx, (ax, var) in enumerate(zip(axes, variables)):

    models = var["models"]

    # Make bars touch:
    # 2 models -> positions -0.5, +0.5
    # 3 models -> positions -1, 0, +1
    n_models = len(models)

    if n_models == 2:
        offsets = np.array([-0.5, 0.5]) * bar_width
    else:
        offsets = np.array([-1, 0, 1]) * bar_width


    for model, offset in zip(models, offsets):

        if model == "individual":
            df = df_xlstm_individual
            label = "Individual"
            hatch = ""

        elif model == "large":
            df = df_xlstm_large
            label = "Multi-spring"
            hatch = "////"

        elif model == "transfer":
            df = df_xlstm_transfer
            label = "Fine-tuned"
            hatch = "xxxx"


        values = (
            df
            .set_index("spring_id")[var["column"]]
            .reindex(transfer_springs)
            .values
            * var["scale"]
        )


        ax.bar(
            x + offset,
            values,
            bar_width,
            label=label,
            color=bar_color,
            edgecolor=edge_color,
            hatch=hatch,
            linewidth=1
        )


    ax.set_title(
        var["title"],
        fontweight="bold"
    )

    ax.set_ylabel(
        var["ylabel"]
    )

    ax.set_xlabel(
        "Spring ID"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(transfer_springs)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.7
    )


    # Only legend in top-left subplot
    # if idx == 0:
    ax.legend(loc="upper left")

    # Add extra headroom for legend
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.25)


plt.tight_layout()


plt.savefig(
    "evaluation/transfer/plots/xlstm_transfer_energy_barplots.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()