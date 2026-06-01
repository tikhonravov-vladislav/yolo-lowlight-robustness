import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Computer Modern Roman', 'Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['lines.linewidth'] = 1.6
mpl.rcParams['lines.markersize'] = 6

conditions = ['Normal', 'Mild', 'Medium', 'Severe']

data = {
    'Baseline': {
        'P':         [0.673, 0.661, 0.631, 0.560],
        'R':         [0.538, 0.515, 0.455, 0.358],
        'F1':        [0.598, 0.579, 0.529, 0.437],
        'mAP50-95':  [0.430, 0.409, 0.351, 0.266],
    },
    'CBRT': {
        'P':         [0.651, 0.650, 0.636, 0.613],
        'R':         [0.511, 0.507, 0.486, 0.443],
        'F1':        [0.572, 0.569, 0.550, 0.514],
        'mAP50-95':  [0.404, 0.396, 0.374, 0.330],
    },
    'Naive': {
        'P':         [0.656, 0.663, 0.646, 0.622],
        'R':         [0.520, 0.511, 0.486, 0.446],
        'F1':        [0.580, 0.578, 0.554, 0.519],
        'mAP50-95':  [0.409, 0.402, 0.380, 0.338],
    },
}

styles = {
    'Baseline': {'color': '#333333', 'marker': 'o', 'linestyle': '-',  'label': 'Baseline'},
    'CBRT':     {'color': '#1f77b4', 'marker': 's', 'linestyle': '-',  'label': 'CBRT'},
    'Naive':    {'color': '#2ca02c', 'marker': '^', 'linestyle': '--', 'label': 'Naive'},
}

metric_panels = [
    ('P',        'Precision'),
    ('R',        'Recall'),
    ('F1',       'F1-score'),
    ('mAP50-95', 'mAP@0.5:0.95'),
]

fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), sharex=True)
axes = axes.flatten()

for i, (key, title) in enumerate(metric_panels):
    ax = axes[i]
    for model, style in styles.items():
        ax.plot(conditions, data[model][key], **style)
    ax.set_ylabel(title)
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(direction='in', length=3)
    # Subtle panel label (a), (b), (c), (d) in upper-right
    ax.text(0.97, 0.95, f'({chr(97 + i)})',
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=10, style='italic', color='#555555')

for ax in axes[2:]:
    ax.set_xlabel('Illumination condition')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center',
           ncol=3,
           bbox_to_anchor=(0.5, -0.02),
           frameon=False)

plt.tight_layout(rect=[0, 0.04, 1, 1])

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, '..', 'figures')
os.makedirs(out_dir, exist_ok=True)

out_pdf = os.path.join(out_dir, 'fig_6_1_panel.pdf')
out_png = os.path.join(out_dir, 'fig_6_1_panel.png')

plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(out_png, bbox_inches='tight', dpi=200)

print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
