import matplotlib.pyplot as plt
import numpy as np

conditions = ['Normal', 'Mild', 'Medium', 'Severe']
precision = [0.673, 0.661, 0.631, 0.560]
recall    = [0.538, 0.515, 0.455, 0.358]

prec_drops = [2, 6, 17]
rec_drops  = [4, 15, 33]

x = np.arange(len(conditions))

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(x, precision, color='#2E5C8A', linewidth=2, marker='o', markersize=7, label='Precision')
ax.plot(x, recall,    color='#C04545', linewidth=2, marker='s', markersize=7, label='Recall')

for i in range(1, len(conditions)):
    ax.annotate(f'-{prec_drops[i-1]:.0f}%', (x[i], precision[i]),
                xytext=(0, 10), textcoords='offset points',
                ha='center', color='#2E5C8A', fontsize=10, fontweight='bold')
    ax.annotate(f'-{rec_drops[i-1]:.0f}%', (x[i], recall[i]),
                xytext=(0, -14), textcoords='offset points',
                ha='center', color='#C04545', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_xlabel('Illumination Condition')
ax.set_ylabel('Metric Value')
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig('figure_3_1.png', dpi=300, bbox_inches='tight')
plt.show()
