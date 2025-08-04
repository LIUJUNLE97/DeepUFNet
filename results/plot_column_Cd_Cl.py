
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Patch


forecast = [1.643, 0.461]
experimental = [1.65, 0.501]
values = forecast + experimental

labels = [
    r'$\overline{C}_d$',
    r'${C}_{l,std}$',
    r'$\overline{C}_d$',
    r'${C}_{l,std}$'
]

x = np.arange(len(values))
colors = ['#E3F0F7'] * 2 + ['#FFEEEC'] * 2


fig, ax = plt.subplots(figsize=(4.5, 4), dpi=300)

bars = []
depth = 0.15  

for i, (xi, val) in enumerate(zip(x, values)):
    if i < 2:  # Forecast
        bar = ax.bar(xi, val, color=colors[i], zorder=2)[0]
    else:  # Experimental with hatching
        bar = ax.bar(xi, val, color=colors[i], hatch='///', edgecolor='gray', zorder=2)[0]
    bars.append(bar)

    
    x0 = bar.get_x()
    y0 = 0
    width = bar.get_width()
    height = bar.get_height()

    
    side = Polygon([
        [x0 + width, y0],
        [x0 + width + depth, y0 + depth],
        [x0 + width + depth, height + depth],
        [x0 + width, height]
    ], closed=True, facecolor='gray', edgecolor='none', alpha=0.2, zorder=1)
    ax.add_patch(side)

    
    top = Polygon([
        [x0, height],
        [x0 + depth, height + depth],
        [x0 + width + depth, height + depth],
        [x0 + width, height]
    ], closed=True, facecolor='lightgray', edgecolor='none', alpha=0.3, zorder=3)
    ax.add_patch(top)



for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=14)


ax.set_xticks(x)
ax.set_xticklabels(labels, ha='center')
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(14)

ax.tick_params(axis='y', direction='in', labelsize=14)
ax.set_ylim(0, max(values) * 1.45)


legend_patches = [
    Patch(facecolor='#E3F0F7', label='Forecast'),
    Patch(facecolor='#FFEEEC', hatch='///', edgecolor='gray', label='Experimental data')
]
ax.legend(handles=legend_patches, fontsize=14)


plt.tight_layout()
plt.savefig('comparison_cd_cl_v3_3dlook.pdf', dpi=300)
