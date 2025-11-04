import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('results.csv')


fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# 1：Runtime
pivot1 = df.pivot(index='nBlocks', columns='nThreads', values='Runtime_ms')
sns.heatmap(pivot1, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=axes[0],
            cbar_kws={'label': 'Runtime (ms)'}, 
            norm=plt.matplotlib.colors.LogNorm())
axes[0].set_title('GPU Kernel Runtime (ms)', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Threads per Block', fontsize=12)
axes[0].set_ylabel('Number of Blocks', fontsize=12)

# 2：Occupancy
pivot2 = df.pivot(index='nBlocks', columns='nThreads', values='Occupancy_pct')
sns.heatmap(pivot2, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1],
            cbar_kws={'label': 'Occupancy (%)'}, vmin=0, vmax=100)
axes[1].set_title('Achieved Occupancy (%)', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Threads per Block', fontsize=12)
axes[1].set_ylabel('Number of Blocks', fontsize=12)

# 3：Memory Bandwidth
pivot3 = df.pivot(index='nBlocks', columns='nThreads', values='MemBW_pct')
sns.heatmap(pivot3, annot=True, fmt='.1f', cmap='Blues', ax=axes[2],
            cbar_kws={'label': 'Memory BW (%)'}, vmin=0, vmax=45)
axes[2].set_title('Memory Bandwidth Utilization (%)', fontsize=16, fontweight='bold')
axes[2].set_xlabel('Threads per Block', fontsize=12)
axes[2].set_ylabel('Number of Blocks', fontsize=12)

plt.tight_layout()
plt.savefig('cuda_performance_heatmaps.png', dpi=300, bbox_inches='tight')
print('✅ Heatmaps saved as cuda_performance_heatmaps.png')
plt.show()