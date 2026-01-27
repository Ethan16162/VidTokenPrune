import matplotlib.pyplot as plt
import numpy as np

# 原始数据（单位：ms）
frames = [32, 64, 128, 256]
e2e_ms = [667.7, 1406.8, 2043.1, 2871.3]
dpp_ms = [153.0, 307.6, 517.1, 602.5]

# 转换为秒
e2e = [x / 1000.0 for x in e2e_ms]
dpp = [x / 1000.0 for x in dpp_ms]

x = np.arange(len(frames))
width = 0.4

fig, ax = plt.subplots(figsize=(8, 5))

# E2E 柱状图（使用 #DAE3F5）
bars_e2e = ax.bar(x, e2e, width, label='E2E Latency (s)', color='#FFC3BF', edgecolor='black')

# DPP 柱状图（使用 #FFC3BF，虚线边框 + 半透明）
bars_dpp = ax.bar(x, dpp, width, label='DPP Cost (s)',
                  color='#DAE3F5', alpha=0.8,
                  edgecolor='black')

# 设置坐标轴
ax.set_xlabel('# Frames')
ax.set_ylabel('Latency (s)')
ax.set_title('E2E vs DPP Latency across Frame Counts')
ax.set_xticks(x)
ax.set_xticklabels(frames)
ax.legend(loc='upper left')
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
# plt.show()
plt.savefig('plot/e2e_dpp_seconds.png', dpi=300, bbox_inches='tight')