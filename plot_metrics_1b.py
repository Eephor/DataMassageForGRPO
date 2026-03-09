import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('assets', exist_ok=True)

# 1B data
df_sft_1b = pd.read_csv('sft_1b.tsv', sep='\\t')
df_grpo_1b = pd.read_csv('grpo_1b.tsv', sep='\\t')

plt.style.use('ggplot')
plt.figure(figsize=(10, 5))
plt.plot(df_sft_1b['Step'], df_sft_1b['Training Loss'], marker='o', linestyle='-', color='dodgerblue', linewidth=2)
plt.title('SFT Pre-Warming Training Loss (Llama 3.2 1B Instruct)', fontsize=14, fontweight='bold')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('assets/sft_loss_1b.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:red'
ax1.set_xlabel('Step', fontsize=12)
ax1.set_ylabel('Training Loss', color=color, fontsize=12)
ax1.plot(df_grpo_1b['Step'], df_grpo_1b['Training Loss'], color=color, linewidth=2, label='Training Loss')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
colors = ['tab:blue', 'tab:green', 'tab:purple', 'tab:orange']
ax2.set_ylabel('Rewards', color='black', fontsize=12) 
ax2.plot(df_grpo_1b['Step'], df_grpo_1b['reward'], color=colors[0], linewidth=1.5, label='Overall Reward')
ax2.plot(df_grpo_1b['Step'], df_grpo_1b['rewards / format_reward / mean'], color=colors[1], linewidth=1.5, linestyle='--', label='Format Reward (Mean)')
ax2.plot(df_grpo_1b['Step'], df_grpo_1b['rewards / safety_level_reward / mean'], color=colors[2], linewidth=1.2, linestyle='-.', label='Safety Level Reward (Mean)')
ax2.plot(df_grpo_1b['Step'], df_grpo_1b['rewards / group_reward / mean'], color=colors[3], linewidth=1.2, linestyle=':', label='Group Reward (Mean)')
ax2.tick_params(axis='y', labelcolor='black')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True, fontsize=10)

plt.title('GRPO Training Metrics (Llama 3.2 1B Instruct - First 150 Steps)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('assets/grpo_metrics_1b.png', dpi=300, bbox_inches='tight')
plt.close()
