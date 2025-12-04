import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_speedup(input_csv: Path, output_pdf: str, ylim: tuple, yticks: np.ndarray):
    """生成 speedup 垂直条形图"""

    # 读取数据
    df = pd.read_csv(input_csv)

    # 清理数据：去除空格
    df.columns = df.columns.str.strip()
    df['Case Name'] = df['Case Name'].str.strip()

    # 按 speedup 降序排列（最大值在最右边）
    df_sorted = df.sort_values('speedup', ascending=False)

    # 全局绘图风格设置
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams['font.family'] = 'sans-serif'

    # 创建画布
    fig, ax = plt.subplots(figsize=(14, 5))

    # 定义颜色
    bar_color = '#6A8DC2'
    edge_color = 'black'
    error_bar_color = 'black'

    # 绘制垂直条形图
    x_positions = np.arange(len(df_sorted)) * 2
    bars = ax.bar(
        x=x_positions,
        height=df_sorted['speedup'],
        yerr=df_sorted['speedup_std'],
        color=bar_color,
        edgecolor=edge_color,
        linewidth=0.5,
        width=1.4,
        capsize=3,
        error_kw={'ecolor': error_bar_color, 'elinewidth': 1.0, 'capthick': 1.0},
        label='Mean Speedup'
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df_sorted['Case Name'], rotation=30, ha='right')

    # 添加图例的误差条代理
    error_line = ax.errorbar([x_positions[-1] + 10], [1.5], yerr=[0.05], fmt='none',
                              ecolor='black', capsize=3, elinewidth=1.0, label='Standard Deviation')

    # 添加基准线
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, zorder=1)

    # 设置X轴范围
    ax.set_xlim(-1, x_positions[-1] + 1)

    # 坐标轴设置
    ax.set_ylabel('Speedup', fontsize=32, labelpad=10)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)

    ax.tick_params(axis='x', pad=2, labelsize=20, labelcolor='0.2')
    ax.tick_params(axis='y', pad=0, labelsize=26)

    # 网格线
    ax.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.7, zorder=0)
    ax.grid(axis='x', visible=False)

    # 边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    # 图例
    ax.legend(loc='upper right', fontsize=22, frameon=True, labelcolor='0')

    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_pdf}")


def main():
    script_dir = Path(__file__).resolve().parent

    # 生成 unroll speedup 图
    plot_speedup(
        input_csv=script_dir / 'unroll_stats.csv',
        output_pdf='unroll_speedup.pdf',
        ylim=(0.8, 1.7),
        yticks=np.arange(0.8, 1.8, 0.2)
    )

    # 生成 mask speedup 图
    plot_speedup(
        input_csv=script_dir / 'mask_stats.csv',
        output_pdf='mask_speedup.pdf',
        ylim=(0.8, 3.6),
        yticks=np.arange(1.0, 3.7, 0.5)
    )


if __name__ == "__main__":
    main()
