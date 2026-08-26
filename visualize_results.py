import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.gridspec import GridSpec

# 设置matplotlib参数，显示英文字符
plt.rcParams['font.sans-serif'] = ['Arial']  # Changed from SimHei to Arial
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    """加载CSV数据文件"""
    df = pd.read_csv(file_path)
    return df

def create_output_dir():
    """创建输出目录"""
    output_dir = "./Results/Visualizations"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_dataset_comparison(df, output_dir):
    """比较不同数据集的性能"""
    plt.figure(figsize=(14, 8))
    
    # 计算每个数据集的平均指标
    dataset_metrics = df.groupby('dataset').agg({
        'avg_rouge1': 'mean',
        'avg_rouge2': 'mean',
        'avg_rougeL': 'mean',
        'max_rouge1': 'max',
        'max_rouge2': 'max',
        'max_rougeL': 'max'
    }).reset_index()
    
    # 设置绘图颜色和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 创建柱状图
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. 平均ROUGE得分柱状图
    avg_data = dataset_metrics[['dataset', 'avg_rouge1', 'avg_rouge2', 'avg_rougeL']]
    avg_data = avg_data.melt('dataset', var_name='metric', value_name='score')
    sns.barplot(x='dataset', y='score', hue='metric', data=avg_data, ax=axes[0], palette='viridis')
    axes[0].set_title('Average ROUGE Scores by Dataset', fontsize=16)  # Changed from Chinese to English
    axes[0].set_xlabel('Dataset', fontsize=14)  # Changed from Chinese to English
    axes[0].set_ylabel('ROUGE Score', fontsize=14)  # Changed from Chinese to English
    axes[0].legend(title='Metric', title_fontsize=12)  # Changed from Chinese to English
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. 最大ROUGE得分柱状图
    max_data = dataset_metrics[['dataset', 'max_rouge1', 'max_rouge2', 'max_rougeL']]
    max_data = max_data.melt('dataset', var_name='metric', value_name='score')
    sns.barplot(x='dataset', y='score', hue='metric', data=max_data, ax=axes[1], palette='viridis')
    axes[1].set_title('Maximum ROUGE Scores by Dataset', fontsize=16)  # Changed from Chinese to English
    axes[1].set_xlabel('Dataset', fontsize=14)  # Changed from Chinese to English
    axes[1].set_ylabel('ROUGE Score', fontsize=14)  # Changed from Chinese to English
    axes[1].legend(title='Metric', title_fontsize=12)  # Changed from Chinese to English
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/dataset_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_rag_method_comparison(df, output_dir):
    """比较不同RAG方法的性能"""
    plt.figure(figsize=(14, 8))
    
    # 计算每种RAG方法的平均指标
    method_metrics = df.groupby('rag_method').agg({
        'avg_rouge1': 'mean',
        'avg_rouge2': 'mean',
        'avg_rougeL': 'mean',
        'max_rouge1': 'max',
        'max_rouge2': 'max',
        'max_rougeL': 'max'
    }).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. 平均ROUGE得分柱状图
    avg_data = method_metrics[['rag_method', 'avg_rouge1', 'avg_rouge2', 'avg_rougeL']]
    avg_data = avg_data.melt('rag_method', var_name='metric', value_name='score')
    sns.barplot(x='rag_method', y='score', hue='metric', data=avg_data, ax=axes[0], palette='magma')
    axes[0].set_title('Average ROUGE Scores by RAG Method', fontsize=16)  # Changed from Chinese to English
    axes[0].set_xlabel('RAG Method', fontsize=14)  # Changed from Chinese to English
    axes[0].set_ylabel('ROUGE Score', fontsize=14)  # Changed from Chinese to English
    axes[0].legend(title='Metric', title_fontsize=12)  # Changed from Chinese to English
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. 最大ROUGE得分柱状图
    max_data = method_metrics[['rag_method', 'max_rouge1', 'max_rouge2', 'max_rougeL']]
    max_data = max_data.melt('rag_method', var_name='metric', value_name='score')
    sns.barplot(x='rag_method', y='score', hue='metric', data=max_data, ax=axes[1], palette='magma')
    axes[1].set_title('Maximum ROUGE Scores by RAG Method', fontsize=16)  # Changed from Chinese to English
    axes[1].set_xlabel('RAG Method', fontsize=14)  # Changed from Chinese to English
    axes[1].set_ylabel('ROUGE Score', fontsize=14)  # Changed from Chinese to English
    axes[1].legend(title='Metric', title_fontsize=12)  # Changed from Chinese to English
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/rag_method_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_thief_method_comparison(df, output_dir):
    """比较不同盗窃方法的性能"""
    plt.figure(figsize=(16, 10))
    
    # 计算每种攻击方法的平均指标
    thief_metrics = df.groupby('thief_method').agg({
        'avg_rouge1': 'mean',
        'avg_rouge2': 'mean',
        'avg_rougeL': 'mean',
        'max_rouge1': 'max',
        'max_rouge2': 'max',
        'max_rougeL': 'max',
        'num_chunks': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. 平均ROUGE得分柱状图
    avg_data = thief_metrics[['thief_method', 'avg_rouge1', 'avg_rouge2', 'avg_rougeL']]
    avg_data = avg_data.melt('thief_method', var_name='metric', value_name='score')
    sns.barplot(x='thief_method', y='score', hue='metric', data=avg_data, ax=axes[0, 0], palette='plasma')
    axes[0, 0].set_title('Average ROUGE Scores by Attack Method', fontsize=16)  # Changed from Chinese to English
    axes[0, 0].set_xlabel('Attack Method', fontsize=14)  # Changed from Chinese to English
    axes[0, 0].set_ylabel('ROUGE Score', fontsize=14)  # Changed from Chinese to English
    axes[0, 0].legend(title='Metric', title_fontsize=12)  # Changed from Chinese to English
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. 最大ROUGE得分柱状图
    max_data = thief_metrics[['thief_method', 'max_rouge1', 'max_rouge2', 'max_rougeL']]
    max_data = max_data.melt('thief_method', var_name='metric', value_name='score')
    sns.barplot(x='thief_method', y='score', hue='metric', data=max_data, ax=axes[0, 1], palette='plasma')
    axes[0, 1].set_title('Maximum ROUGE Scores by Attack Method', fontsize=16)  # Changed from Chinese to English
    axes[0, 1].set_xlabel('Attack Method', fontsize=14)  # Changed from Chinese to English
    axes[0, 1].set_ylabel('ROUGE Score', fontsize=14)  # Changed from Chinese to English
    axes[0, 1].legend(title='Metric', title_fontsize=12)  # Changed from Chinese to English
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. 平均块数柱状图
    sns.barplot(x='thief_method', y='num_chunks', data=thief_metrics, ax=axes[1, 0], palette='plasma')
    axes[1, 0].set_title('Average Chunk Count by Attack Method', fontsize=16)  # Changed from Chinese to English
    axes[1, 0].set_xlabel('Attack Method', fontsize=14)  # Changed from Chinese to English
    axes[1, 0].set_ylabel('Average Chunk Count', fontsize=14)  # Changed from Chinese to English
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. ROUGE-L和块数的关系
    sns.scatterplot(x='avg_rougeL', y='num_chunks', hue='thief_method', data=df, 
                   s=100, alpha=0.7, ax=axes[1, 1], palette='plasma')
    axes[1, 1].set_title('ROUGE-L Score vs Chunk Count', fontsize=16)  # Changed from Chinese to English
    axes[1, 1].set_xlabel('Average ROUGE-L Score', fontsize=14)  # Changed from Chinese to English
    axes[1, 1].set_ylabel('Chunk Count', fontsize=14)  # Changed from Chinese to English
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    axes[1, 1].legend(title='Attack Method', title_fontsize=12)  # Changed from Chinese to English

    plt.tight_layout()
    plt.savefig(f"{output_dir}/thief_method_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_dataset_method_matrix(df, output_dir):
    """绘制数据集和方法的矩阵图"""
    # 为每个数据集和攻击方法组合计算平均ROUGE-L分数
    pivot_rougeL = df.pivot_table(
        index='dataset', 
        columns='thief_method', 
        values='avg_rougeL',
        aggfunc='mean'
    )
    
    # 为每个数据集和攻击方法组合计算平均ROUGE-1分数
    pivot_rouge1 = df.pivot_table(
        index='dataset', 
        columns='thief_method', 
        values='avg_rouge1',
        aggfunc='mean'
    )
    
    # 创建热力图
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # ROUGE-L热力图
    sns.heatmap(pivot_rougeL, annot=True, cmap='YlGnBu', fmt=".3f", ax=axes[0])
    axes[0].set_title('Average ROUGE-L Scores by Dataset and Attack Method', fontsize=16)  # Changed from Chinese to English
    axes[0].set_xlabel('Attack Method', fontsize=14)  # Changed from Chinese to English
    axes[0].set_ylabel('Dataset', fontsize=14)  # Changed from Chinese to English
    
    # ROUGE-1热力图
    sns.heatmap(pivot_rouge1, annot=True, cmap='YlGnBu', fmt=".3f", ax=axes[1])
    axes[1].set_title('Average ROUGE-1 Scores by Dataset and Attack Method', fontsize=16)  # Changed from Chinese to English
    axes[1].set_xlabel('Attack Method', fontsize=14)  # Changed from Chinese to English
    axes[1].set_ylabel('Dataset', fontsize=14)  # Changed from Chinese to English
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dataset_method_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_comprehensive_analysis(df, output_dir):
    """综合分析图"""
    # 创建大型图表
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. 最佳组合TOP10
    ax1 = fig.add_subplot(gs[0, 0])
    # 计算每个组合的平均ROUGE-L
    combo_metrics = df.groupby(['dataset', 'rag_method', 'thief_method']).agg({
        'avg_rougeL': 'mean'
    }).reset_index()
    # 获取TOP10组合
    top_combos = combo_metrics.sort_values('avg_rougeL', ascending=False).head(10)
    # 创建组合标签
    top_combos['combo'] = top_combos.apply(lambda x: f"{x['dataset']}\n{x['rag_method']}-{x['thief_method']}", axis=1)
    # 绘制柱状图
    sns.barplot(x='combo', y='avg_rougeL', data=top_combos, palette='viridis', ax=ax1)
    ax1.set_title('Top 10 Combinations by ROUGE-L Score', fontsize=16)  # Changed from Chinese to English
    ax1.set_xlabel('Dataset-RAG Method-Attack Method', fontsize=14)  # Changed from Chinese to English
    ax1.set_ylabel('Average ROUGE-L Score', fontsize=14)  # Changed from Chinese to English
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. 数据集和RAG方法的交互影响
    ax2 = fig.add_subplot(gs[0, 1])
    dataset_rag_metrics = df.groupby(['dataset', 'rag_method']).agg({
        'avg_rougeL': 'mean'
    }).reset_index()
    pivot_dataset_rag = dataset_rag_metrics.pivot(index='dataset', columns='rag_method', values='avg_rougeL')
    sns.heatmap(pivot_dataset_rag, annot=True, cmap='coolwarm', fmt=".3f", ax=ax2)
    ax2.set_title('Dataset and RAG Method Interaction (ROUGE-L)', fontsize=16)  # Changed from Chinese to English
    
    # 3. 方法效率分析 (ROUGE-L/块数)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # 计算每个攻击方法的效率(ROUGE-L/块数)
    df['efficiency'] = df['avg_rougeL'] / df['num_chunks']
    method_efficiency = df.groupby('thief_method').agg({
        'efficiency': 'mean'
    }).reset_index()
    
    # 对效率进行排序
    method_efficiency = method_efficiency.sort_values('efficiency', ascending=False)
    
    sns.barplot(x='thief_method', y='efficiency', data=method_efficiency, palette='magma', ax=ax3)
    ax3.set_title('Attack Method Efficiency (ROUGE-L/Chunk Count)', fontsize=16)  # Changed from Chinese to English
    ax3.set_xlabel('Attack Method', fontsize=14)  # Changed from Chinese to English
    ax3.set_ylabel('Efficiency (ROUGE-L/Chunk Count)', fontsize=14)  # Changed from Chinese to English
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. 散点图：块数与ROUGE得分的关系，通过大小表示ROUGE-2得分
    ax4 = fig.add_subplot(gs[1, 1])
    scatter = ax4.scatter(df['avg_rouge1'], df['avg_rougeL'], 
                        s=df['num_chunks']*3, # 点大小基于块数
                        c=df['avg_rouge2'], # 颜色基于ROUGE-2
                        alpha=0.6, cmap='viridis')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('ROUGE-2 Score', fontsize=12)  # Changed from Chinese to English
    
    # 为部分点添加标签
    top_points = df.sort_values('avg_rouge1', ascending=False).head(5)
    for _, row in top_points.iterrows():
        ax4.annotate(f"{row['dataset']}-{row['thief_method']}", 
                   (row['avg_rouge1'], row['avg_rougeL']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10)
    
    ax4.set_title('ROUGE-1 vs ROUGE-L (Point Size=Chunk Count, Color=ROUGE-2)', fontsize=16)  # Changed from Chinese to English
    ax4.set_xlabel('Average ROUGE-1 Score', fontsize=14)  # Changed from Chinese to English
    ax4.set_ylabel('Average ROUGE-L Score', fontsize=14)  # Changed from Chinese to English
    ax4.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    # 加载数据
    data_file = "./Results/theft_evaluation_summary_combined.csv"
    df = load_data(data_file)
    
    # 创建输出目录
    output_dir = create_output_dir()
    
    # 生成各种可视化图表
    plot_dataset_comparison(df, output_dir)
    plot_rag_method_comparison(df, output_dir)
    plot_thief_method_comparison(df, output_dir)
    plot_dataset_method_matrix(df, output_dir)
    plot_comprehensive_analysis(df, output_dir)
    
    print(f"Visualization charts saved to: {output_dir}")  # Changed from Chinese to English

if __name__ == "__main__":
    main()
