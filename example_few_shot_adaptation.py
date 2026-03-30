#!/usr/bin/env python
"""
示例脚本：Few-shot + Transductive Domain Adaptation 测试
使用单个被试验证 Few-shot 适配的效果

使用方法：
  python example_few_shot_adaptation.py --subject 1 --adapt_ratio 0.1
"""

import argparse
import subprocess
import os
import pickle
import numpy as np
from pathlib import Path


def run_baseline_test(subject_id, dataset='mcs', epochs=25, seed=0):
    """
    运行基准测试 (无 Few-shot 适配)
    """
    print(f"\n{'='*70}")
    print(f"[BASELINE TEST] Subject {subject_id} - No Adaptation")
    print(f"{'='*70}\n")
    
    cmd = [
        'python', 'run_CNN_EMG.py',
        '--dataset', dataset,
        '--domain_generalization', 'CORAL',
        '--leave_one_subject_out', 'True',
        '--leftout_subject', str(subject_id),
        '--adapt_ratio', '0.0',
        '--lambda_coral', '0.01',
        '--lambda_prototype', '1.0',
        '--epochs', str(epochs),
        '--seed', str(seed),
    ]
    
    result = subprocess.run(cmd, cwd='.')
    return result.returncode == 0


def run_adapted_test(subject_id, adapt_ratio=0.1, dataset='mcs', 
                     epochs=25, seed=0):
    """
    运行 Few-shot 适配测试
    """
    print(f"\n{'='*70}")
    print(f"[ADAPTED TEST] Subject {subject_id} - Adaptation Ratio {adapt_ratio}")
    print(f"{'='*70}\n")
    
    cmd = [
        'python', 'run_CNN_EMG.py',
        '--dataset', dataset,
        '--domain_generalization', 'CORAL',
        '--leave_one_subject_out', 'True',
        '--leftout_subject', str(subject_id),
        '--adapt_ratio', str(adapt_ratio),
        '--lambda_coral', '0.01',
        '--lambda_prototype', '1.0',
        '--epochs', str(epochs),
        '--seed', str(seed),
    ]
    
    result = subprocess.run(cmd, cwd='.')
    return result.returncode == 0


def load_results(subject_id, adapted=False):
    """
    加载测试结果
    """
    if adapted:
        pkl_path = f'result_subject_{subject_id}_adapt/adaptation_results.pkl'
    else:
        pkl_path = f'result_subject_{subject_id}/test_results.pkl'
    
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"警告: 结果文件不存在 {pkl_path}")
        return None


def compare_results(subject_id):
    """
    对比基准和适配结果
    """
    print(f"\n{'='*70}")
    print(f"[对比分析] Subject {subject_id}")
    print(f"{'='*70}\n")
    
    baseline = load_results(subject_id, adapted=False)
    adapted = load_results(subject_id, adapted=True)
    
    if baseline is None or adapted is None:
        print("错误: 无法加载结果文件")
        return None
    
    # 基本准确率
    baseline_acc = baseline['accuracy']
    adapted_acc = adapted['accuracy']
    improvement = (adapted_acc - baseline_acc) * 100
    
    print(f"基准准确率 (无适配):  {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"适配准确率 (adapt):   {adapted_acc:.4f} ({adapted_acc*100:.2f}%)")
    print(f"相对改进:           {improvement:+.2f}%")
    
    # F1 分数对比
    from sklearn.metrics import f1_score
    baseline_f1 = f1_score(baseline['labels'], baseline['predictions'], average='macro')
    adapted_f1 = f1_score(adapted['labels'], adapted['predictions'], average='macro')
    
    print(f"\n基准 F1 分数 (macro):  {baseline_f1:.4f}")
    print(f"适配 F1 分数 (macro):  {adapted_f1:.4f}")
    print(f"F1 改进:             {(adapted_f1-baseline_f1)*100:+.2f}%")
    
    # 支持集信息
    print(f"\n支持集信息:")
    print(f"  适配比例: {adapted['adapt_ratio']}")
    print(f"  支持集大小: {adapted['support_size']*2} " +
          f"(validation: {adapted['support_size']}, test: {adapted['support_size']})")
    
    # 混淆矩阵对角线和
    baseline_cm_diag = np.trace(baseline['confusion_matrix'])
    adapted_cm_diag = np.trace(adapted['confusion_matrix'])
    
    print(f"\n混淆矩阵对角线和:")
    print(f"  基准: {baseline_cm_diag} / {len(baseline['labels'])}")
    print(f"  适配: {adapted_cm_diag} / {len(adapted['labels'])}")
    
    # 统计分布对比
    print(f"\n特征分布统计:")
    print(f"  训练集 Mean norm: {np.linalg.norm(adapted['mean_train']):.4f}")
    print(f"  Subject 0 Mean norm: {np.linalg.norm(adapted['mean_0']):.4f}")
    print(f"  训练集 Var mean: {adapted['var_train'].mean():.6f}")
    print(f"  Subject 0 Var mean: {adapted['var_0'].mean():.6f}")
    
    return {
        'subject_id': subject_id,
        'baseline_acc': baseline_acc,
        'adapted_acc': adapted_acc,
        'improvement': improvement,
        'baseline_f1': baseline_f1,
        'adapted_f1': adapted_f1,
    }


def batch_test(subjects, adapt_ratios=[0.0, 0.1, 0.2], dataset='mcs'):
    """
    批量测试多个被试和多个适配比例
    """
    results_summary = []
    
    for subject_id in subjects:
        subj_results = {}
        print(f"\n\n{'#'*70}")
        print(f"# 被试 {subject_id}")
        print(f"{'#'*70}")
        
        for adapt_ratio in adapt_ratios:
            print(f"\n>>> 适配比例: {adapt_ratio}")
            
            cmd = [
                'python', 'run_CNN_EMG.py',
                '--dataset', dataset,
                '--domain_generalization', 'CORAL',
                '--leave_one_subject_out', 'True',
                '--leftout_subject', str(subject_id),
                '--adapt_ratio', str(adapt_ratio),
                '--lambda_coral', '0.01',
                '--lambda_prototype', '1.0',
                '--epochs', '25',
            ]
            
            result = subprocess.run(cmd, cwd='.')
            
            if result.returncode == 0:
                # 加载结果
                if adapt_ratio == 0.0:
                    pkl_path = f'result_subject_{subject_id}/test_results.pkl'
                else:
                    pkl_path = f'result_subject_{subject_id}_adapt/adaptation_results.pkl'
                
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        res = pickle.load(f)
                    acc = res['accuracy']
                    subj_results[adapt_ratio] = acc
                    print(f"    准确率: {acc:.4f}")
        
        if subj_results:
            results_summary.append((subject_id, subj_results))
    
    # 打印总结
    print(f"\n\n{'='*70}")
    print("批量测试总结")
    print(f"{'='*70}\n")
    
    for subject_id, accs in results_summary:
        print(f"Subject {subject_id}:")
        for ratio, acc in sorted(accs.items()):
            print(f"  adapt_ratio={ratio:.2f}: {acc:.4f}")


def generate_comparison_plot(subject_id, output_file='comparison.png'):
    """
    生成对比图表
    """
    import matplotlib.pyplot as plt
    
    baseline = load_results(subject_id, adapted=False)
    adapted = load_results(subject_id, adapted=True)
    
    if baseline is None or adapted is None:
        print("无法生成图表: 结果文件缺失")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 混淆矩阵对比
    axes[0].imshow(baseline['confusion_matrix'], cmap='Blues', aspect='auto')
    axes[0].set_title(f'Baseline\nAcc: {baseline["accuracy"]:.4f}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    axes[1].imshow(adapted['confusion_matrix'], cmap='Greens', aspect='auto')
    axes[1].set_title(f'Adapted (ratio={adapted["adapt_ratio"]})\nAcc: {adapted["accuracy"]:.4f}')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    # 准确率对比条形图
    adapt_ratios = [0.0, adapted['adapt_ratio']]
    accuracies = [baseline['accuracy'], adapted['accuracy']]
    colors = ['#FF6B6B', '#51CF66']
    
    axes[2].bar(range(len(adapt_ratios)), accuracies, color=colors, alpha=0.7)
    axes[2].set_ylabel('Accuracy')
    axes[2].set_xlabel('Adaptation Ratio')
    axes[2].set_xticks(range(len(adapt_ratios)))
    axes[2].set_xticklabels([f'{r:.2f}' for r in adapt_ratios])
    axes[2].set_ylim([0.5, 1.0])
    axes[2].set_title('Accuracy Comparison')
    axes[2].grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (ratio, acc) in enumerate(zip(adapt_ratios, accuracies)):
        axes[2].text(i, acc+0.01, f'{acc:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Few-shot + Transductive Domain Adaptation 示例'
    )
    parser.add_argument('--subject', type=int, default=1,
                        help='被试 ID (1-10)')
    parser.add_argument('--adapt_ratio', type=float, default=0.1,
                        help='Few-shot 适配比例 (0.0-1.0)')
    parser.add_argument('--dataset', type=str, default='mcs',
                        help='数据集 (mcs, ninapro-db5, etc.)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='训练轮数')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'compare', 'batch'],
                        help='运行模式')
    parser.add_argument('--seed', type=int, default=0,
                        help='随机种子')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # 单个被试, 单个适配比例
        if args.adapt_ratio == 0.0:
            success = run_baseline_test(args.subject, args.dataset, args.epochs, args.seed)
        else:
            success = run_adapted_test(args.subject, args.adapt_ratio, 
                                       args.dataset, args.epochs, args.seed)
        
        if success:
            print(f"\n✓ 测试完成！结果已保存。")
    
    elif args.mode == 'compare':
        # 对同一被试的基准和适配结果进行对比
        print("运行基准测试...")
        run_baseline_test(args.subject, args.dataset, args.epochs, args.seed)
        
        print("\n\n运行适配测试...")
        run_adapted_test(args.subject, args.adapt_ratio, args.dataset, 
                        args.epochs, args.seed)
        
        print("\n\n进行结果对比...")
        compare_results(args.subject)
        
        # 生成图表
        try:
            generate_comparison_plot(args.subject)
        except Exception as e:
            print(f"图表生成失败: {e}")
    
    elif args.mode == 'batch':
        # 批量测试多个被试
        subjects = list(range(1, 11))  # 被试 1-10
        adapt_ratios = [0.0, 0.05, 0.1, 0.2]
        batch_test(subjects, adapt_ratios, args.dataset)


if __name__ == '__main__':
    main()
