import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 加载数据
pred = np.load('results/informer_custom_ftM_sl96_ll48_pl24_dm256_nh8_el2_dl1_df512_atprob_fc5_ebtimeF_dtTrue_mxTrue_cpu_test_0/pred.npy')
true = np.load('results/informer_custom_ftM_sl96_ll48_pl24_dm256_nh8_el2_dl1_df512_atprob_fc5_ebtimeF_dtTrue_mxTrue_cpu_test_0/true.npy')

# 2. 检查数据形状
print("预测值形状 (samples, timesteps, features):", pred.shape)
print("真实值形状:", true.shape)

# 3. 可视化函数（改进点：添加逆标准化处理）
def plot_sample(sample_idx=0, feature_idx=0, inverse_scaler=None):
    """绘制指定样本和特征的预测与真实曲线"""
    plt.figure(figsize=(12, 6))
    
    # 处理数据
    true_vals = true[sample_idx, :, feature_idx]
    pred_vals = pred[sample_idx, :, feature_idx]
    
    # 逆标准化（如果适用）
    if inverse_scaler:
        true_vals = inverse_scaler(true_vals.reshape(-1, 1)).flatten()
        pred_vals = inverse_scaler(pred_vals.reshape(-1, 1)).flatten()
    
    plt.plot(true_vals, 'b-', label='True', linewidth=2)
    plt.plot(pred_vals, 'r--', label='Predicted', linewidth=1.5)
    plt.title(f"Sample {sample_idx}, Feature {feature_idx}")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # 防止标签重叠
    plt.show()

# 4. 保存结果（改进点：添加时间戳和列名）
def save_results(pred, true):
    """保存预测和真实值到CSV"""
    timestamp = pd.date_range(start='2025-01-01', periods=pred.shape[1], freq='D')  # 示例时间戳
    
    df_pred = pd.DataFrame(pred.squeeze(), 
                         index=timestamp,
                         columns=[f'Pred_Feature_{i}' for i in range(pred.shape[2])])
    
    df_true = pd.DataFrame(true.squeeze(),
                          index=timestamp,
                          columns=[f'True_Feature_{i}' for i in range(true.shape[2])])
    
    df_pred.to_csv('pred_results.csv')
    df_true.to_csv('true_results.csv')
    print("结果已保存为CSV文件")

# 5. 主程序
if __name__ == "__main__":
    # 检查数据维度
    if pred.shape != true.shape:
        print("警告：预测值和真实值形状不匹配！")
    
    # 示例：绘制第一个样本的第一个特征
    plot_sample(sample_idx=0, feature_idx=0)
    
    # 如果是多变量数据
    if pred.ndim == 3 and pred.shape[2] > 1:
        n_features = pred.shape[2]
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features))
        
        for i in range(n_features):
            axes[i].plot(true[0, :, i], 'b-', label='True')
            axes[i].plot(pred[0, :, i], 'r--', label='Predicted')
            axes[i].set_title(f"Feature {i}")
            axes[i].grid(True)
            if i == 0:
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig('multi_feature_comparison.png', dpi=300)
        plt.show()
    
    # 保存结果
    save_results(pred, true)