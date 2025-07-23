import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 加载数据
pred = np.load('results/informer_custom_ftS_sl90_ll45_pl30_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_adnoad_ptna_test_1/pred.npy')      # 模型预测值
true = np.load('results/informer_custom_ftS_sl90_ll45_pl30_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_adnoad_ptna_test_1/true.npy')      # 真实值
metrics = np.load('results/informer_custom_ftS_sl90_ll45_pl30_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_adnoad_ptna_test_1/metrics.npy', allow_pickle=True)  # 评估指标


# 2. 检查数据形状
print("预测值形状 (samples, timesteps, features):", pred.shape)
print("真实值形状:", true.shape)
print("评估指标:", metrics)  # 可能是标量、数组或字典

# 3. 可视化单个样本的预测 vs 真实值
def plot_sample(sample_idx=0, feature_idx=0):
    """绘制指定样本和特征的预测与真实曲线"""
    plt.figure(figsize=(12, 6))
    plt.plot(true[sample_idx, :, feature_idx], label='True', linewidth=2)
    plt.plot(pred[sample_idx, :, feature_idx], label='Predicted', linestyle='--')
    plt.title(f"Sample {sample_idx}, Feature {feature_idx}")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# 示例：绘制第0个样本、第0个特征的对比
plot_sample(sample_idx=0, feature_idx=0)

# 4. 保存结果为CSV（可选）
pd.DataFrame(pred.squeeze()).to_csv('pred.csv')
pd.DataFrame(true.squeeze()).to_csv('true.csv')

# 5. 如果是多特征数据，绘制所有特征的子图
if pred.ndim == 3 and pred.shape[2] > 1:
    n_features = pred.shape[2]
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features))
    for i in range(n_features):
        axes[i].plot(true[0, :, i], label='True')
        axes[i].plot(pred[0, :, i], label='Predicted')
        axes[i].set_title(f"Feature {i}")
        axes[i].grid(True)
    plt.tight_layout()
    plt.show()