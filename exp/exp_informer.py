from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import pandas as pd
import os
import time
import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }
        if self.args.model in model_dict:
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            model = model_dict[self.args.model](
                enc_in=self.args.enc_in,
                dec_in=self.args.dec_in, 
                c_out=self.args.c_out, 
                seq_len=self.args.seq_len, 
                label_len=self.args.label_len,
                pred_len=self.args.pred_len, 
                factor=self.args.factor,
                d_model=self.args.d_model, 
                n_heads=self.args.n_heads, 
                e_layers=e_layers,
                d_layers=self.args.d_layers, 
                d_ff=self.args.d_ff,
                dropout=self.args.dropout, 
                attn=self.args.attn,
                embed=self.args.embed,
                freq=self.args.freq,
                activation=self.args.activation,
                output_attention=self.args.output_attention,
                distil=self.args.distil,
                mix=self.args.mix,
                device=self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args
        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'custom': Dataset_Custom,
        }
        Data = data_dict.get(self.args.data, Dataset_Custom)
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=args.freq,
            cols=args.cols
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        print(f"{flag} 数据集加载完成 | 样本数: {len(data_set)} | batch数: {len(data_loader)}")
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.AdamW(
            self.model.parameters(), 
            lr=self.args.learning_rate,
            weight_decay=1e-5
        )

    def _select_criterion(self):
        if self.args.loss == 'huber':
            return nn.HuberLoss(delta=1.0)
        elif self.args.loss == 'mae':
            return nn.L1Loss()
        return nn.MSELoss()

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                pred, true = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss)
        return np.average(total_loss)

    def train(self, setting):
        print(f"开始训练 | 配置: {setting}")
        print(f"  - 训练轮次: {self.args.train_epochs} | batch_size: {self.args.batch_size}")
        print(f"  - 早停 patience: {self.args.patience} | 损失函数: {self.args.loss}")

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_loss = []
            start_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                pred, true = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                loss.backward()
                model_optim.step()
                epoch_loss.append(loss.item())

                # 【新增】每10个batch打印进度
                if i % 100 == 0:
                    print(f"  Epoch {epoch+1} | 已处理 {i+1}/{len(train_loader)} batch | 耗时 {time.time()-start_time:.2f}s")

            train_loss = np.mean(epoch_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            # 【优化】更详细的日志
            print(f"Epoch {epoch+1} 结果:")
            print(f"  - 训练损失: {train_loss:.4f} | 验证损失: {vali_loss:.4f} | 测试损失: {test_loss:.4f}")
            print(f"  - 学习率: {model_optim.param_groups[0]['lr']:.6f}")
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered | 验证集损失不再下降")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        print(f"加载最佳模型: {best_model_path}")
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        
        preds, trues = [], []
        timestamps = []  # 新增：存储时间戳
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                pred, true = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                timestamps.extend(batch_y_mark[:, 0].cpu().numpy())  # 假设时间戳在第一列

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # 逆标准化（如果数据被标准化过）
        if hasattr(test_loader.dataset, 'scaler') and test_loader.dataset.scaler is not None:
            preds = test_loader.dataset.inverse_transform(preds)
            trues = test_loader.dataset.inverse_transform(trues)

        # === 保存结果为CSV（替换原NPY）=== 
        result_path = os.path.join('./results/', setting)
        os.makedirs(result_path, exist_ok=True)
        
        # 1. 保存预测值和真实值
        results_df = pd.DataFrame({
            'Timestamp': timestamps[:len(preds)],  # 对齐长度
            'True': trues.squeeze(),
            'Pred': preds.squeeze()
        })
        results_df.to_csv(os.path.join(result_path, 'results.csv'), index=False)

        # 2. 保存指标
        mae, mse, rmse, mape, mspe = self._calculate_metrics(preds, trues)
        with open(os.path.join(result_path, 'metrics.txt'), 'w') as f:
            f.write(f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%\nMSPE: {mspe:.2f}%")

        print(f"Test Results - MAE: {mae:.4f}, MSE: {mse:.4f}")
        return mae, mse

    def _calculate_metrics(self, pred, true):
        """指标计算（含除零保护）"""
        mask = true != 0
        mae = np.mean(np.abs(pred - true))
        mse = np.mean((pred - true)**2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100 if any(mask) else 0
        mspe = np.mean(np.square((pred[mask] - true[mask]) / true[mask])) * 100 if any(mask) else 0
        return mae, mse, rmse, mape, mspe

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in pred_loader:
                pred, true = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(pred.detach().cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        result_path = os.path.join('./results/', setting)
        os.makedirs(result_path, exist_ok=True)
        np.save(os.path.join(result_path, 'pred.npy'), preds)

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        
        # 解码器输入：兼容多变量（所有特征列）
        dec_inp = torch.zeros(
            batch_y.size(0),
            self.args.label_len + self.args.pred_len,
            batch_x.size(-1),  # 保持特征维度
            device=self.device
        )
        dec_inp[:, :self.args.label_len] = batch_y[:, :self.args.label_len]  # 所有特征
        
        # 前向传播
        outputs = self.model(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark.to(self.device) if batch_x_mark is not None else None,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark.to(self.device) if batch_y_mark is not None else None
        )
        
        f_dim = -1
        return outputs[:, -self.args.pred_len:, f_dim:], batch_y[:, -self.args.pred_len:, f_dim:]