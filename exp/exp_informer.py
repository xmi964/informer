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
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device,
                getattr(self.args, 'custom_attention', None),  # 自定义注意力类型
                getattr(self.args, 'peak_threshold', 3.0)      # 新增：异常检测阈值
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
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'mspe':
            return lambda pred, true: torch.mean(torch.square((pred - true) / (true + 1e-8)))
        
        elif self.args.loss == 'huber':  # 原有实现保持不变
            def peak_aware_huber(pred, true, peak_mask=None):
                delta = 1.0
                diff = torch.abs(pred - true)
                loss = torch.where(diff < delta,
                                0.5 * diff.pow(2),
                                delta * (diff - 0.5 * delta))
                if peak_mask is not None:
                    loss = loss * (1 + peak_mask.float())
                return loss.mean()
            return peak_aware_huber
        
        # 新增自适应Huber损失
        elif self.args.loss == 'adaptive_huber':
            def adaptive_huber(pred, true, peak_mask=None):
                # 动态delta：基于真实值的幅度调整阈值
                base_delta = 1.0
                delta = base_delta + true.abs() / 100  # 数值越大容忍误差越大
                
                diff = torch.abs(pred - true)
                loss = torch.where(
                    diff < delta,
                    0.5 * diff.pow(2),
                    delta * (diff - 0.5 * delta)
                )
                
                # 保留原有的峰值惩罚逻辑
                if peak_mask is not None:
                    loss = loss * (1 + peak_mask.float())
                    
                return loss.mean()
            return adaptive_huber
        
        else:
            return nn.MSELoss()
    
    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true, peak_mask = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)

            # 适配损失函数参数
            if self.args.loss == 'huber':
                loss = criterion(pred.detach().cpu(), true.detach().cpu(), peak_mask.detach().cpu())
            else:
                loss = criterion(pred.detach().cpu(), true.detach().cpu())

            total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp and self.args.use_gpu:
            scaler = torch.cuda.amp.GradScaler()
        else:
            self.args.use_amp = False

        for epoch in range(self.args.train_epochs):
            print(f"======= Starting Epoch {epoch+1}/{self.args.train_epochs} =======")
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true, peak_mask = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)

                # 适配损失函数参数
                if self.args.loss == 'huber':
                    loss = criterion(pred, true, peak_mask)
                else:
                    loss = criterion(pred, true)  # 普通损失函数不需要 peak_mask

                train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true, _ = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred, true, _ = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # 1. 设备转移
        device = self.device
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # 2. 构造decoder输入
        dec_inp = torch.zeros(batch_y.size(0), self.args.pred_len, batch_y.size(2)).to(device)
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)

        # 3. 从数据中提取 peak_mask（最后一列是 is_peak）
        if batch_x.size(-1) > 1:  # 确保有足够的特征列
            enc_peak_mask = batch_x[:, :, -1].unsqueeze(-1)  # 提取最后一列作为 peak_mask
            dec_peak_mask = dec_inp[:, :, -1].unsqueeze(-1)  # 提取最后一列作为 peak_mask
        else:
            # 如果没有足够的列，创建全零 mask（不关注异常）
            enc_peak_mask = torch.zeros(batch_x.size(0), batch_x.size(1), 1).to(device)
            dec_peak_mask = torch.zeros(dec_inp.size(0), dec_inp.size(1), 1).to(device)

        # 4. 模型前向传播（移除手动生成的 peak_mask）
        outputs = self.model(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
            enc_self_mask=None,
            dec_self_mask=None,
            dec_enc_mask=None,
            enc_peak_mask=enc_peak_mask,  # 使用从数据中提取的 peak_mask
            dec_peak_mask=dec_peak_mask  # 使用从数据中提取的 peak_mask
        )

        # 5. 提取目标值（排除 peak_mask 列）
        f_dim = -2 if self.args.features == 'MS' else 0  # 倒数第二列是目标值（最后一列是 peak_mask）
        true = batch_y[:, -self.args.pred_len:, f_dim:]

        return outputs[:, -self.args.pred_len:, :], true, enc_peak_mask.squeeze(-1)