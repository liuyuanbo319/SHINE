import os
import time

import mne
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from dataloader import TSDataset, batch_generator
from loss import loss_function
from models.AutoFormer.AutoFormer import Autoformer_Configs, AutoFormer
from models.Conv1DBased.FCN import FCN
from models.Conv1DBased.InceptionTime import Inception1d
from models.Conv1DBased.MVMS import MyNet6View
from models.Conv1DBased.OSCNN import generate_layer_parameter_list, OS_CNN
from models.DeepShallow.DeepShallow import Deep4Net, ShallowFBCSPNet
from models.InterpGN.InterpGN import InterpGN
from models.LTSM.TimerWrapper import TimerWithHead
from models.LaST.LaST import LaST
from models.LaST.LaST_utils import LaST_Configs
from models.SHINE.SHINE import SHINE
from models.TS2Vec import take_per_row, hierarchical_contrastive_loss, TS2Vec, classifier, torch_pad_nan
from models.TimesNet import TimesNet_Configs, TimesNet
from utils import shift, bf, collate_fn, mask


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    algorithm_name = algorithm_name+'_A'
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm():
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.data_shape = 'BFT'

    def fit(self, train_dataloader,test_dataloader):
        show_step = self.configs.show_step
        args = self.configs
        device = args.device
        max_iter = args.iters

        out = False
        model_save_path = args.model_save_path

        count_iters = 0
        pbar = tqdm(total=args.iters)
        pbar.update(count_iters)
        count_iters = 0
        start_time = time.time()
        # for epoch in range(epochs):
        best_acc = 0.0
        best_auc = 0.0
        count_epochs = 0

        i = j = 0
        while out is False:
            for batch in train_dataloader:
                count_iters += 1
                pbar.update()

                x = batch[0].to(device)
                x_label = batch[1].to(device)
                pred_out = self.model(x)[0]

                loss = nn.functional.cross_entropy(pred_out, x_label)
                # print(f'loss:{loss}')
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                '''iter end'''
                if count_iters % show_step == 0:

                    test_acc, test_auc, _, _, f1, acc, precision, recall = self.evaluate(test_dataloader)

                    if (test_acc > best_acc):
                        best_acc = test_acc
                        i = count_iters
                        pbar.write(f'best acc {test_acc}')
                        torch.save(self.model, model_save_path + '-acc-model.pth')

                    if (test_auc > best_auc):
                        best_auc = test_auc
                        j = count_iters
                if count_iters >= max_iter:
                    out = True
                    break  # break iter

            '''epoch end'''
            count_epochs += 1

            # valid_loss = valid(test_dataloader,model,args)
            # early_stopping(valid_loss)
            # if early_stopping.early_stop:
            #     out = True

            if args.epochs is not None and count_epochs >= args.epochs:
                out = True
        '''while end'''

        torch.cuda.empty_cache()
    def evaluate(self,data_loader):
        self.model.eval()
        device = 'cuda'
        with torch.no_grad():
            test_probs = []
            test_true_labels = []
            test_repres = []
            num_samples = 0
            test_predicts_count = 0
            for batch in data_loader:

                x = batch[0].to(device)
                x_label = batch[1].to(device)
                test_pred_out = self.model(x)[0]
                test_pred_prob = nn.functional.softmax(test_pred_out, dim=1)

                test_predicts_count = torch.eq(torch.argmax(test_pred_prob, dim=1),
                                               x_label.to('cuda')).sum() + test_predicts_count

                num_samples += x.shape[0]
                test_probs.append(test_pred_prob)
                test_true_labels.append(x_label)
                # test_repres.append(test_repre)
            test_acc = test_predicts_count / num_samples
            true_labels = torch.cat(test_true_labels).detach().cpu().numpy()
            pred_probs = torch.cat(test_probs).detach().cpu().numpy()
            pred_labels = torch.argmax(torch.cat(test_probs), dim=1).detach().cpu().numpy()
            if len(np.unique(true_labels)) > 2:
                test_auc = roc_auc_score(true_labels, pred_probs, average='macro', multi_class='ovr')
            else:
                test_auc = roc_auc_score(true_labels, pred_probs[:, 1], average='macro')
            f1 = f1_score(true_labels, pred_labels, average='weighted')
            acc = accuracy_score(true_labels, pred_labels)
            precision = precision_score(y_true=true_labels, y_pred=pred_labels, average='weighted')
            recall = recall_score(y_true=true_labels, y_pred=pred_labels, average='weighted')
            cm = confusion_matrix(true_labels, pred_labels)
            # print(cm)
        self.model.train()
        return test_acc, test_auc, pred_probs, None, f1, acc, precision, recall
    def test(self,args):
        # seed = 2023
        result_path = args.result_path
        dataset_name = args.dataset
        run_name = args.run_name
        folder_path = os.path.join(result_path, dataset_name, run_name)
        model_save_path = args.model_save_path
        # np.random.seed(seed)
        path = args.preprocessed_dir
        if dataset_name == 'ECG200' or dataset_name == 'mit' or dataset_name=='cgm' or dataset_name=='PTBXL':
            if dataset_name=='mit':
                dataset_name='mit-bih-arrhythmia'
            test_data_path = os.path.join(path, dataset_name + '_TEST.npz')

        else:
            test_data_path = os.path.join(path, dataset_name + '_TEST_10s_norm.npz')

        raw_data_test = np.load(test_data_path)
        data_features_test = raw_data_test['data_features'].astype(np.float32)
        data_labels_test = raw_data_test['data_labels']
        if dataset_name == 'mit-bih-arrhythmia':
            data_features_test = data_features_test[:,0:1,:]
        print(data_features_test.mean(), data_features_test.std())
        p = f'{model_save_path}-acc-model.pth'

        self.model = torch.load(p).to('cuda').float()
        d = {
            'ORIGINAL': 'o',
            'NOISE': [0.1, 0.01, 0.001],
            'IN': [0.1, 0.01, 0.001],
            'D': [6, 4],
            'BP': ['1-30', '1-25', '0.5-30'],
            'M':[0.1,0.2,0.3]
        }
        # d = {
        #     'ORIGINAL': 'o',
        #     'NOISE': [0.1],
        # }
        for key, value in d.items():
            for s in value:

                if key == 'NOISE':
                    data_features_test_transfromed = shift(data_features_test, s, IN=False)
                if key == 'IN':
                    data_features_test_transfromed = shift(data_features_test, s, IN=True)
                if key == 'BP':
                    lowf = float(s.split('-')[0])
                    highf = float(s.split('-')[1])
                    data_features_test_transfromed = bf(96, lowf, highf, data_features_test).astype(np.float32)
                if key == 'D':
                    data_features_test_transfromed = data_features_test.round(s)
                if key=='M':
                    data_features_test_transfromed = mask(data_features_test,s)
                if key == 'ORIGINAL':
                    data_features_test_transfromed = data_features_test

                # del data_features_test
                # print(key, s, data_features_test_transfromed.mean(), data_features_test_transfromed.std())

                test_dateset = TSDataset(data_features_test_transfromed, data_labels_test,norm=args.norm,shape=self.data_shape)


                data_loader = self.get_Dataloader(test_dateset)
                # if key=='ORIGINAL':
                #     self.get_repres(data_loader,key)
                #
                # if key=='NOISE' and s==0.1:
                #     self.get_repres(data_loader, key)
                test_acc, test_auc, test_probs, _, f1, acc, precision, recall = self.evaluate(data_loader)
                if p.__contains__('acc'):
                    with open('results\\record.csv', 'a') as record_file:
                        record_file.write(
                            f'{run_name},{dataset_name}_{key}{s},{test_acc:.4f},\n')
                    print('acc', test_acc, acc)
                else:
                    print('auc', test_auc)
                    with open('results\\record.csv', 'a') as record_file:
                        record_file.write(
                            f'{test_auc:.4f}\n')
    def get_Dataloader(self,dateset):
        return DataLoader(dateset, batch_size=64, shuffle=False, drop_last=False)
    def get_repres(self,data_loader,type):


        self.model.eval()
        device = 'cuda'
        with torch.no_grad():
            test_true_labels = []
            test_repres = []
            for batch in data_loader:

                x = batch[0].to(device)
                x_label = batch[1].to(device)
                test_pred_out,test_repre = self.model(x)
                test_repres.append(test_repre)
                test_true_labels.append(x_label)
                # test_repres.append(test_repre)

            true_labels = torch.cat(test_true_labels).detach().cpu().numpy()
            test_repres = torch.cat(test_repres).detach().cpu().numpy()
        np.savez(f'F:\python project\\SHINE\\repres/{self.__class__.__name__}_{type}.npz',repres = test_repres,labels = true_labels)

class OSCNN_A(Algorithm):
    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.data_shape = 'BFT'
        paramenter_number_of_layer_list = [8 * 128 * configs.in_channels, 5 * 128 * 256 + 2 * 256 * 128]
        # paramenter_number_of_layer_list  =  [8 * 128*19, 5 * 128 * 256 + 2 * 256 * 128]
        receptive_field_shape = min(int(configs.ts_len / 4), 89)
        layer_parameter_list = generate_layer_parameter_list(1,
                                                             receptive_field_shape,
                                                             paramenter_number_of_layer_list,
                                                             in_channel=int(configs.in_channels))
        self.model = OS_CNN(layer_parameter_list, configs.num_classes, False).to(configs.device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=configs.lr, betas=(0.9, 0.999))
        self.device = configs.device
    def load_dataset(self,data_loader):
        pass


class TimesNet_A(Algorithm):
    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.data_shape='BTF'
        configs_model = TimesNet_Configs()
        configs_model.seq_len = configs.ts_len
        configs_model.max_len = configs.ts_len
        configs_model.num_class = configs.num_classes
        configs_model.enc_in = configs.in_channels
        self.model = TimesNet(configs_model).to(configs.device)


        self.optim = torch.optim.Adam(self.model.parameters(), lr=configs.lr, betas=(0.9, 0.999))
        self.device = configs.device
    def get_Dataloader(self,dateset):
        return DataLoader(dateset, batch_size=64, shuffle=False, drop_last=False,
                   collate_fn=lambda x: collate_fn(x))
    def load_dataset(self):
        pass
    def fit(self, train_dataloader,test_dataloader):
        show_step = self.configs.show_step
        args = self.configs
        device = args.device
        max_iter = args.iters

        out = False
        model_save_path = args.model_save_path

        count_iters = 0
        pbar = tqdm(total=args.iters)
        pbar.update(count_iters)
        count_iters = 0
        start_time = time.time()
        # for epoch in range(epochs):
        best_acc = 0.0
        best_auc = 0.0
        count_epochs = 0

        i = j = 0
        while out is False:
            for batch in train_dataloader:
                count_iters += 1
                pbar.update()

                x = batch[0].to(device)
                x_label = batch[1].to(device)
                x_mask = batch[2].to(device)
                pred_out = self.model(x, x_mask, None, None)[0]
                loss = nn.functional.cross_entropy(pred_out, x_label)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                '''iter end'''
                if count_iters % show_step == 0:

                    test_acc, test_auc, _, _, f1, acc, precision, recall = self.evaluate(test_dataloader)

                    if (test_acc > best_acc):
                        pbar.write(f'best acc {test_acc}')
                        best_acc = test_acc
                        i = count_iters
                        torch.save(self.model, model_save_path + '-acc-model.pth')

                    if (test_auc > best_auc):
                        best_auc = test_auc
                        j = count_iters
                if count_iters >= max_iter:
                    out = True
                    break  # break iter

            '''epoch end'''
            count_epochs += 1

            # valid_loss = valid(test_dataloader,model,args)
            # early_stopping(valid_loss)
            # if early_stopping.early_stop:
            #     out = True

            if args.epochs is not None and count_epochs >= args.epochs:
                out = True
        '''while end'''

        torch.cuda.empty_cache()


    def evaluate(self,data_loader):
        self.model.eval()
        device = 'cuda'
        with torch.no_grad():
            test_probs = []
            test_true_labels = []
            test_repres = []
            num_samples = 0
            test_predicts_count = 0
            for batch in tqdm(data_loader):
                x = batch[0].to(device)
                x_label = batch[1].to(device)
                x_mask = batch[2].to(device)
                test_pred_out = self.model(x, x_mask, None, None)

                test_pred_prob = nn.functional.softmax(test_pred_out, dim=1)

                test_predicts_count = torch.eq(torch.argmax(test_pred_prob, dim=1),
                                               x_label.to('cuda')).sum() + test_predicts_count

                num_samples += x.shape[0]
                test_probs.append(test_pred_prob)
                test_true_labels.append(x_label)
                # test_repres.append(test_repre)
            test_acc = test_predicts_count / num_samples
            true_labels = torch.cat(test_true_labels).detach().cpu().numpy()
            pred_probs = torch.cat(test_probs).detach().cpu().numpy()
            pred_labels = torch.argmax(torch.cat(test_probs), dim=1).detach().cpu().numpy()
            if len(np.unique(true_labels)) > 2:
                test_auc = roc_auc_score(true_labels, pred_probs, average='macro', multi_class='ovr')
            else:
                test_auc = roc_auc_score(true_labels, pred_probs[:, 1], average='macro')
            f1 = f1_score(true_labels, pred_labels, average='weighted')
            acc = accuracy_score(true_labels, pred_labels)
            precision = precision_score(y_true=true_labels, y_pred=pred_labels, average='weighted')
            recall = recall_score(y_true=true_labels, y_pred=pred_labels, average='weighted')
            cm = confusion_matrix(true_labels, pred_labels)
            # print(cm)
        self.model.train()
        return test_acc, test_auc, pred_probs, None, f1, acc, precision, recall

class SHINE_A(Algorithm):
    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.data_shape = 'BFT'

        self.model = SHINE(configs.ts_len,configs.z_dim,configs.in_channels,configs.num_classes,configs.device,K=configs.K,P=configs.P).to(configs.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=configs.lr, betas=(0.9, 0.999))
        self.device = configs.device
    def load_dataset(self):
        pass
    def fit(self, train_dataloader,test_dataloader):
        args = self.configs
        device = args.device
        max_iter = args.iters

        out = False
        model_save_path = args.model_save_path

        count_iters = 0
        pbar = tqdm(total=args.iters)
        pbar.update(count_iters)
        count_iters = 0
        start_time = time.time()
        # for epoch in range(epochs):
        best_acc = 0.0
        best_auc = 0.0
        count_epochs = 0
        show_step = self.configs.show_step

        while out is False:
            for batch in train_dataloader:
                count_iters += 1
                pbar.update()

                x_input_weak_shifted = batch[0].to(device)
                x_input_strong_shifted = batch[1].to(device)
                x_label = batch[2].to(device)
                x_recon, x_noise_learned, x_learned_trend, x_learned_seasonal, mu_noise, log_var_noise, pred_out = self.model(
                    x_input_weak_shifted, x_input_strong_shifted)

                loss, recon_loss, smooth_loss, noise_loss, pred_loss = loss_function(pred_out,
                                                                                     x_label,
                                                                                     x_recon,
                                                                                     x_input_strong_shifted,
                                                                                     x_learned_trend,
                                                                                     x_learned_seasonal,
                                                                                     mu_noise,
                                                                                     log_var_noise,
                                                                                     args.alpha,
                                                                                     True)


                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if count_iters % show_step == 0:

                    test_acc, test_auc, _, _, f1, acc, precision, recall = self.evaluate(test_dataloader)

                    if (test_acc > best_acc):
                        best_acc = test_acc
                        i = count_iters
                        pbar.write(f'best acc {test_acc}')
                        torch.save(self.model, model_save_path + '-acc-model.pth')

                    if (test_auc > best_auc):
                        best_auc = test_auc
                        j = count_iters
                if count_iters >= max_iter:
                    out = True
                    break  # break iter


                if count_iters >= max_iter:
                    out = True
                    break  # break iter

            '''epoch end'''
            count_epochs += 1

            # valid_loss = valid(test_dataloader,model,args)
            # early_stopping(valid_loss)
            # if early_stopping.early_stop:
            #     out = True

            if args.epochs is not None and count_epochs >= args.epochs:
                out = True
        '''while end'''
        torch.cuda.empty_cache()
    def evaluate(self,data_loader):
        self.model.eval()
        device = 'cuda'
        with torch.no_grad():
            test_probs = []
            test_true_labels = []
            test_repres = []
            num_samples = 0
            test_predicts_count = 0
            for batch in data_loader:


                x = batch[0].to(device)
                x_label = batch[1].to(device)
                test_pred_out = self.model(x,x)[-2]
                test_pred_prob = nn.functional.softmax(test_pred_out, dim=1)

                test_predicts_count = torch.eq(torch.argmax(test_pred_prob, dim=1),
                                               x_label.to('cuda')).sum() + test_predicts_count

                num_samples += x.shape[0]
                test_probs.append(test_pred_prob)
                test_true_labels.append(x_label)
                # test_repres.append(test_repre)
            test_acc = test_predicts_count / num_samples
            true_labels = torch.cat(test_true_labels).detach().cpu().numpy()
            pred_probs = torch.cat(test_probs).detach().cpu().numpy()
            pred_labels = torch.argmax(torch.cat(test_probs), dim=1).detach().cpu().numpy()
            if len(np.unique(true_labels)) > 2:
                test_auc = roc_auc_score(true_labels, pred_probs, average='macro', multi_class='ovr')
            else:
                test_auc = roc_auc_score(true_labels, pred_probs[:, 1], average='macro')
            f1 = f1_score(true_labels, pred_labels, average='weighted')
            acc = accuracy_score(true_labels, pred_labels)
            precision = precision_score(y_true=true_labels, y_pred=pred_labels, average='weighted')
            recall = recall_score(y_true=true_labels, y_pred=pred_labels, average='weighted')
            cm = confusion_matrix(true_labels, pred_labels)
            # print(cm)
        self.model.train()
        return test_acc, test_auc, pred_probs, None, f1, acc, precision, recall

    def get_repres(self, data_loader, type):

        self.model.eval()
        device = 'cuda'
        with torch.no_grad():
            test_true_labels = []
            test_repres = []
            for batch in data_loader:
                x = batch[0].to(device)
                x_label = batch[1].to(device)
                test_repre = self.model(x, x)[-1]
                test_repres.append(test_repre)
                test_true_labels.append(x_label)
                # test_repres.append(test_repre)

            true_labels = torch.cat(test_true_labels).detach().cpu().numpy()
            test_repres = torch.cat(test_repres).detach().cpu().numpy()
        np.savez(f'F:\python project\\SHINE\\repres/{self.__class__.__name__}_{type}.npz', repres=test_repres,
                 labels=true_labels)

class TS2Vec_A(Algorithm):
    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.data_shape = 'BTF'
        self.model = TS2Vec(configs.in_channels).to(configs.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=configs.lr, betas=(0.9, 0.999))
        self.device = configs.device
    def fit(self, train_dataloader):
        args = self.configs
        device = args.device
        max_iter = args.iters

        out = False
        model_save_path = args.model_save_path

        count_iters = 0
        pbar = tqdm(total=args.iters)
        pbar.update(count_iters)
        count_iters = 0
        start_time = time.time()
        # for epoch in range(epochs):
        best_acc = 0.0
        best_auc = 0.0
        count_epochs = 0

        i = j = 0
        while out is False:
            for batch in train_dataloader:
                count_iters += 1
                pbar.update()


                x = batch[0].to(device)

                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 , high=ts_l + 1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                out1 = self.model(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]

                out2 = self.model(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]

                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=0
                )


                self.optim.zero_grad()
                loss.backward()
                self.optim.step()



                if count_iters >= max_iter:
                    out = True
                    break  # break iter

            '''epoch end'''
            count_epochs += 1

            # valid_loss = valid(test_dataloader,model,args)
            # early_stopping(valid_loss)
            # if early_stopping.early_stop:
            #     out = True

            if args.epochs is not None and count_epochs >= args.epochs:
                out = True
        '''while end'''

        torch.save(self.model, model_save_path + '-acc-model.pth')
        torch.cuda.empty_cache()

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.model(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]

        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)

        else:
            if slicing is not None:
                out = out[:, slicing]

        return out.cpu()
    def encode(self, data, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0,
               batch_size=None):
        ''' Compute representations using the model.

        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.

        Returns:
            repr: The representations for data.
        '''
        assert self.model is not None, 'please train or load a net first'
        assert data.ndim == 3
        batch_size = 128
        n_samples, ts_l, _ = data.shape

        org_training = self.model.training
        self.model.eval()

        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0): min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0

                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size=out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)

                output.append(out)

            output = torch.cat(output, dim=0)

        self.model.train(org_training)
        return output.numpy()

    def evaluate(self,train_data,test_data,train_labels,test_labels):
        train_repr = self.encode(train_data, encoding_window='full_series')
        print('encoding test samples')
        test_repr = self.encode(test_data, encoding_window='full_series')


        print('encoding done')

        self.classifier = classifier(320, len(np.unique(train_labels))).to(device='cuda')
        batchsize = 128

        train_dataset = TensorDataset(torch.from_numpy(train_repr).to(torch.float),
                                      torch.from_numpy(train_labels).to(torch.long))
        test_dataset = TensorDataset(torch.from_numpy(test_repr).to(torch.float),
                                     torch.from_numpy(test_labels).to(torch.long))
        train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, drop_last=False)
        device = 'cuda'

        learning_rate = 0.1
        max_iter = 10000
        # show_step = args.show_step
        pbar = tqdm(total=max_iter)
        optim = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        count_iters = 0
        out = False
        best_acc = 0
        best_auc = 0
        while out is False:
            for x, y in train_dataloader:
                count_iters += 1
                pbar.update()

                # x_input_shifted = x_input.to(device)
                x = x.to(device)
                y = y.to(device)

                pred_out = self.classifier(x)

                pred_loss = F.cross_entropy(pred_out, y)
                optim.zero_grad()

                pred_loss.backward()
                optim.step()
                # optim_classification.step()
                '''iter end'''
                if count_iters % 100 == 0:

                    test_acc, test_auc, test_probs, f1, acc, precision, recall = self.eval_classifier(self.classifier, test_dataloader)

                    if (test_acc > best_acc):
                        best_acc = test_acc
                        i = count_iters
                        # np.save(probs_name, test_probs)
                        torch.save(self.classifier, self.configs.model_save_path + '-acc-model-classifier.pth')
                    if (test_auc > best_auc):
                        best_auc = test_auc
                        j = count_iters
                    #     torch.save(model, model_save_path + '-auc-model.pth')
                    print(
                        f'acc: {best_acc} ,acc: {acc:.3f}, F1: {f1:.3f},precision: {precision:.3f},recall:{recall:.3f}')

                if count_iters >= max_iter:
                    out = True
                    print(count_iters)
                    break  # break iter

            '''epoch end'''
        return [], {'acc': best_acc, 'auc': best_auc}

    def eval_classifier(self,model, data_loader):
        model.eval()
        with torch.no_grad():
            test_probs = []
            test_true_labels = []
            num_samples = 0
            test_predicts_count = 0
            for x_input_test, x_label_test in data_loader:
                test_pred_out = model(
                    x_input_test.to('cuda'))
                test_pred_prob = nn.functional.softmax(test_pred_out, dim=1)

                test_predicts_count = torch.eq(torch.argmax(test_pred_prob, dim=1),
                                               x_label_test.to('cuda')).sum() + test_predicts_count

                num_samples += x_input_test.shape[0]
                test_probs.append(test_pred_prob)
                test_true_labels.append(x_label_test)
            test_acc = test_predicts_count / num_samples
            true_labels = torch.cat(test_true_labels).detach().cpu().numpy()
            pred_probs = torch.cat(test_probs).detach().cpu().numpy()
            pred_labels = torch.argmax(torch.cat(test_probs), dim=1).detach().cpu().numpy()
            if len(np.unique(true_labels)) > 2:
                test_auc = roc_auc_score(true_labels, pred_probs, average='macro', multi_class='ovr')
            else:
                test_auc = roc_auc_score(true_labels, pred_probs[:, 1], average='macro')
            # test_auc = 0.0
            f1 = f1_score(true_labels, pred_labels, average='weighted')
            acc = accuracy_score(true_labels, pred_labels)
            precision = precision_score(
                y_true=true_labels, y_pred=pred_labels, average='weighted')
            recall = recall_score(
                y_true=true_labels, y_pred=pred_labels, average='weighted')
        model.train()

        return test_acc, test_auc, pred_probs, f1, acc, precision, recall
    def test(self,args):
        # seed = 2023
        result_path = args.result_path
        dataset_name = args.dataset
        run_name = args.run_name
        folder_path = os.path.join(result_path, dataset_name, run_name)
        model_save_path = args.model_save_path
        # np.random.seed(seed)
        path = args.preprocessed_dir
        if dataset_name == 'ECG200' or dataset_name == 'mit-bih-arrhythmia' or dataset_name=='cgm' or dataset_name=='PTBXL':
            test_data_path = os.path.join(path, dataset_name + '_TEST.npz')
            norm = False
        else:
            test_data_path = os.path.join(path, dataset_name + '_TEST_10s.npz')
            norm = False
        raw_data_test = np.load(test_data_path)
        data_features_test = raw_data_test['data_features'].astype(np.float32)
        data_labels_test = raw_data_test['data_labels']
        transform = {}
        labels = np.unique(data_labels_test)
        for i, l in enumerate(labels):
            transform[l] = i
        data_labels_test = np.vectorize(transform.get)(data_labels_test).astype(np.int64)
        if dataset_name == 'mit-bih-arrhythmia':
            data_features_test = data_features_test[:, 0:1, :]
        if dataset_name == 'ECG200':
            data_features_test = data_features_test[:, np.newaxis, :]
        print(data_features_test.mean(), data_features_test.std())
        p = f'{model_save_path}-acc-model.pth'
        p_c = f'{model_save_path}-acc-model-classifier.pth'
        self.model = torch.load(p).to('cuda').float()
        self.classifier = torch.load(p_c).to('cuda').float()
        if not self.classifier:
            self.model = torch.load(p).to('cuda').float()
            self.classifier = torch.load(p_c).to('cuda').float()
        else:
            print('===============using trained model===================')
        d = {
            'ORIGINAL': 'o',
            'NOISE': [0.5,0.1, 0.01, 0.001],
            'IN': [0.1, 0.01, 0.001],
            'D': [6, 4],
            'BP': ['1-30', '1-25', '0.5-30'],
            'M':[0.1,0.2,0.3]
        }
        norm = True
        for key, value in d.items():
            for s in value:

                if key == 'NOISE':
                    data_features_test_transfromed = shift(data_features_test, s, IN=False)
                if key == 'IN':
                    data_features_test_transfromed = shift(data_features_test, s, IN=True)
                if key == 'BP':
                    lowf = float(s.split('-')[0])
                    highf = float(s.split('-')[1])
                    data_features_test_transfromed = bf(96, lowf, highf, data_features_test).astype(np.float32)
                if key == 'D':
                    data_features_test_transfromed = data_features_test.round(s)
                if key=='M':
                    data_features_test_transfromed = mask(data_features_test,s)
                if key == 'ORIGINAL':
                    data_features_test_transfromed = data_features_test

                if norm:
                    print('================normalizing data=================')
                    scaler = StandardScaler()
                    for batch_data in batch_generator(data_features_test_transfromed, 1000):
                        orig_shape = batch_data.shape
                        batch_data = np.reshape(batch_data.transpose(0, 2, 1), (-1, orig_shape[1]))
                        scaler.partial_fit(batch_data)
                    orig_shape = data_features_test_transfromed.shape
                    data_features_test_transfromed = np.reshape(data_features_test_transfromed.transpose(0, 2, 1), (-1, orig_shape[1]))
                    data_features_test_transfromed = scaler.transform(data_features_test_transfromed).astype(np.float32)
                    data_features_test_transfromed.shape = (orig_shape[0], orig_shape[2], orig_shape[1])
                    data_features_test_transfromed = data_features_test_transfromed.transpose(0, 2, 1)
                if np.ndim(data_features_test_transfromed)==2:
                    data_features_test_transfromed = data_features_test_transfromed[:, np.newaxis, :]
                if self.data_shape == 'BTF':
                    data_features_test_transfromed = data_features_test_transfromed.transpose(0, 2, 1)
                print(key, s, data_features_test_transfromed.mean(), data_features_test_transfromed.std())

                test_repr = self.encode(torch.from_numpy(data_features_test_transfromed), encoding_window='full_series')
                # test_repr = self.encode(data_features_test_transfromed, encoding_window='full_series')
                test_dataset = TensorDataset(torch.from_numpy(test_repr).to(torch.float),
                                             torch.from_numpy(data_labels_test).to(torch.long))

                data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)



                test_acc, test_auc, pred_probs, f1, acc, precision, recall = self.eval_classifier(self.classifier,data_loader)
                if p.__contains__('acc'):
                    with open('results\\record.csv', 'a') as record_file:
                        record_file.write(
                            f'{run_name},{dataset_name}_{key}{s},{test_acc:.4f},\n')
                    print('acc', test_acc, acc)
                else:
                    print('auc', test_auc)
                    with open('results\\record.csv', 'a') as record_file:
                        record_file.write(
                            f'{test_auc:.4f}\n')
class InceptionTime_A(Algorithm):
    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.data_shape = 'BFT'
        self.model = Inception1d(configs.num_classes,configs.in_channels).to(configs.device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=configs.lr, betas=(0.9, 0.999))
        self.device = configs.device
    def load_dataset(self):
        pass

class MVMS_A(Algorithm):
    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.data_shape = 'BFT'
        self.model = MyNet6View(configs.in_channels,configs.ts_len,configs.num_classes).to(configs.device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=configs.lr, betas=(0.9, 0.999))
        self.device = configs.device
    def load_dataset(self):
        pass

class FCN_A(Algorithm):
    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.data_shape = 'BFT'
        self.model = FCN(configs.in_channels,configs.ts_len,configs.num_classes).to(configs.device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=configs.lr, betas=(0.9, 0.999))
        self.device = configs.device
    def load_dataset(self):
        pass

class Deep4Net_A(Algorithm):
    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.data_shape = 'BFT'
        self.model = Deep4Net(configs.in_channels,configs.num_classes,configs.ts_len,final_conv_length='auto').to(configs.device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=configs.lr, betas=(0.9, 0.999))
        self.device = configs.device
    def load_dataset(self):
        pass

class Shallow_A(Algorithm):
    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.data_shape = 'BFT'
        self.model = ShallowFBCSPNet(configs.in_channels,configs.num_classes,configs.ts_len,final_conv_length='auto').to(configs.device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=configs.lr, betas=(0.9, 0.999))
        self.device = configs.device
    def load_dataset(self):
        pass

class InterpGN_A(Algorithm):
    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.data_shape = 'BFT'
        self.model = InterpGN(configs).to(configs.device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=configs.lr, betas=(0.9, 0.999))
        self.device = configs.device
    def compute_beta(self,epoch, max_epoch, schedule='cosine'):
        if schedule == 'cosine':
            beta = 1/2 * (1 + np.cos(np.pi*epoch/max_epoch))
        elif schedule == 'linear':
            beta = 1 - epoch/max_epoch
        else:
            beta = 1
        return beta
    def fit(self, train_dataloader,test_dataloader):
        show_step = self.configs.show_step
        args = self.configs
        device = args.device
        max_iter = args.iters

        out = False
        model_save_path = args.model_save_path

        count_iters = 0
        pbar = tqdm(total=args.iters)
        pbar.update(count_iters)
        count_iters = 0
        start_time = time.time()
        # for epoch in range(epochs):
        best_acc = 0.0
        best_auc = 0.0
        count_epochs = 0
        train_epochs = args.iters//len(train_dataloader)
        i = j = 0
        while out is False:

            for batch in train_dataloader:
                count_iters += 1
                pbar.update()
                x = batch[0].to(device)
                x_label = batch[1].to(device)

                logits, model_info = self.model(x)
                loss = nn.functional.cross_entropy(logits, x_label) + model_info.loss.mean()
                beta = self.compute_beta(count_epochs, train_epochs, 'constant')
                loss += beta * nn.functional.cross_entropy(model_info.shapelet_preds, x_label)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                '''iter end'''
                if count_iters % show_step == 0:

                    test_acc, test_auc, _, _, f1, acc, precision, recall = self.evaluate(test_dataloader)

                    if (test_acc > best_acc):
                        best_acc = test_acc
                        i = count_iters
                        pbar.write(f'best acc {test_acc}')
                        torch.save(self.model, model_save_path + '-acc-model.pth')

                    if (test_auc > best_auc):
                        best_auc = test_auc
                        j = count_iters
                if count_iters >= max_iter:
                    out = True
                    break  # break iter

            '''epoch end'''
            count_epochs += 1

            # valid_loss = valid(test_dataloader,model,args)
            # early_stopping(valid_loss)
            # if early_stopping.early_stop:
            #     out = True

            if args.epochs is not None and count_epochs >= args.epochs:
                out = True
    def evaluate(self,data_loader):
        self.model.eval()
        device = 'cuda'
        with torch.no_grad():
            test_probs = []
            test_true_labels = []
            test_repres = []
            num_samples = 0
            test_predicts_count = 0
            for batch in data_loader:

                x = batch[0].to(device)
                x_label = batch[1].to(device)
                test_pred_out = self.model(x)[0]
                test_pred_prob = nn.functional.softmax(test_pred_out, dim=1)

                test_predicts_count = torch.eq(torch.argmax(test_pred_prob, dim=1),
                                               x_label.to('cuda')).sum() + test_predicts_count

                num_samples += x.shape[0]
                test_probs.append(test_pred_prob)
                test_true_labels.append(x_label)
                # test_repres.append(test_repre)
            test_acc = test_predicts_count / num_samples
            true_labels = torch.cat(test_true_labels).detach().cpu().numpy()
            pred_probs = torch.cat(test_probs).detach().cpu().numpy()
            pred_labels = torch.argmax(torch.cat(test_probs), dim=1).detach().cpu().numpy()
            if len(np.unique(true_labels)) > 2:
                test_auc = roc_auc_score(true_labels, pred_probs, average='macro', multi_class='ovr')
            else:
                test_auc = roc_auc_score(true_labels, pred_probs[:, 1], average='macro')
            f1 = f1_score(true_labels, pred_labels, average='weighted')
            acc = accuracy_score(true_labels, pred_labels)
            precision = precision_score(y_true=true_labels, y_pred=pred_labels, average='weighted')
            recall = recall_score(y_true=true_labels, y_pred=pred_labels, average='weighted')
            cm = confusion_matrix(true_labels, pred_labels)
            # print(cm)
        self.model.train()
        return test_acc, test_auc, pred_probs, None, f1, acc, precision, recall


class AutoFormer_A(Algorithm):
    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.data_shape = 'BTF'
        configs_model = Autoformer_Configs()
        configs_model.seq_len = configs.ts_len
        configs_model.max_len = configs.ts_len
        configs_model.num_class = configs.num_classes
        configs_model.enc_in = configs.in_channels
        self.model = AutoFormer(configs_model).to(configs.device)


        self.optim = torch.optim.Adam(self.model.parameters(), lr=configs.lr, betas=(0.9, 0.999))
        self.device = configs.device
    def load_dataset(self):
        pass
    def get_Dataloader(self,dateset):
        return DataLoader(dateset, batch_size=64, shuffle=False, drop_last=False,
                          collate_fn=lambda x: collate_fn(x))
    def fit(self, train_dataloader,test_dataloader):
        show_step = self.configs.show_step
        args = self.configs
        device = args.device
        max_iter = args.iters

        out = False
        model_save_path = args.model_save_path

        count_iters = 0
        pbar = tqdm(total=args.iters)
        pbar.update(count_iters)
        count_iters = 0
        start_time = time.time()
        # for epoch in range(epochs):
        best_acc = 0.0
        best_auc = 0.0
        count_epochs = 0

        i = j = 0
        while out is False:
            for batch in train_dataloader:
                count_iters += 1
                pbar.update()

                x = batch[0].to(device)
                x_label = batch[1].to(device)
                x_mask = batch[2].to(device)
                pred_out = self.model(x, x_mask, None, None)
                loss = nn.functional.cross_entropy(pred_out, x_label)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                '''iter end'''
                if count_iters % show_step == 0:

                    test_acc, test_auc, _, _, f1, acc, precision, recall = self.evaluate(test_dataloader)

                    if (test_acc > best_acc):
                        pbar.write(f'best acc {test_acc}')
                        best_acc = test_acc
                        i = count_iters
                        torch.save(self.model, model_save_path + '-acc-model.pth')

                    if (test_auc > best_auc):
                        best_auc = test_auc
                        j = count_iters
                if count_iters >= max_iter:
                    out = True
                    break  # break iter

            '''epoch end'''
            count_epochs += 1

            # valid_loss = valid(test_dataloader,model,args)
            # early_stopping(valid_loss)
            # if early_stopping.early_stop:
            #     out = True

            if args.epochs is not None and count_epochs >= args.epochs:
                out = True
        '''while end'''
        torch.cuda.empty_cache()


    def evaluate(self,data_loader):
        self.model.eval()
        device = 'cuda'
        with torch.no_grad():
            test_probs = []
            test_true_labels = []
            test_repres = []
            num_samples = 0
            test_predicts_count = 0
            for batch in data_loader:
                x = batch[0].to(device)
                x_label = batch[1].to(device)
                x_mask = batch[2].to(device)
                test_pred_out = self.model(x, x_mask, None, None)

                test_pred_prob = nn.functional.softmax(test_pred_out, dim=1)

                test_predicts_count = torch.eq(torch.argmax(test_pred_prob, dim=1),
                                               x_label.to('cuda')).sum() + test_predicts_count

                num_samples += x.shape[0]
                test_probs.append(test_pred_prob)
                test_true_labels.append(x_label)
                # test_repres.append(test_repre)
            test_acc = test_predicts_count / num_samples
            true_labels = torch.cat(test_true_labels).detach().cpu().numpy()
            pred_probs = torch.cat(test_probs).detach().cpu().numpy()
            pred_labels = torch.argmax(torch.cat(test_probs), dim=1).detach().cpu().numpy()
            if len(np.unique(true_labels)) > 2:
                test_auc = roc_auc_score(true_labels, pred_probs, average='macro', multi_class='ovr')
            else:
                test_auc = roc_auc_score(true_labels, pred_probs[:, 1], average='macro')
            f1 = f1_score(true_labels, pred_labels, average='weighted')
            acc = accuracy_score(true_labels, pred_labels)
            precision = precision_score(y_true=true_labels, y_pred=pred_labels, average='weighted')
            recall = recall_score(y_true=true_labels, y_pred=pred_labels, average='weighted')
            cm = confusion_matrix(true_labels, pred_labels)
            # print(cm)
        self.model.train()
        return test_acc, test_auc, pred_probs, None, f1, acc, precision, recall

class LaST_A(Algorithm):
    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.data_shape='BFT'
        model_configs = LaST_Configs()
        self.model = LaST(
            input_len=configs.in_channels,
            output_len=model_configs.pred_len,
            input_dim=configs.ts_len,
            out_dim=configs.ts_len,
            var_num=1,
            latent_dim=model_configs.latent_size,
            num_class= configs.num_classes,
            dropout=model_configs.dropout, device=configs.device).to(configs.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=configs.lr, betas=(0.9, 0.999))
        self.device = configs.device
    def load_dataset(self):
        pass
    def fit(self, train_dataloader,test_dataloader):
        args = self.configs
        device = args.device
        max_iter = args.iters

        out = False
        model_save_path = args.model_save_path

        count_iters = 0
        pbar = tqdm(total=args.iters)
        pbar.update(count_iters)
        count_iters = 0
        start_time = time.time()
        # for epoch in range(epochs):
        best_acc = 0.0
        best_auc = 0.0
        count_epochs = 0
        show_step = self.configs.show_step
        i = j = 0
        while out is False:
            for batch in train_dataloader:
                count_iters += 1
                pbar.update()


                x = batch[0].to(device)
                x_label = batch[1].to(device)
                x_pred, elbo, mlbo, mubo,xs_rec,xt_rec,qz_s,qz_t = self.model(x)

                loss = (- elbo - mlbo + mubo)+nn.functional.cross_entropy(x_pred,x_label)


                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if count_iters % show_step == 0:

                    test_acc, test_auc, _, _, f1, acc, precision, recall = self.evaluate(test_dataloader)
                    pbar.write(f'test acc {test_acc}')
                    if (test_acc > best_acc):
                        best_acc = test_acc
                        i = count_iters
                        pbar.write(f'best acc {test_acc}')
                        torch.save(self.model, model_save_path + '-acc-model.pth')

                    if (test_auc > best_auc):
                        best_auc = test_auc
                        j = count_iters
                if count_iters >= max_iter:
                    out = True
                    break  # break iter


                if count_iters >= max_iter:
                    out = True
                    break  # break iter

            '''epoch end'''
            count_epochs += 1

            # valid_loss = valid(test_dataloader,model,args)
            # early_stopping(valid_loss)
            # if early_stopping.early_stop:
            #     out = True

            if args.epochs is not None and count_epochs >= args.epochs:
                out = True
        '''while end'''
        torch.cuda.empty_cache()
    def evaluate(self,data_loader):
        self.model.eval()
        device = 'cuda'
        with torch.no_grad():
            test_probs = []
            test_true_labels = []
            test_repres = []
            num_samples = 0
            test_predicts_count = 0
            for batch in data_loader:


                x = batch[0].to(device)
                x_label = batch[1].to(device)
                test_pred_out = self.model(x)[0]
                test_pred_prob = nn.functional.softmax(test_pred_out, dim=1)

                test_predicts_count = torch.eq(torch.argmax(test_pred_prob, dim=1),
                                               x_label.to('cuda')).sum() + test_predicts_count

                num_samples += x.shape[0]
                test_probs.append(test_pred_prob)
                test_true_labels.append(x_label)
                # test_repres.append(test_repre)
            test_acc = test_predicts_count / num_samples
            true_labels = torch.cat(test_true_labels).detach().cpu().numpy()
            pred_probs = torch.cat(test_probs).detach().cpu().numpy()
            pred_labels = torch.argmax(torch.cat(test_probs), dim=1).detach().cpu().numpy()
            if len(np.unique(true_labels)) > 2:
                test_auc = roc_auc_score(true_labels, pred_probs, average='macro', multi_class='ovr')
            else:
                test_auc = roc_auc_score(true_labels, pred_probs[:, 1], average='macro')
            f1 = f1_score(true_labels, pred_labels, average='weighted')
            acc = accuracy_score(true_labels, pred_labels)
            precision = precision_score(y_true=true_labels, y_pred=pred_labels, average='weighted')
            recall = recall_score(y_true=true_labels, y_pred=pred_labels, average='weighted')
            cm = confusion_matrix(true_labels, pred_labels)
            # print(cm)
        self.model.train()
        return test_acc, test_auc, pred_probs, None, f1, acc, precision, recall

class Timer_A(Algorithm):
    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.data_shape = 'BFT'
        self.model = TimerWithHead(configs.num_classes,configs.in_channels,configs.ts_len).to(configs.device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=configs.lr, betas=(0.9, 0.999))
        self.device = configs.device
    def load_dataset(self):
        pass