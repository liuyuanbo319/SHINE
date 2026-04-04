import os
import time

import mne
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from analysis import analyze_model_efficiency, analyze_runtime, profile_model_with_hooks, analyze_runtime_fixed
from configs.dataset_configs import get_dataset_class
from dataloader import get_dataset, TSDataset

from models.Algorithm import get_algorithm_class

from utils import bf, sw_shift, strong_shift, shift, collate_fn


def load_dataset(args):
    batch_size = args.batch_size
    model_name = args.model_name
    aug = True
    print('=====================loading dataset=====================================')
    if model_name == 'TS2Vec' or model_name == 'TimesNet' or model_name == 'AutoFormer':  # or model_name=='LaST':
        train_dataset, test_dataset = get_dataset(args.preprocessed_dir, args.dataset, shape='BTF', aug=aug,norm=args.norm)
    else:
        if model_name == 'SHINE':
            aug = False
        train_dataset, test_dataset = get_dataset(args.preprocessed_dir, args.dataset, aug=aug,norm=args.norm)

    if model_name == 'SHINE':
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                      collate_fn=lambda x: sw_shift(x,sw=args.aug_type))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    elif model_name == 'TimesNet' or model_name == 'AutoFormer':
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                      collate_fn=lambda x: collate_fn(x))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                     collate_fn=lambda x: collate_fn(x))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    return train_dataset,test_dataset,train_dataloader,test_dataloader
def train(args):
    # early_stopping = EarlyStopping(patience=10, verbose=True)
    tbx = SummaryWriter('log/' + args.run_name, filename_suffix='sf')


    dataset_config = get_dataset_class(args.dataset)()
    for key, value in vars(dataset_config).items():
        setattr(args, key, value)



    # args.num_feature_channel = 1
    # args.ts_length = 288
    # args.num_classes = 4
    folder_path = os.path.join(args.result_path, args.dataset, args.run_name)
    args.folder_path = folder_path

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    algorithm_class = get_algorithm_class(args.model_name)

    for nr in range(args.num_runs):
        train_dataset, test_dataset, train_dataloader, test_dataloader = load_dataset(args)
        args.model_save_path = os.path.join(folder_path, args.dataset + '_run' + str(nr+1))
        # args.model_save_path = os.path.join(folder_path,  'mit-bih-arrhythmia_run' + str(nr+1))
        m = algorithm_class(args)

        print('fitting')
        print(f'start training:  {args.dataset} {args.model_name} run {nr + 1}')

        if args.model_name=='TS2Vec':
            # m.fit(train_dataloader)
            # m.evaluate(train_dataset.ts_data,test_dataset.ts_data,train_dataset.data_labels.numpy(),test_dataset.data_labels.numpy())

            print(analyze_runtime_fixed(m.model,sample_input=test_dataset.ts_data,train_loader=train_dataloader))
            # del train_dataset
            # del test_dataset
            # del train_dataloader
            # del test_dataloader
            # m.test(args)
        else:
            # print(analyze_model_efficiency(m.model, sample_input=test_dataset.ts_data,train_loader=train_dataloader))
            print(analyze_runtime(m.model, sample_weak=test_dataset.ts_data,sample_strong=test_dataset.ts_data))
            print(analyze_runtime_fixed(m.model, sample_weak=test_dataset.ts_data))
            print(analyze_model_efficiency(m.model, sample_input=test_dataset.ts_data))
            # m.fit(train_dataloader,test_dataloader)
            # m.evaluate(test_dataloader)
            # m.get_repres(test_dataloader)
            # del train_dataset
            # del test_dataset
            # del train_dataloader
            # del test_dataloader
            # m.test(args)
            # m.test_per_session(args)
