import os

import mne
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from utils import bf, strong_shift


def batch_generator(data, batch_size):
    num_samples = data.shape[0]
    num_batches = num_samples // batch_size
    remainder = num_samples % batch_size

    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        yield data[batch_start:batch_end]

    if remainder != 0:
        yield data[num_batches * batch_size:]


class TSDataset(Dataset):
    def __init__(self, data_features=None, data_labels=None, file_path=None, shape='BFT', norm=False,aug=False) -> None:

        self.shape = shape
        self.aug = aug
        if file_path is not None:
            print(f'loading data from path {file_path}')
            raw_data = np.load(file_path)
            data_features = raw_data['data_features']
            data_labels = raw_data['data_labels']
        if np.ndim(data_features) == 2:
            data_features = data_features[:, np.newaxis, :]
        print(data_features.shape)
        if file_path and file_path.__contains__('mit'):

            data_features = data_features[:, 0:1, :]
            # Move the labels to {0, ..., L-1}
            print(data_features.shape)
        labels = np.unique(data_labels)
        transform = {}
        for i, l in enumerate(labels):
            transform[l] = i

        data_labels = np.vectorize(transform.get)(data_labels).astype(np.int64)

        '''
        use mne scaler to normalize EEG signal along channels
        '''
        # norm = True
        if norm:

            print('================normalizing data=================')
            scaler = StandardScaler()
            for batch_data in batch_generator(data_features, 1000):
                orig_shape = batch_data.shape
                batch_data = np.reshape(batch_data.transpose(0, 2, 1), (-1, orig_shape[1]))
                scaler.partial_fit(batch_data)
            orig_shape = data_features.shape
            data_features = np.reshape(data_features.transpose(0, 2, 1), (-1, orig_shape[1]))
            data_features = scaler.transform(data_features).astype(np.float32)
            data_features.shape = (orig_shape[0], orig_shape[2], orig_shape[1])
            data_features = data_features.transpose(0, 2, 1)
            # np.savez(file_path.split('.')[0]+'_norm.npz',data_features=data_features,data_labels=data_labels)
        if shape == 'BFT':
            self.ts_data = torch.from_numpy(data_features).to(torch.float32)
            self.data_labels = torch.from_numpy(data_labels)
            self.T = self.ts_data.shape[2]
            self.F = self.ts_data.shape[1]
        else:
            self.ts_data = torch.from_numpy(data_features.transpose(0, 2, 1))
            self.data_labels = torch.from_numpy(data_labels)
            self.T = self.ts_data.shape[1]
            self.F = self.ts_data.shape[2]
        print(np.unique(self.data_labels).shape[0],data_features.shape, data_features.dtype,data_features.mean(),data_features.std(),np.count_nonzero(data_labels))

    def get_numpy_data_and_label(self):
        return self.ts_data.squeeze().numpy(), self.data_labels.squeeze().numpy()

    def num_classes(self):

        return np.unique(self.data_labels).shape[0]

    def ts_len(self):
        return self.T

    def f_dim(self):
        return self.F

    def __len__(self):
        return self.ts_data.shape[0]
    def _aug(self,features):
        r = np.random.randint(0, 7, 1)
        if r == 0:
            aug_features = torch.round(features, decimals=6)
        elif r == 1:
            aug_features = torch.round(features, decimals=4)
        elif r == 2:
            aug_features = torch.from_numpy(bf(1280, 0.5, 30, features.numpy()))
        elif r == 3:
            aug_features = torch.from_numpy(bf(1280, 1, 30, features.numpy()))
        elif r == 4:
            aug_features = torch.from_numpy(bf(1280, 1, 25, features.numpy()))
        elif r == 5:
            aug_features = strong_shift(features, sigma=0.1, IN=False)
        else:
            aug_features = strong_shift(features, sigma=0.1, IN=True)
        return aug_features
    def __getitem__(self, index):
        return self.ts_data[index], self.data_labels[index], index


def get_dataset(path, dataset_name,shape='BFT',aug=False,norm=False):
    if dataset_name == 'ECG200' or dataset_name == 'mit' or dataset_name=='cgm' or dataset_name=='PTBXL':
        if dataset_name=='mit':
            dataset_name = 'mit-bih-arrhythmia'
        train_data_path = os.path.join(path, dataset_name + '_train.npz')
        test_data_path = os.path.join(path, dataset_name + '_test.npz')

        train_dateset = TSDataset(file_path=train_data_path,shape=shape,aug=aug,norm=norm)
        test_dateset = TSDataset(file_path=test_data_path,shape=shape,aug=aug,norm=norm)
    else:
        train_data_path = os.path.join(path, dataset_name + '_TRAIN_10s_norm.npz')
        test_data_path = os.path.join(path, dataset_name + '_TEST_10s_norm.npz')
        train_dateset = TSDataset(file_path=train_data_path, norm=False,shape=shape,aug=aug)
        test_dateset = TSDataset(file_path=test_data_path, norm=False,shape=shape,aug=aug)
    return train_dateset, test_dateset
