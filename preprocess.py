import argparse
import os

import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import wfdb


def load_mit_arr(raw_dir):

    all_data = []
    all_label = []

    used_record = [101, 105, 106, 107, 108, 109, 111, 112, 113, 115, 116, 118, 119, 121,
                   122, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215,
                   217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]
    beat_types = ['N', 'L', 'R', 'A', 'V', 'a', 'P', 'F', 'S', 'j', 'f', 'E', 'J', 'e', 'Q']

    path = raw_dir
    for r in tqdm(used_record):
        annotation = wfdb.rdann(
            path + str(r), 'atr',
            return_label_elements=(['symbol', 'description']))
        # annotation.fs = 360

        # wfdb.plot_wfdb(record = record,annotation=annotation, time_units='seconds')

        record = wfdb.rdrecord(path + str(r), physical=True)
        sig = record.p_signal
        i = 0
        while i < len(annotation.symbol):
            tqdm.write("current %d" % i)
            s = annotation.symbol[i]
            if s not in beat_types:
                i = i + 1
                continue
            idx = annotation.sample[i]
            if idx < 99 or idx + 201 > 650000:
                i = i + 1
                continue
            all_data.append(sig[idx - 99:idx + 201])
            all_label.append(s)
            i = i + 1

    all_data = np.array(all_data).transpose(0, 2, 1).astype(np.float32)
    all_labels = []
    for s in all_label:
        if s in ['L', 'N', 'R', 'e', 'j']:
            all_labels.append(0)
        if s in ['A', 'J', 'S', 'a']:
            all_labels.append(1)
        if s in ['E', 'V']:
            all_labels.append(2)
        if s in ['F']:
            all_labels.append(3)
        if s in ['P', 'Q', 'f']:
            all_labels.append(4)
    all_labels = np.array(all_labels, dtype=np.int64)
    return all_data, all_labels


def load_tuab(raw_edf_dir, save_path, epochs_len):
    # raw_edf_dir = 'F:\时间序列相关数据\\TUAB\\tuh_eeg_abnormal\\v3.0.0\\edf'
    meta_data = pd.read_csv('./tuab_metainfo.csv')
    train_labels = []
    train_datas = []
    test_labels = []
    test_datas = []

    errs = []
    cns = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF',
           'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF',
           'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF',
           'EEG T5-REF', 'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']
    for s in ['train_normal', 'train_abnormal']:  # 'train_normal','train_abnormal',
        files = meta_data[s]
        files = files[files.notna()]
        train, normal = s.split('_')
        for f in tqdm(files, total=len(files)):
            try:
                raw = mne.io.read_raw_edf(
                    os.path.join(raw_edf_dir, train, normal, '01_tcp_ar',
                                 f), verbose=False).reorder_channels(cns).resample(128)
            except Exception as e:
                errs.append(str(e) + str(f))
                print(e, f)
            try:
                epochs = mne.make_fixed_length_epochs(raw, duration=epochs_len, verbose=False)  # get 10 secs epochs
            except Exception as e:
                errs.append(str(e))
                print(e)
            data = epochs.get_data(units='uV').astype(np.float32)  # get chanels data B,C,T

            train_datas.append(data)
            for i in range(data.shape[0]):
                if normal == 'normal':
                    train_labels.append(1)
                else:
                    train_labels.append(0)
    trian_file_name = os.path.join(save_path, f'tuab_train_{epochs_len}s.npz')
    tl = np.array(train_labels).astype(np.int64)
    td = np.concatenate(train_datas, 0)
    np.savez(trian_file_name, data_features=td, data_labels=tl)
    del tl, td, train_datas, train_labels
    for s in ['eval_normal', 'eval_abnormal']:  # 'train_normal','train_abnormal',
        files = meta_data[s]
        files = files[files.notna()]
        train, normal = s.split('_')
        for f in tqdm(files, total=len(files)):
            try:
                raw = mne.io.read_raw_edf(
                    os.path.join(raw_edf_dir, train, normal, '01_tcp_ar',
                                 f), verbose=False).reorder_channels(cns).resample(128)
            except Exception as e:
                errs.append(str(e) + str(f))
                print(e, f)
            try:
                epochs = mne.make_fixed_length_epochs(raw, duration=epochs_len, verbose=False)  # get 10 secs epochs
            except Exception as e:
                errs.append(str(e))
                print(e)
            data = epochs.get_data(units='uV').astype(np.float32)  # get chanels data B,C,T

            test_datas.append(data)
            for i in range(data.shape[0]):
                if normal == 'normal':
                    test_labels.append(1)
                else:
                    test_labels.append(0)
    tel = np.array(test_labels).astype(np.int64)
    ted = np.concatenate(test_datas, 0)
    test_file_name = os.path.join(save_path, f'tuab_test_{epochs_len}s.npz')
    np.savez(test_file_name, data_features=ted, data_labels=tel)
    print(errs)


def load_nmt(raw_edf_dir, save_path, epochs_len=10):
    # raw_edf_dir = 'F:\时间序列相关数据\\nmt_scalp_eeg_dataset\\nmt_scalp_eeg_dataset'
    meta_data = pd.read_csv(os.path.join(raw_edf_dir, 'Labels.csv'), index_col='recordname')
    train_labels = []
    train_datas = []
    test_labels = []
    test_datas = []
    errs = []
    cns = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ',
           'CZ', 'PZ']

    for rn in meta_data.index:
        l = meta_data.loc[rn]['label']
        loc = meta_data.loc[rn]['loc']
        try:
            #         count = count+1
            #         open(os.path.join('F:\时间序列相关数据\\nmt_scalp_eeg_dataset\\nmt_scalp_eeg_dataset',l,loc,rn))
            raw = mne.io.read_raw_edf(
                os.path.join(raw_edf_dir, l, loc,
                             rn))
            raw = raw.reorder_channels(cns).resample(128)
        except Exception as e:
            errs.append((l, loc, rn))
        try:
            epochs = mne.make_fixed_length_epochs(raw, duration=epochs_len)  # get 10 secs epochs
        except Exception as e:
            errs.append(str(e))
        data = epochs.get_data(units='uV')
        data = data.astype(np.float32)  # get chanels data B,C,T

        if loc == 'train':
            train_datas.append(data)
            for i in range(data.shape[0]):
                if l == 'normal':
                    train_labels.append(0)
                elif l == 'abnormal':
                    train_labels.append(1)
        elif loc == 'eval':
            test_datas.append(data)
            for i in range(data.shape[0]):
                if l == 'normal':
                    test_labels.append(0)
                elif l == 'abnormal':
                    test_labels.append(1)
                else:
                    print('other label ' + l)
        else:
            print('other loc ' + loc)

    tl = np.array(train_labels).astype(np.int64)
    tel = np.array(test_labels).astype(np.int64)
    td = np.concatenate(train_datas, 0)
    ted = np.concatenate(test_datas, 0)
    print(tl.shape, td.shape, tel.shape, ted.shape)
    trian_file_name = os.path.join(save_path, f'nmt_train_{epochs_len}s.npz')
    test_file_name = os.path.join(save_path, f'nmt_test_{epochs_len}s.npz')
    np.savez(trian_file_name, data_features=td, data_labels=tl)
    np.savez(test_file_name, data_features=ted, data_labels=tel)
    print(errs)


def preprocessing(raw_dir, save_dir, dataset, partition_len):
    if dataset == 'mit-bih-arrhythmia':
        load_mit_arr_per_patient(raw_dir)
        all_data, all_labels = load_mit_arr(raw_dir)
        ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=0)
        train_saved_path = os.path.join(save_dir, f'{dataset}_train.npz')
        test_saved_path = os.path.join(save_dir, f'{dataset}_test.npz')

        for train_index, test_index in ss.split(all_data, all_labels):
            print(all_data[train_index].shape)
            print(all_data[test_index].shape)

            np.savez(train_saved_path, data_features=all_data[train_index], data_labels=all_labels[train_index])
            np.savez(test_saved_path, data_features=all_data[test_index], data_labels=all_labels[test_index])
    elif dataset == 'tuab':
        all_data, all_labels = load_tuab(raw_dir)
    elif dataset == 'nmt':
        load_nmt(raw_dir, save_dir, epochs_len=partition_len)
    else:
        raise (f'Not implement for {dataset}')
    pass




if __name__ == '__main__':
    parser = argparse.ArgumentParser("preprocessing.")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default='F:\TS_Data\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0/',
        help="Full path to raw data.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Full path to dir to save the patched signals.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='mit-bih-arrhythmia',
        help="The name of the dataset to be preprocessed.",
        choices=['mit-bih-arrhythmia', 'nmt', 'tuab']
    )
    parser.add_argument(
        "--partition_len",
        type=int,
        default=10,
        help="The length of the partitioned signal.",
    )
    args = parser.parse_args()

    preprocessing(args.raw_dir, args.save_dir, args.dataset, args.partition_len)
