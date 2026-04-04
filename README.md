# Towards Learning Shift-Invariant Representations for Healthcare Series Classification

---
## Download Data
- [ECG200](https://www.timeseriesclassification.com/description.php?Dataset=ECG200)
- [Mit-Bih-Arrhythmia](https://physionet.org/content/mitdb/1.0.0/)
- [TUAB](https://isip.piconepress.com/projects/tuh\_eeg/index.shtml) (need registration)
- [NMT](https://dll.seecs.nust.edu.pk/downloads/)
- [CGM](https://doi.org/10.17632/chd8hx65r4.2)
## Preprocessing

The preprocessing divides the recordings into contiguous, non-overlapping epochs of fixed length. Then the epochs and their corresponding labels are saved the as `npz` files.

On terminal, run the following:
```
python ./preprocess.py --raw_dir <data-dir> --preprocessed_dir <npz-dir> --dataset <dataset-name> --partition_len <S>
```

where `<data-dir>` is the directory where the downloaded data are located, and `<npz-dir>` is the directory where the train and test dataset files will be saved.

- For Mit-Bih-Arrhythmia, the raw data is divided into one heartbeat length, and split randomly into train and test dataset in a ratio of 8:2
- For EEG signals, the preprocessing step first resamples all EEG signals to 128HZ. Then reorder 19 channels in the order of FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8,479
T3, T4, T5, T6, FZ, CZ, PZ and retrieve the data in unit uV. The recordings are divided into contiguous, non-overlapping epochs of `<S>` seconds. Default `<S>` is 10.

## Usage

To train and evaluate SHINE on a dataset, run the following command:

```train & evaluate
python run.py <dataset> <run_name> --preprocessed_dir <npz-dir> --z_dim <d> --alpha <smooth-coef>
```
After training and evaluation, the trained model, output can be found in `result_path` folder. The default is results 

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- braincode (https://github.com/braindecode/braindecode.git)
- TS2Vec (https://github.com/yuezhihan/ts2vec)
- OS-CNN (https://github.com/Wensi-Tang/OS-CNN)
- InceptionTime (https://github.com/hfawaz/InceptionTime.git)
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)

## Citation
If you're using this repository in your research or applications, please cite using the following BibTeX:

```
@article{liu2026towards,
  title={Towards Learning Shift-Invariant Representations for Healthcare Series Classification},
  author={Liu, Yuanbo and Li, Xiucheng and Chen, Xinyang and Liu, Hongwei and Li, Zhijun},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2026},
  publisher={IEEE}
}
```