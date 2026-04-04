# Towards Learning Shift-Invariant Representations for Healthcare Series Classification

This repository contains the official PyTorch implementation of SHINE, a framework for learning shift-invariant representations for healthcare time series classification, published in IEEE Transactions on Knowledge and Data Engineering (TKDE), 2026.

---

## 📥 Download Data

- [ECG200](https://www.timeseriesclassification.com/description.php?Dataset=ECG200)
- [MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/1.0.0/)
- [TUAB](https://isip.piconepress.com/projects/tuh_eeg/index.shtml) (registration required)
- [NMT](https://dll.seecs.nust.edu.pk/downloads/)
- [CGM](https://doi.org/10.17632/chd8hx65r4.2)

---

## ⚙️ Preprocessing

The preprocessing pipeline divides raw recordings into contiguous, non-overlapping epochs of fixed length.  
The resulting epochs and their corresponding labels are then stored as `.npz` files.

Run the following command:
```bash
python ./preprocess.py --raw_dir <data-dir> --preprocessed_dir <npz-dir> --dataset <dataset-name> --partition_len <S>
```
- `<data-dir>`: directory containing the downloaded raw data
- `<npz-dir>`: directory to save the processed train/test datasets

### Dataset-specific details
**MIT-BIH Arrhythmia** 
- The raw ECG signals are segmented into individual heartbeats and randomly split into training and testing sets with an 8:2 ratio.

**EEG datasets**
-  All EEG signals are resampled to 128 Hz
- Channels are reordered into the standard 10–20 system:
FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, FZ, CZ, PZ
- Signals are converted to μV (microvolts)
- Recordings are segmented into contiguous, non-overlapping epochs of `<S>` seconds (default: S = 10)

## 🚀 Usage

To train and evaluate SHINE on a dataset, run:
```bash
python run.py <dataset> <run_name> \
    --preprocessed_dir <npz-dir> \
    --z_dim <d> \
    --alpha <smooth-coef>
```
After training, the model checkpoints and outputs will be saved in the result_path directory (default: ./results).

## 🙏 Acknowledgement

We sincerely thank the following open-source projects for their valuable contributions:
- braincode (https://github.com/braindecode/braindecode.git) -
- TS2Vec (https://github.com/yuezhihan/ts2vec)
- OS-CNN (https://github.com/Wensi-Tang/OS-CNN)
- InceptionTime (https://github.com/hfawaz/InceptionTime.git) 
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)

## 📄 Citation

If you find this repository useful, please consider citing:
```
@article{liu2026towards,
  title={Towards Learning Shift-Invariant Representations for Healthcare Series Classification},
  author={Liu, Yuanbo and Li, Xiucheng and Chen, Xinyang and Liu, Hongwei and Li, Zhijun},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2026},
  publisher={IEEE}
}
```