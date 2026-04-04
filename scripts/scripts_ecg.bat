@echo off


rem F:\TS_DATA\dataset\EEG\nmt
set DATASET=PTBXL
set NUM_RUNS=3
rem SHINE  AutoFormer
set METHODS=MVMS FCN InceptionTime OSCNN TimesNet TS2Vec
set METHODS=SHINE
rem for %%p in (%P%) do (
    rem python ../run.py --run_name %METHODS%%%p --model_name %METHODS% --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir F:\TS_DATA\dataset\EEG\tuab --K 256 --P %%p
rem )

for %%M in (%METHODS%) do (
  python ../run.py --run_name %%M --model_name %%M --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir F:\TS_DATA\dataset\ECG\%DATASET%
)
