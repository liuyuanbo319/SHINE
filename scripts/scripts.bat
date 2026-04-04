@echo off


rem F:\TS_DATA\dataset\EEG\nmt
set DATASET=mit-bih-arrhythmia
set NUM_RUNS=1
set Z=48
rem set K=32 64 128 256  InceptionTime OSCNN TimesNet AutoFormer Deep4Net Shallow
set P=0 1 2 3
rem set METHODS=  SHINE FCN InceptionTime OSCNN TS2Vec TimesNet MVMS CLOCS
set METHODS=  SHINE FCN InceptionTime OSCNN TS2Vec TimesNet
set DIR=F:\TS_DATA\dataset\ECG\%DATASET%
rem set METHODS=TS2Vec
rem set METHODS=SHINE1 SHINE2 SHINE3 SHINE32 SHINE64 SHINE128 SHINE256 SHINE1024
rem for %%p in (%P%) do (
    rem python ../run.py --run_name %METHODS%%%p --model_name %METHODS% --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir F:\TS_DATA\dataset\EEG\tuab --K 256 --P %%p
rem )

for %%M in (%METHODS%) do (
  python ../run.py --run_name %%M --model_name %%M --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir %DIR% --z_dim %Z%
)
