@echo off


rem F:\TS_DATA\dataset\EEG\nmt FCN InceptionTime OSCNN TimesNet
set DATASET=cgm
set NUM_RUNS=1

set METHODS=TS2Vec SHINE

rem for %%p in (%P%) do (
    rem python ../run.py --run_name %METHODS%%%p --model_name %METHODS% --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir F:\TS_DATA\dataset\EEG\tuab --K 256 --P %%p
rem )

for %%M in (%METHODS%) do (
  python ../run.py --run_name %%M --model_name %%M --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir ..\data
)
