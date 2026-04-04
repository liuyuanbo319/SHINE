@echo off


rem F:\TS_DATA\dataset\EEG\nmt
set DATASET=PTBXL
set NUM_RUNS=3
set Z=96

rem set K=32 64 128 256  InceptionTime OSCNN TimesNet AutoFormer Deep4Net Shallow "F:\TS_DATA\Synthetic Continuous Glucose Monitoring (CGM) Signals"
rem F:\TS_DATA\dataset\ECG\%DATASET%
set METHODS= SHINE
set DIR=F:\TS_DATA\dataset\ECG\PTBXL

python ../run.py --run_name SHINE/Aug-sw --model_name SHINE --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir %DIR% --aug_type sw --z_dim %Z%
rem python ../run.py --run_name SHINE/N --model_name SHINE --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir %DIR% --z_dim %Z%
rem python ../run.py --run_name SHINE/S --model_name SHINE --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir %DIR% --alpha 0.0 --z_dim %Z%
rem python ../run.py --run_name SHINE/Aug --model_name SHINE --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir %DIR% --aug_type oo --z_dim %Z%

rem python ../run.py --run_name SHINE/Aug-ww --model_name SHINE --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir %DIR% --aug_type ww --z_dim %Z%
rem python ../run.py --run_name SHINE/Aug-ss --model_name SHINE --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir %DIR% --aug_type ss --z_dim %Z%

rem python ../run.py --run_name SHINE/I --model_name SHINE --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir %DIR% --K 0 --P 0 --z_dim %Z%
rem python ../run.py --run_name SHINE/Se --model_name SHINE --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir %DIR% --K 0 --z_dim %Z%
rem python ../run.py --run_name SHINE/Tr --model_name SHINE --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir %DIR% --P 0 --z_dim %Z%
rem python ../run.py --run_name MVMS --model_name MVMS --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir F:\TS_DATA\dataset\ECG\%DATASET% --P 0
rem set P=1 2 3
rem set K=64 128 1024
rem for %%p in (%P%) do ( python ../run.py --run_name SHINE-P%%p --model_name SHINE --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir F:\TS_DATA\dataset\EEG\%DATASET% --P %%p)
rem for %%k in (%K%) do (python ../run.py --run_name SHINE-K%%k --model_name SHINE --dataset %DATASET% --num_runs %NUM_RUNS% --preprocessed_dir F:\TS_DATA\dataset\EEG\%DATASET% --K %%k)
