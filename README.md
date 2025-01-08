# Description of Code Files

Files with `_parallel` are the same as the normal, but use `ProcessPoolExecutor` to run the code in parallel.

- autogluon_standalone_linear_ensemble_read_out: It reads data from a slurm output file and only change the is_plus item using the plus_or_minus functions. Prints the output of this as a json file. This allows for much quicker full runs, as only the plus_or_minus funcs need to be run.
- autogluon_semi-supervised: Pipeline utilizing Autogluon's built-in support for Semi-Supervised Learning.
- autogluon_standalone_calc_pymfe: Trains an Autogluon based Safeguard System Predictor.
- autogluon_standalone_load_model_pymfe: Load a TabularPredictor to run it for testing
- autogluon_standalone_pymfe_full_study_*: Run through the full dataset(s) mentioned with the full linear-ensembling pipeline. Uses Autogluon based Safeguard System.
- custom_datasets: the custom datasets as curated by Fusi et al. in "Probabilistic Matrix Factorization for Automated Machine Learning"
- pyme-runner: Creates a CSV file that can be used to train a safeguard system predictor from a given slurm_file.
- read_out_compare: Counts which run performed better given 2 slurm_output runs by counting which run won for each dataset.
- read_out_txt: Reads out the output from linear-ensembling runs and converts it into a defined datastructure (OpenMLDatasetResult).
- safeguard_perf: Calculate the mean accuracy of runs with and without the safeguard system. Also scatter plot the runs with and without the safeguard system.
- sklearn_linear_ensemble_read_out: It reads data from a slurm output file and only change the is_plus item using the plus_or_minus functions. Prints the output of this as a json file. This allows for much quicker full runs, as only the plus_or_minus funcs need to be run.
- sklearn_tree_calc_pymfe: Trains an Sklearn based RandomForestClassifier Safeguard System Predictor.
- sklearn_tree_debug_pymfe: Graph and explain the output from the RandomForestClassifier. Also is able to tune the hyperparameters of the RandomForestClassifier.
- sklearn_tree_pymfe_full_study: Run through the full dataset(s) selected with the full linear-ensembling pipeline. Uses Sklearn based Safeguard System.
- test_linear_ensemble: A test file for testing the linear ensembling process.

# Running meta feature analysis (v2, using pymfe and autogluon as the predictor)

1. Run `pymfe-runner.py -i <slurm_output file> -o <csv_file>`. Decide which metric for meta-features (`MFE`) should be used, e.g. landmarking, model-based or a combination.
2. Run `autogluon_standalone_calc_pymfe.py -f <csv file>` to create the Autogluon model, which will be saved in "AutogluonModels". To change the eval model used, it can be imported and changed when calling the `main` function. The default is `recall`.
3. Use this AutoGluon model in e.g., `autogluon_standalone_pymfe_full_study_100.py`. Edit `slurm_full_study_pymfe-diff-cluster.sh` to include the file name of the model output in step 2. Don't forget to edit MFE to use the same metrics as in `pymfe-runner.py`.

## Examples
`./autogluon_standalone_calc_pymfe.py -f csv_files/fullpymfe-model-based-cc18+100.csv`

Usage:
`autogluon_standalone_pymfe_full_study.py -f <saved model from autogluon>` or e.g., `slurm_full_study_pymfe_custom_datasets-diff-cluster-cd3.sh` for a slurm version

# Running meta feature analysis (v2, using pymfe and sklearn as the predictor)

1. Run `pymfe-runner.py -i <slurm_output file> -o <csv_file>`. Decide which metric for meta-features (`MFE`) should be used, e.g. landmarking, model-based or a combination.
2. Run `sklearn_tree_calc_pymfe.py -f <csv file> -o <output file>.joblib` to create the sklearn model, which will be saved in "SavedSklearnModels".
3. Use this sklearn model in `sklearn_tree_pymfe_full_study.py`. Edit `slurm_full_study_sklearn_pymfe-diff-cluster.sh` to include the file name of the model output in step 2. Don't forget to edit MFE to use the same metrics as in `pymfe-runner.py`.

## Examples
`./sklearn_tree_calc_pymfe.py -f csv_files/fullpymfe-model-based-cc18+100.csv`

Usage:
`sklearn_tree_pymfe_full_study.py -f <saved model from sklearn>.joblib` or `slurm_full_study_sklearn_pymfe-diff-cluster.sh` for a slurm version

# Faster Version of v2, Step 3

If a slurm output file of the selected dataset and an AutoGluon/sklearn model already exist, and only the `_full_study` etc.. needs to be done to calculate what the model will predict ("plus" or "minus"), then the `co_ensemble_read_out.py` (sklearn) and `autogluon_co_ensemble_read_out.py` (autogluon) can be used. The syntax is as follows: `python3.10 co_ensemble_read_out.py -f <sklearn model> -s <slurm_output> -o <output file.json>` for sklearn, similar for autogluon. Instead of a regular formatted slurm_output file, this code returns a json version, as described in "Output Format.md".

# Even Faster Version of Step 3

`co_ensemble_read_out.py` and `autogluon_co_ensemble_read_out.py` also have a parallel versions, which run the same code but in parallel with the help of `ProcessPoolExecutor`, which means that the GIL is not a [problem](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor).