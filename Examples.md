# Coensemble read out

## Landmarking based

`python3.10 co_ensemble_read_out.py -f SavedSklearnModels/fullpymfe-landmarking-cc18+100.joblib -s ../auto-gluon/slurm_output/sk8tmf10l.4037934.i23r05c01s05.out.txt -o output_stuff.json`

`python3.10 autogluon_co_ensemble_read_out.py -f SavedAutogluonModels/ag-20240424_124230-landmarking-cc18+100 -s ../auto-gluon/slurm_output/sk8tmf10l.4037934.i23r05c01s05.out.txt -o output_stuff.json`

## Model based

```
python3.10 co_ensemble_read_out.py -f SavedSklearnModels/fullpymfe-model-based-cc18+100-weighted.joblib -s ../auto-gluon/slurm_output/ag8tmf10m.4031864.i23r06c03s03.out.txt -o output_stuff.json
```

```
python3.10 co_ensemble_read_out.py -f SavedSklearnModels/fullpymfe-model-based-cc18+100+cd1+cd2.joblib -s ../auto-gluon/slurm_output/ag8tmfcd3m.4048423.i23r05c05s02.out.txt -o output_stuff.json
```

```
python3.10 autogluon_co_ensemble_read_out.py -f SavedAutogluonModels/ag-20240424_124230-model-based-cc18+100-weighted -s ../auto-gluon/slurm_output/ag8tmf10m.4031864.i23r06c03s03.out.txt -o output_stuff.json
```

```
python3.10 autogluon_co_ensemble_read_out_parallel.py -f SavedAutogluonModels/ag-20240428_133206-model-based-f1-score-cc18+100 -s slurm_output/ag8tmfcd3m.4048423.i23r05c05s02.out.txt -o ag8tmfcd3m.4048423-f1-score-cc18+100-ag.json -g "model-based"
```

# Parallel read out, Clustering
```
python3.10 co_ensemble_read_out_parallel.py -f SavedSklearnModels/fullpymfe-clustering-100+cc18-v2.joblib -s ../auto-gluon/slurm_output/ag8tmfcd3m.4048423.i23r05c05s02.out.txt -o ag8tmfcd3m.4048423.fullpymfe-clustering-100+cc18-v2-sk.json -g clustering
```

```
python3.10 autogluon_co_ensemble_read_out_parallel.py -f SavedAutogluonModels/ag-20240531_163028-clustering-fbeta-cc18+100 -s ../auto-gluon/slurm_output/ag8tmfcd3m.4048423.i23r05c05s02.out.txt -o ag8tmfcd3m.4048423.fullpymfe-clustering-100+cc18-fbeta-ag.json -g clustering
```

# Parallel read out, Clustering + model based
```
python3.10 co_ensemble_read_out_parallel.py -f SavedSklearnModels/fullpymfe-clustering+model-based-100+cc18+cd1+cd2.joblib -s ../auto-gluon/slurm_output/ag8tmfcd3m.4048423.i23r05c05s02.out.txt -o ag8tmfcd3m.4048423.fullpymfe-clustering+model-based-100+cc18+cd1+cd2-sk.json -g clustering "model-based"
```


# meta feature analysis

`python3.10 sklearn_tree_calc_pymfe.py -f csv_files/fullpymfe-model-based-cc18+100.csv -o fullpymfe-model-based-cc18+100-weighted.joblib`

`python3.10 sklearn_tree_calc_pymfe.py -f csv_files/fullpymfe-clustering-100+cc18.csv -o fullpymfe-clustering-100+cc18-v2.joblib`


