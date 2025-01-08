# Text Format
This is printed when various code is run, contains all of the output. Canonical format :
```
openml_dataset='kr-vs-kp'
class_name='class'
2556
split_labeled=255, split_unlabeled=2555
(255, 37) (2300, 36), (640, 37)
(2300, 2)
(1099,)
1201 ignored
(1201, 2)
(255, 37) (1354, 37), (640, 37)
X_train_2.shape=(2300, 36)
prediction=plus
eval_predictor_1={'accuracy': 0.9359375, 'balanced_accuracy': 0.9342888471177945, 'mcc': 0.8727271686912782, 'roc_auc': 0.9772722822681704, 'f1': 0.9406657018813314, 'precision': 0.9154929577464789, 'recall': 0.9672619047619048}
paper_version: 0.06406250000000002
eval_predictor_2={'accuracy': 0.95, 'balanced_accuracy': 0.9483082706766917, 'mcc': 0.9012162381020238, 'roc_auc': 0.9848547149122808, 'f1': 0.953757225433526, 'precision': 0.9269662921348315, 'recall': 0.9821428571428571}
paper_version: 0.050000000000000044
Accuracy is better
is_plus=True
```

Annotated:
```
openml_dataset='kr-vs-kp' # name of the dataset
class_name='class' # name of the column of the labels to be inferred
2556 # number of datasets used for training
split_labeled=255, split_unlabeled=2555 # index where the split occurs
(255, 37) (2300, 36), (640, 37) # size of the various datasets (labeled, unlabeled, test)
(2300, 2) # size of labels removed from unlabeled datasets (amount of data items, num labels)
(1099,) # number of pseudo-labeled datasets included
1201 ignored # number ignored
(1201, 2) # shape of the pseudo-labeled datasets
(255, 37) (1354, 37), (640, 37) # Shape after M1, (amount of labeled initial, X_train_2_new: labeled + pseudo-labeled, test)
X_train_2.shape=(2300, 36)  # the shape of the unlabeled dataset (should not have changed from before, debug info)
prediction=plus # the prediction of the safeguard system as plus or minus
# predictor M2 accuracy etc...
eval_predictor_1={'accuracy': 0.9359375, 'balanced_accuracy': 0.9342888471177945, 'mcc': 0.8727271686912782, 'roc_auc': 0.9772722822681704, 'f1': 0.9406657018813314, 'precision': 0.9154929577464789, 'recall': 0.9672619047619048}
# co-ensembling paper uses 1 - accuracy
paper_version: 0.06406250000000002
# predictor M1 accuracy etc...
eval_predictor_2={'accuracy': 0.95, 'balanced_accuracy': 0.9483082706766917, 'mcc': 0.9012162381020238, 'roc_auc': 0.9848547149122808, 'f1': 0.953757225433526, 'precision': 0.9269662921348315, 'recall': 0.9821428571428571}
# co-ensembling paper uses 1 - accuracy
paper_version: 0.050000000000000044
Accuracy is better # whether accuracy has imprved
is_plus=True # output from safeguard system as a boolean
```


# Dataclass definition
The text format as it gets read by python as input:

```python
openml_dataset: str  # name of the dataset
class_name: str  # name of the column of the labels to be inferred
# contains the accuracy and other metrics of predictor 1 (M1, trained only on labeled data)
# and predictor 2 (M2, trained on labeled and pseudo-labeled data)
# example: eval_predictor_1={'accuracy': 0.72, 'balanced_accuracy': 0.5261427837699024, 'mcc': 0.491271579492345}
eval_predictor_1: dict
eval_predictor_2: dict
# how many data items were ignored because their predicted probability was too low
number_ignored: int
# how many data items will be unlabeled after the initial split
initial_unlabeled: int
# the number of labels in the class to be inferred
number_of_labels: int
# output of the safeguard system (might be None on error)
is_plus: bool | None
# any other lines that could not be characterized
other_data: list[str]
```

# JSON Format
Similar to the dataclass definition, other data are all the extra lines in the text format.
They are useful for debugging, but not for much else.

```json
{
    "openml_dataset": "breast-w",
    "class_name": "Class",
    "eval_predictor_1": {
      "accuracy": 0.9214285714285714,
      "balanced_accuracy": 0.9301412872841444,
      "mcc": 0.8375798210570724,
      "roc_auc": 0.9747701278313523,
      "f1": 0.8952380952380952,
      "precision": 0.8392857142857143,
      "recall": 0.9591836734693877
    },
    "eval_predictor_2": {
      "accuracy": 0.9214285714285714,
      "balanced_accuracy": 0.9065934065934066,
      "mcc": 0.8257835379663219,
      "roc_auc": 0.987777528593855,
      "f1": 0.8842105263157894,
      "precision": 0.9130434782608695,
      "recall": 0.8571428571428571
    },
    "number_ignored": 202,
    "initial_unlabeled": 503,
    "number_of_labels": 2,
    "is_plus": false,
    "other_data": [
      "559\n",
      "split_labeled=55, split_unlabeled=558\n",
      "(55, 10) (503, 9), (140, 10)\n",
      "(301,)\n",
      "(55, 10) (356, 10), (140, 10)\n",
      "prediction=plus\n",
      "paper_version: 0.07857142857142863\n",
      "paper_version: 0.07857142857142863\n"
    ]
}
```

