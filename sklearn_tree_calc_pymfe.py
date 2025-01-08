"""Trains an Sklearn based RandomForestClassifier Safeguard System Predictor."""
from joblib import dump, load
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import argparse

def main(csv_file_name, output_joblib):
    combined_df: DataFrame = read_csv(csv_file_name)
    class_name = 'class'

    print(combined_df)

    # drop all NaN lines:
    combined_df = combined_df.dropna()

    X = combined_df.drop(columns=[class_name])
    y = combined_df[class_name]

    print(X.shape, y.shape, sep="\t")

    # how much of the data is used for training
    training_ratio = 0.9
    # how much of training data stays labeled
    labeled_ratio = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - training_ratio, random_state=42)

    # from https://scikit-learn.org/stable/modules/ensemble.html#random-forests-and-other-randomized-tree-ensembles
    # see https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_6_ImbalancedLearning/CostSensitive.html
    # clf = RandomForestClassifier(bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100, class_weight={"plus": 5, "minus":1}, random_state=42)
    # calculated from sklearn_tree_debug_pymfe.py
    clf = RandomForestClassifier(bootstrap=False, class_weight={'plus': 2, 'minus': 1}, max_depth= None, min_samples_leaf=1, min_samples_split=2, n_estimators=200, random_state=42)
    clf: RandomForestClassifier = clf.fit(X_train, y_train)
    print(clf.predict_proba(X_test))
    print(clf.score(X_test, y_test))
    # NOTE: better name for this?
    sklearn_model_is_plus = "SavedSklearnModels/" + output_joblib
    dump(clf, sklearn_model_is_plus)

    clf2: RandomForestClassifier = load(sklearn_model_is_plus)
    print(clf2.predict(X_test.head(1)).item())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='csv file name', required=True)
    parser.add_argument('-o', '--output', type=str, help='output file name (needs joblib extension)', default=None)
    args = parser.parse_args()
    output_joblib = args.file.replace(".csv", ".joblib") if args.output is None else args.output
    main(args.file, output_joblib)
