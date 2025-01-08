from joblib import dump, load
from pandas import DataFrame, read_csv, Series
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import export_text
import argparse
import numpy as np
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor, TabularDataset

import shap


def main(csv_file_name, input_ag):
    combined_df: DataFrame = read_csv(csv_file_name)
    class_name = 'class'

    print(combined_df)

    # drop all NaN lines:
    combined_df = combined_df.dropna()

    X = combined_df.drop(columns=[class_name])
    y = combined_df[class_name]

    print(X.shape, y.shape, sep="\t")

    # how much of the data is used for training
    training_ratio = 0.8
    # how much of training data stays labeled
    labeled_ratio = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - training_ratio, random_state=42)

    clf = TabularPredictor.load(input_ag)

    print(clf.predict_proba(X_test))
    # print(clf.score(X_test, y_test))

    # feature_importances(clf, combined_df)
    cross_val_score_autogluon(combined_df, class_name)
    confusion_matrix_classification_report(clf, X_test, y_test)
    # tree_depth(clf, X)

    # plots:
    # plot_learning_curve(clf, X, y)
    # explainer_shap(clf, X)
    # graph_importances(clf, X)

    # tune_hyperparams(clf, X_train, y_train)
    # best hyperparams cc18+100 landmarking
    # {'bootstrap': True, 'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200}
    # best hyperparams cd3 landmarking
    # {'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 100}
    # best hyperparams cc18+100 model based weighted v2
    # {'bootstrap': False, 'class_weight': {'plus': 1, 'minus': 10}, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}


# from https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
def graph_importances(clf, X):
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    feature_names = [f"feature {i}" for i in range(X.shape[1])]

    forest_importances = Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()


# started from chatgpt code
def explainer_shap(clf, X):
    explainer = shap.Explainer(clf)
    shap_values = explainer(X)
    shap.plots.bar(shap_values[..., 0])
    shap.plots.beeswarm(shap_values[..., 0])
    shap.summary_plot(shap_values[..., 0], X)


# from chatgpt
def tree_depth(clf, X):
    for i in range(len(clf.estimators_)):
        print(export_text(clf.estimators_[i], feature_names=list(X.columns)))


# from chatgpt
def feature_importances(clf, combined_df):
    importances = clf.feature_importance(combined_df)
    print(importances)


# from chatgpt
def cross_val_score_autogluon(train_data, label):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_index, val_index in kf.split(train_data):
        train_fold = train_data.iloc[train_index]
        val_fold = train_data.iloc[val_index]

        predictor = TabularPredictor(label=label).fit(train_fold, verbosity=0)

        eval_result = predictor.evaluate(val_fold)

        score = eval_result["accuracy"]

        cv_scores.append(score)

    print("CV Scores:", cv_scores)
    print("Mean CV Score: ", np.mean(cv_scores))


# from chatgpt
def confusion_matrix_classification_report(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(
"""
[[TN FP]
 [FN TP]]
"""
    )
    print(cm)

    print(classification_report(y_test, y_pred))


# from chatgpt
def plot_learning_curve(clf, X, y):
    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, label='Training score')
    plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend()
    plt.show()


# from chatgpt
from sklearn.model_selection import GridSearchCV


def tune_hyperparams(clf, X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'class_weight': [{"plus": 5, "minus": 1}, {"plus": 1, "minus": 10}]
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Best params = {grid_search.best_params_}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='csv file name', required=True)
    parser.add_argument('-i', '--input', type=str, help='input saved autogluon model name', required=True)
    args = parser.parse_args()
    main(args.file, args.input)
