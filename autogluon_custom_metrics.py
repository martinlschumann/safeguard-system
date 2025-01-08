import numpy as np
from sklearn.metrics import fbeta_score, confusion_matrix
from autogluon.core.metrics import make_scorer

# see https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-metric.html#custom-mean-squared-error-metric
def f_beta(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # beta of 2 means that "recall [is] twice as important as precision"
    # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
    return fbeta_score(y_true, y_pred, beta=0.5)

ag_f_beta_custom_scorer = make_scorer(name='f_beta',
                                      score_func=f_beta,
                                      optimum=1,
                                      greater_is_better=True)
# modified from chatgpt
def weighted_false_negative_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    # true negatives, false positives, false negatives, true positives
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # weights for the 4 types
    # weigh false negatives the highest
    weight_fn = 10
    weight_fp = 1
    weight_tn = 0
    weight_tp = 0

    # Calculate weighted loss
    weighted_loss = (weight_fn * fn) + (weight_fp * fp) + (weight_tn * tn) + (weight_tp * tp)

    return weighted_loss

ag_weighted_false_negative_loss_custom_scorer = make_scorer(name='weighted_false_negative_loss',
                                      score_func=weighted_false_negative_loss,
                                      optimum=0,
                                      greater_is_better=False)
