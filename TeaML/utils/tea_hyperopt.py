from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from .tea_utils import compute_ks
import time


def hyperopt_cv(X_train, y_train, classifier, n_iter=50, metrics='ks', quick_opt=False, verbose=False, std_weight=0.2, diff_weight=0.2):
    """

    :param X_train: pandas.DataFrame
    :param y_train: pandas.Series or numpy.array
    :param classifier: str
    :param n_iter: numbers of iteration
    :param quick_opt: subsample train set at CV step
    :param verbose: Trials' verbose
    :param std_weight: stdev's weight on loss function
    :param diff_weight: difference's weight between train KS and test KS on loss function
    """

    # ----- step1: define a objective to minimize -----#
    # 必须是最小化
    def objective(params):
        if classifier == "xgb":
            clf = XGBClassifier(**params)
        elif classifier == "lgb":
            clf = LGBMClassifier(**params)
        elif classifier == "lr":
            clf = LogisticRegression(**params)
        else:
            raise ValueError("classifier type currently not support")

        skf = StratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)
        train_scores = []
        cv_scores = []

        for train_index, test_index in skf.split(X_train, y_train):
            if quick_opt:
                train_index = np.random.choice(train_index, size=int(0.1*len(train_index)))
            X_tr, X_vali = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
            y_tr, y_vali = y_train[train_index], y_train[test_index]
            clf.fit(X_tr, y_tr)
            y_train_pred = clf.predict_proba(X_tr)[:, 1]
            y_vali_pred = clf.predict_proba(X_vali)[:, 1]
            if metrics == 'ks':
                train_scores.append(compute_ks(y_train_pred, y_tr))
                cv_scores.append(compute_ks(y_vali_pred, y_vali))
            elif metrics == 'auc':
                train_scores.append(roc_auc_score(y_tr, y_train_pred))
                cv_scores.append(roc_auc_score(y_vali, y_vali_pred))
            elif metrics == 'f1':
                train_scores.append(f1_score(y_tr, y_train_pred))
                cv_scores.append(f1_score(y_vali, y_vali_pred))
            else:
                raise KeyError

        # cv performance
        cv_score = np.mean(cv_scores)
        # cv stability
        cv_std = np.std(cv_scores)
        # train vs. cv differences
        diff = sum([abs(train_scores[i] - cv_scores[i]) for i in range(len(train_scores))]) / N_SPLITS

        # objective: high cv score + low cv std + low train&cv diff
        loss = (1 - cv_score) + cv_std * std_weight + diff * diff_weight

        return {
            'loss': loss,
            'status': STATUS_OK,
            # -- store other results like this
            'eval_time': time.time(),
            'other_stuff': {'cv_std': cv_std, 'cv_score': cv_score},
            #         # -- attachments are handled differently
            #         'attachments':
            #             {'time_module': pickle.dumps(time.time)}
        }

    # ----- step2: define a search space -----#
    # search_space = {}
    if classifier == "xgb":
        search_space = {
            'n_estimators': hp.choice('n_estimators', np.arange(50, 200, dtype=int)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            # A problem with max_depth casted to float instead of int with
            # the hp.quniform method.
            'max_depth': hp.choice('max_depth', np.arange(2, 5, dtype=int)),
            # "max_delta_step": hp.quniform('max_delta_step', 0, 20, 1),
            # 'min_child_weight': hp.quniform('min_child_weight', 0, 100, 1),
            # 'subsample': hp.uniform('subsample', 0.2, 1.0),
            # 'gamma': hp.uniform('gamma', 0.1, 50),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
            "reg_lambda": hp.uniform('reg_lambda', 0.1, 100),
            "reg_alpha": hp.uniform('reg_alpha', 0.1, 100),
            "scale_pos_weight": 1,
            'eval_metric': 'auc',
            'objective': 'binary:logistic',
            # Increase this number if you have more cores. Otherwise, remove it and it will default
            # to the maxium number.
            'nthread': 6,
            # 'booster': hp.choice("booster", ['dart',  "gbtree"]),
            'booster': 'gbtree',
            'tree_method': hp.choice("tree_method", ['exact', "approx"]),
            'silent': True,
            'seed': 42
        }
    elif classifier == "lr":
        search_space = {
            # 'class_weight': hp.choice("class_weight", ["balanced", None]),
            "C": hp.uniform("C", 1e-3, 1.0),
            # 'penalty': hp.choice("penalty", ["l1", "l2"]),
        }
    elif classifier == "lgb":
        search_space = {
            'learning_rate': hp.uniform("learning_rate", 0.01, 0.2),
            'num_leaves': hp.choice('num_leaves', np.arange(8, 50, dtype=int)),
            # 'max_depth': (0, 5),
            'min_child_samples': hp.choice('min_child_samples', np.arange(20, 200, dtype=int)),
            # 'max_bin': (100, 1000),
            'subsample': hp.uniform('subsample', 0.1, 1.0),
            'subsample_freq': hp.choice("subsample_freq", np.arange(0, 10, dtype=int)),
            'colsample_bytree': hp.uniform("colsample_bytree", 0.01, 1.0),
            'min_child_weight': hp.uniform("min_child_weight", 1e-3, 10),
            # 'subsample_for_bin': (100000, 500000),
            'reg_lambda': hp.uniform("reg_lambda", 1.0, 1000),
            'reg_alpha': hp.uniform("reg_alpha", 1.0, 100),
            'scale_pos_weight': hp.uniform("scale_pos_weight", 1.0, 50),
            'n_estimators': hp.choice('n_estimators', np.arange(50, 200, dtype=int)),
        }
    elif classifier == "rf":
        search_space = {
            'n_estimators': hp.choice("n_estimators", np.arange(100, 1000, dtype=int)),
            "max_depth": hp.choice('max_depth', np.arange(3, 5, dtype=int)),
            'min_samples_split': hp.choice("min_samples_split", np.arange(2, 200, dtype=int)),
            "max_features": hp.uniform("max_features", 0.2, 1.0),
        }
    else:
        raise ValueError("classifier type currently not support")

    # ----- step3: define a trial object for tracking -----#
    trials = Trials()
    # ----- step4: optimization by fmin -----#
    N_SPLITS = 5
    best = fmin(
        objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=n_iter,
        trials=trials,
        verbose=verbose)
    best_params = space_eval(search_space, best)
    stds = [t['result']['other_stuff']['cv_std'] for t in trials.trials]
    amin = np.argmin([t['result']['loss'] for t in trials.trials])
    best_ks = 1-(trials.trials[amin]['result']['loss'] - stds[amin] * std_weight)

    print('Max cv Loss std: %.3f' % max(stds))
    print('Best cv KS: %.3f+-%.3f ' % (best_ks, stds[amin]))
    return best_params, trials

