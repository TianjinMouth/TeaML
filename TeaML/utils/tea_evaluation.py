import numpy as np
import pandas as pd


def model_evaluation(score, y_true, base_score=630, method='e-freq'):
    if method == 'e-freq':
        cuts = np.arange(10, 100, 10)
        cut = np.percentile(score, cuts)
    elif method == 'e-gap':
        cut = np.arange(base_score, base_score+100, 20)
    else:
        raise KeyError

    cut = np.append(np.array([float('-Inf')]), cut, axis=0)
    cut = np.append(cut, np.array([float('Inf')]), axis=0)
    result = pd.DataFrame({'y': y_true, 'pred': pd.cut(score, cut)})
    rr = result['y'].groupby(result['pred']).agg(['mean', 'count', 'sum'])

    pass_ratio = []
    bad_ratio = []
    bad_callback = []
    bad_precision = []
    total = rr['count'].sum()
    bad_cnt = rr['sum'].sum()
    lift_base = rr['mean'].tolist()[-1]
    lift = []
    for i in rr.index:
        pass_ratio.append(rr[rr.index >= i]['count'].sum() / total)
        bad_ratio.append(rr[rr.index >= i]['sum'].sum() / total)
        bad_callback.append(rr[rr.index <= i]['sum'].sum() / bad_cnt)
        bad_precision.append(rr[rr.index <= i]['sum'].sum() / rr[rr.index <= i]['count'].sum())
        lift.append(rr[rr.index == i]['mean'].values[0] / lift_base)

    rr['pass_ratio'] = ['%.2f'%(i*100) for i in pass_ratio]
    rr['bad_ratio'] = ['%.2f'%(i*100) for i in bad_ratio]
    rr['bad_recall'] = ['%.2f'%(i*100) for i in bad_callback]
    rr['bad_precision'] = ['%.2f'%(i*100) for i in bad_precision]
    rr['lift'] = ['%.2f' % i for i in lift]
    rr['mean'] = rr['mean'].apply(lambda x: '%.2f' % (x * 100))
    return rr
