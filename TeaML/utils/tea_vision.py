import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd


def plotcut(pred, y_true, bin_split=10):
    cuts = np.arange(bin_split, 100, bin_split)
    fig, ax1 = plt.subplots(figsize=(12, 7), facecolor='white')
    cut = np.percentile(pred, cuts)
    cut = np.append(np.array([float('-Inf')]), cut, axis=0)
    cut = np.append(cut, np.array([float('Inf')]), axis=0)
    result = pd.DataFrame({'y': y_true, 'pred': pd.cut(pred, cut)})
    result['y'].groupby(result['pred']).mean().plot(kind='bar')

    ax1.set_ylabel('bad rate')

    ax2 = ax1.twinx()
    cumsum = result['y'].groupby(result['pred']).sum().cumsum() / sum(y_true)
    plt.plot(cumsum.index.astype('str'), cumsum.tolist(), color='r')
    ax2.set_ylabel('bad catch')
    plt.show()


def plot_ks(y_test, y_pred_prob):
    """
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    输入值:
    y_pred_prob: 一维数组或series，代表模型得分（一般为预测正类的概率）
    y_test: 真实值，一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    """
    fpr, tpr, thresholds = roc_curve(y_test,y_pred_prob)
    ks = max(tpr-fpr)
#    # 画ROC曲线
#     plt.plot([0,1],[0,1],'k--')
#     plt.plot(fpr,tpr)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.show()
    # 画ks曲线
    plt.plot(tpr)
    plt.plot(fpr)
    plt.plot(tpr-fpr)
    plt.show()


def monthly_bad_rate(data_set, time_col):
    year_month = pd.to_datetime(data_set[time_col]).dt.strftime('%Y-%m')

    yms = []
    cnt = []
    bad_rate = []
    for ym in sorted(year_month.unique()):
        _tmp = data_set[year_month == ym]
        yms.append(ym)
        bad_rate.append(_tmp['is_bad'].mean())
        cnt.append(_tmp['is_bad'].count())

    plt.figure(figsize=(12, 12))
    plt.subplot(211)
    plt.bar(yms, bad_rate_15, width=0.6)
    plt.title('bad rate')

    ax = plt.subplot(212)
    cat_counts = data_set.pivot_table(index=year_month, columns=['funds_channel'], values=['uid'], aggfunc='count')
    cat_counts.columns = ['xw', 'dsd', 'zy', 'dkcs']
    cat_counts.plot(kind='bar', stacked=True, ax=ax, width=0.6)
    ax.set_xticklabels(sorted(year_month.unique()), rotation=0)
    plt.title('count')
