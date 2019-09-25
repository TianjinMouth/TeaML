# -*- coding: utf-8 -*-
# @Time    : 2019-01-15 10:17
# @Author  : finupgroup
# @FileName: VariableCluster.py
# @Software: PyCharm

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy import stats
import numpy as np
import pandas as pd
import random


def _choose_cluster(arg):
    """
    Choose subspace closest to the given variable
    The most similar subspace is choose based on R^2
    Parameters
    ----------
    column: variable to be assigned
    pcas: orthogonal basis for different subspaces
    number_clusters: number of subspaces (clusters)
    :return: index number of subspace closest to variable
    """
    column, pcas, number_clusters = arg
    v1 = np.var(column)
    arg_1 = [(pcas[i], column, v1) for i in range(number_clusters)]
    choose_rule = [_find_best_rule(elem) for elem in arg_1]
    return np.argmax(choose_rule)


def _find_best_rule(arg):
    pca, column, v1 = arg
    try:
        model = LinearRegression().fit(X=pca, y=column)
    except ValueError:
        print(column)
    coif = model.coef_
    intercept = model.intercept_
    res = column - np.dot(pca, coif) - intercept
    v2 = np.var(res)
    temp = -float("inf") if v1 == 0 else (v1 - v2) / v1
    return temp


def _choose_cluster_BIC(arg):
    """
    Selects subspace closest to given variable (according to BIC)
    The most similar subspace is choosen based on BIC
    Parameters
    ----------
    column: variable to be assigned
    pcas: orthogonal basis for different subspaces
    number_clusters: number of subspaces (clusters)
    :return: index number of subspace closest to variable
    """
    column, pcas, number_clusters = arg
    arg_1 = [(pcas[i], column) for i in range(number_clusters)]
    BICs = [_find_best_BIC(elem) for elem in arg_1]
    return np.argmax(BICs)


def _find_best_BIC(arg):
    pca, column = arg
    nparams = pca.shape[1]
    n = len(column)
    model = LinearRegression().fit(pca, column)
    coef = model.coef_
    interept = model.intercept_
    res = column - np.dot(pca, coef) - interept
    sigma_hat = np.sqrt(np.var(res))
    if sigma_hat < 1e-15:
        print("In function _choose_cluster_BIC: estimated value of noise in cluster is %f <1e-16." \
              " It might corrupt the result." % sigma_hat)
    loglik = sum(np.log(stats.norm.cdf(res, 0, sigma_hat)))
    return 2 * loglik - nparams * np.log(n)


def _pca_new_BIC(data, k):
    """
    Computes the value of BIC-like criterion for given data set and
    number of factors. Assumes that number of variables is large
    compared to number of observations
    :param data:pd.DataFrame
    :param k: number of principal components fitted
    :return: BIC value of BIC criterion
    """
    d = data.shape[0]
    N = data.shape[1]
    m = d * k - k * (k + 1) / 2
    lamb = np.linalg.eigvals(np.cov(data.T))
    v = np.sum(lamb[range(k, d, 1)]) / (d - k)
    t0 = -N * d / 2 * np.log(2 * np.pi)
    t1 = -N / 2 * np.sum(np.log(lamb[range(k)]))
    t2 = -N * (d - k) / 2 * np.log(v)
    t3 = -N * d / 2
    pen = -(m + d + k + 1) / 2 * np.log(N)
    return t0 + t1 + t2 + t3 + pen


def variable_cluster(x, number_clusters=10, stop_criterion=1, max_iter=100, max_subspace_dim=4,
                     initial_segmentation=None, estimate_dimension=False):
    """
    Performs k-means based subspace clustering. Center of each cluster is some number of principal components.
    Similarity measure is R^2 coefficient
    :param x: a pd.DataFrame with only continuous variables
    :param number_clusters: an integer, number of clusters to be used
    :param stop_criterion: an integer indicating how many changes in partitions triggers stopping the algorithm
    :param max_iter: an integer, maximum number of iterations of k-means
    :param max_subspace_dim: an integer, maximum dimension of subspaces
    :param initial_segmentation: a list, initial segmentation of variables to clusters
    :param estimate_dimension: a boolean, if TRUE subspaces dimensions are estimated, else value set by default
    :return: segmentation : a list containing the partition of the variables
             pcas : a list of matrices, basis vectors for each cluster (subspace)
    """
    np.random.seed(521)
    num_vars = x.shape[1]
    num_row = x.shape[0]
    pcas = []
    if initial_segmentation is None:
        los = random.sample(range(num_vars), number_clusters)
        pcas = [pd.DataFrame(x[x.columns[los[i]]]) for i in range(len(los))]
        arg = [(x[x.columns[i]], pcas, number_clusters) for i in range(num_vars)]
        segmentation = [_choose_cluster(elem) for elem in arg]
    else:
        segmentation = initial_segmentation
    iteration = 0
    while iteration < max_iter:
        print("第 %d 次迭代" % iteration)
        for i in range(number_clusters):
            index = [j for j, x in enumerate(segmentation) if x == i]
            sub_dim = len(index)
            if sub_dim > max_subspace_dim:
                if estimate_dimension:
                    arg_2 = [(x[x.columns[index]], k) for k in
                             range(np.minimum(np.floor(np.sqrt(sub_dim)), max_subspace_dim))]
                    cut_set = [_pca_new_BIC(elem) for elem in arg_2]
                    cut = np.argmax(cut_set)
                else:
                    cut = max_subspace_dim
                pcas[i] = pd.DataFrame(PCA(n_components=cut).fit_transform(x[x.columns[index]]))
            else:
                dim = np.maximum(1, np.int(np.sqrt(sub_dim)))
                pcas[i] = pd.DataFrame(np.random.randn(num_row, dim))
        if estimate_dimension:
            arg_1 = [(x[x.columns[m]], pcas, number_clusters) for m in range(num_vars)]
            new_segmentation = [_choose_cluster_BIC(elem) for elem in arg_1]
        else:
            arg = [(x[x.columns[n]], pcas, number_clusters) for n in range(num_vars)]
            new_segmentation = [_choose_cluster(elem) for elem in arg]
        if np.count_nonzero(np.array(new_segmentation) - np.array(segmentation)) < stop_criterion:
            break
        segmentation = new_segmentation
        iteration += 1
    cluster_id = list(set(segmentation))
    segmentation_result = pd.DataFrame({"cluster": segmentation, "variable": x.columns})
    segmentation = [list(segmentation_result[segmentation_result.cluster == k]["variable"]) for k in cluster_id]
    return segmentation, pcas

# segmentation, pcas = variable_cluster(X)
# tmp = pcas[0]
# for i in range(1,10):
#     tmp = pd.concat([tmp,pcas[i]],axis=1)
