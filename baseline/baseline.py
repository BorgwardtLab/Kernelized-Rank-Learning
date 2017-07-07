#!/usr/bin/env python

import numpy as np

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor


def KRR(X_train, Y_train, X_test, alpha, gamma, cv, seed):
    Y_pred = np.zeros([X_test.shape[0], Y_train.shape[1]])
    model = GridSearchCV(KernelRidge(kernel='rbf'), param_grid={"alpha": alpha, "gamma": gamma}, scoring='neg_mean_squared_error', cv=KFold(n_splits=cv, shuffle=True, random_state=seed))
    for i in range(Y_train.shape[1]):
        y_train = Y_train[:, i]
        x_train = X_train[~np.isnan(y_train)]
        y_train = y_train[~np.isnan(y_train)]
        model.fit(x_train, y_train)
        Y_pred[:, i] = model.predict(X_test)
    return Y_pred


def EN(X_train, Y_train, X_test, alpha, l1ratio, cv, seed):
    Y_pred = np.zeros([X_test.shape[0], Y_train.shape[1]])
    model = GridSearchCV(ElasticNet(random_state=seed), param_grid={"alpha": alpha, "l1_ratio": l1ratio}, scoring='neg_mean_squared_error', cv=KFold(n_splits=cv, shuffle=True, random_state=seed))
    for i in range(Y_train.shape[1]):
        y_train = Y_train[:, i]
        x_train = X_train[~np.isnan(y_train)]
        y_train = y_train[~np.isnan(y_train)]
        model.fit(x_train, y_train)
        Y_pred[:, i] = model.predict(X_test)
    return Y_pred    


def RF(X_train, Y_train, X_test, n_estimators, cv, seed):
    Y_pred = np.zeros([X_test.shape[0], Y_train.shape[1]])
    model = GridSearchCV(RandomForestRegressor(random_state=seed), param_grid={"n_estimators": n_estimators}, scoring='neg_mean_squared_error', cv=KFold(n_splits=cv, shuffle=True, random_state=seed))
    for i in range(Y_train.shape[1]):
        y_train = Y_train[:, i]
        x_train = X_train[~np.isnan(y_train)]
        y_train = y_train[~np.isnan(y_train)]
        model.fit(x_train, y_train)
        Y_pred[:, i] = model.predict(X_test)
    return Y_pred 
