import pandas as pd
import numpy as np
import torch

class Dataset:
    def __init__(self, flat_onehot_features_2d, y, max_len, attribute_dims, mask):
        self.flat_onehot_features_2d = flat_onehot_features_2d
        self.y = y
        self.max_len = max_len
        self.attribute_dims = attribute_dims
        self.mask = mask

class MinMaxScaler(object):
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        self.min = torch.min(data.x, dim=0, keepdim=True)[0]
        self.max = torch.max(data.x, dim=0, keepdim=True)[0]

    def transform(self, data):
        data.x = (data.x - self.min) / (self.max - self.min)
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


def precision_at_k(y_true, y_scores, k):
    from sklearn.metrics import precision_score
    threshold = np.percentile(y_scores, 100 - k)
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return precision_score(y_true, y_pred)


def recall_at_k(y_true, y_scores, k):
    from sklearn.metrics import recall_score
    threshold = np.percentile(y_scores, 100 - k)
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return recall_score(y_true, y_pred)