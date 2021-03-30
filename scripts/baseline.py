import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import IDAOData, IDAODataTest, train_transforms, val_transforms

import matplotlib.pyplot as plt

from sklearn import metrics
import xgboost
from sklearn.linear_model import LogisticRegression, LinearRegression
from tqdm.auto import tqdm


def get_features(image):
    thresholds = np.concatenate((
        [-np.inf],
        np.arange(-1, 2.2, 0.1),
        [np.inf],
    ))
    out = []
    for lower, upper in zip(thresholds[:-1], thresholds[1:]):
        mask = ((lower <= image)
              & (image < upper))
        out.append(mask.sum())
    return np.array(out)


def create_X_y(train_ds, idces=range(10)):
    """Returns X, y_r_type, y_energy"""
    X, y_r_type, y_energy = [], [], []
    for idx in idces:
        (image, r_type, energy) = train_ds[idx]
        X.append(get_features(image[0]))
        y_r_type.append(r_type)
        y_energy.append(energy)

    X = np.array(X)
    y_r_type = np.array(y_r_type)
    y_energy = np.array(y_energy)

    return X, y_r_type, y_energy


def create_idx_X(test_ds, idces=range(10)):
    """Returns image_idces, X"""
    image_idces, X = [], []
    for idx in idces:
        image_idx, image = test_ds[idx]
        image = get_features(image[0])
        X.append(image)
        image_idces.append(image_idx)

    image_idces = np.array(image_idces)
    X = np.array(X)

    return image_idces, X


def create_X_y_with_repetitions(ds, idces, num_repetitions):
    """As we have transformation (flips/rotations) it may be beneficial
    to sample augmented data"""
    list_X_nk, list_y_r_type_n, list_y_energy_n = [], [], []
    for i in range(num_repetitions):
        X_nk, y_r_type_n, y_energy_n = create_X_y(ds, idces)
        list_X_nk.append(X_nk)
        list_y_r_type_n.append(y_r_type_n)
        list_y_energy_n.append(y_energy_n)

    return np.concatenate(list_X_nk), np.concatenate(list_y_r_type_n), np.concatenate(list_y_energy_n)


def get_clf_metrics(y_true, y_pred):
    out = dict()
    out['AUC'] = round(metrics.roc_auc_score(y_true, y_pred), 3)
    return out


def get_reg_metrics(y_true, y_pred):
    out = dict()
    out['MAE'] = round(metrics.mean_absolute_error(y_true, y_pred), 3)
    out['RMSE'] = round(metrics.mean_squared_error(y_true, y_pred, squared=False), 3)
    return out


class ClfModel:
    def __init__(self):
        # self.model = xgboost.XGBClassifier(use_label_encoder=False)
        self.model = LogisticRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict_class(self, X, *, threshold=0.5):
        return (self.model.predict_proba(X)[:, 1] > threshold).astype(int)

    def get_metrics(self, X, y, **kwargs):
        y_pred = self.predict(X, **kwargs)
        out = get_clf_metrics(y, y_pred)
        return out

class RegModel:
    def __init__(self):
        # self.model = xgboost.XGBRegressor()
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return np.array([self.closest_energy(val) for val in self.model.predict(X)])

    def predict_float(self, X):
        return self.model.predict(X)

    @staticmethod
    def closest_energy(K, lst=np.array([1, 3, 6, 10, 20, 30])):
        idx = (np.abs(lst - K)).argmin()
        return lst[idx]

    def get_metrics(self, X, y, **kwargs):
        y_pred = self.predict(X, **kwargs)
        out = get_reg_metrics(y, y_pred)
        return out

# Please upload your predictions into the system in the .csv format. The ﬁle should consist of 16564 rows and contain three columns:
# id, classiﬁcation_predictions, regression_predictions
# A sample submission can be found here.

def generate_submission(clf, reg, suffix='_public_test'):
    """
    Saves .csv file with three columns:
    id, classiﬁcation_predictions, regression_predictions
    """
    all_image_idces, all_pred_r_type, all_pred_energy = [], [], []
    for test_dir in ['idao_dataset/public_test', 'idao_dataset/private_test']:
        test_ds = IDAODataTest(test_dir, transform=val_transforms())
        image_idces, X = create_idx_X(test_ds, idces=range(len(test_ds)))
        pred_r_type = clf.predict(X)
        pred_energy = reg.predict(X)
        all_image_idces.append(image_idces)
        all_pred_r_type.append(pred_r_type)
        all_pred_energy.append(pred_energy)

    all_image_idces = np.concatenate(all_image_idces)
    all_pred_r_type = np.concatenate(all_pred_r_type)
    all_pred_energy = np.concatenate(all_pred_energy)

    dict_pred = dict(id=all_image_idces,
                     classification_predictions=all_pred_r_type,
                     regression_predictions=all_pred_energy)

    data_frame = pd.DataFrame(dict_pred, columns=["id", "classification_predictions", "regression_predictions"])
    data_frame.to_csv(f'_assets/submission{suffix}.csv', index=False, header=True)


if __name__ == '__main__':
    train_ds = IDAOData('data/train', transform=train_transforms())
    val_ds = IDAOData('data/val', transform=val_transforms())
    test_ds = IDAOData('data/test', transform=val_transforms())
    test_holdout_ds = IDAOData('data/test_holdout', transform=val_transforms())

    data = dict()
    for name, ds, num_repetitions in [
        ['train', train_ds, 10],
        # ['train', train_ds, 1], # default train
        ['val', val_ds, 1],
        ['test_holdout', test_holdout_ds, 1]
    ]:
        assert name in ['train', 'val', 'test_holdout']
        postfix = num_repetitions if num_repetitions > 1 else ''
        data_fname = f'data/{name}_base_features{postfix}.npz'
        if os.path.exists(data_fname):
            print(f'Loading features from {data_fname}')
            data[name] = np.load(data_fname)
        else:
            X, y_r_type, y_energy = create_X_y_with_repetitions(ds, idces=range(len(ds)),
                                                                 num_repetitions=num_repetitions)
            # X, y_r_type, y_energy = create_X_y(ds, idces=range(len(ds)))
            data[name] = dict(X=X, y_r_type=y_r_type, y_energy=y_energy)
            print(f'Saving features to {data_fname}')
            np.savez(data_fname, **data[name])


    # Predicting particle type: classification
    clf = ClfModel()
    clf.fit(data['train']['X'], data['train']['y_r_type'])

    results_r_type = []

    for name in ['train', 'val']:
        out = clf.get_metrics(data[name]['X'], data[name]['y_r_type'])
        out['description'] = name
        results_r_type.append(out)

    print('r_type')
    print(*results_r_type, sep='\n')

    # Predicting particle energy: regression
    reg = RegModel()
    reg.fit(data['train']['X'], data['train']['y_energy'])

    results_energy = []

    for name in ['train', 'val']:
        out = reg.get_metrics(data[name]['X'], data[name]['y_energy'])
        out['description'] = name
        results_energy.append(out)

    print('energy')
    print(*results_energy, sep='\n')

    # y_r_type_pred = clf.predict_class(data['test_holdout']['X'])
    # y_energy_pred = reg.predict(data['test_holdout']['X'])
    # print('y_r_type  y_r_type_pred  y_energy  y_energy_pred')
    # print(*zip(data['test_holdout']['y_r_type'],
    #            y_r_type_pred,
    #            data['test_holdout']['y_energy'],
    #            y_energy_pred), sep='\n')

    generate_submission(clf, reg)
    print("Submission generated")

