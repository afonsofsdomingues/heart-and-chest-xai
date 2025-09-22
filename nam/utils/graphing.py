from typing import Sequence
from typing import Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_feature_contributions(
    model,
    dataset,
    custom_features
):
    feature_contributions = []
    unique_features = custom_features
    for i, feature in enumerate(unique_features):
        feature = torch.tensor(dataset[:,i], dtype=torch.float32)
        feat_contribution = model.models[0].feature_nns[i](feature).cpu().detach().numpy().squeeze()
        feature_contributions.append(feat_contribution)

    return feature_contributions


def calc_mean_prediction(
    model,
    dataset,
    custom_features
):
    #@title Calculate the mean prediction

    feature_contributions = get_feature_contributions(model, dataset, custom_features)
    avg_hist_data = {col: contributions for col, contributions in zip(custom_features, feature_contributions)}
    all_indices, mean_pred = {}, {}

    # for i, col in enumerate(custom_features):
    #     feature_i = custom_features[:, i].cpu()
    #     all_indices[col] = np.searchsorted(custom_features[i][:, 0], feature_i, 'left')

    for col in custom_features:
        mean_pred[col] = np.mean([avg_hist_data[col]])  #[i] for i in all_indices[col]]) TODO: check the error here

    return mean_pred, avg_hist_data


def plot_mean_feature_importance(model, dataset, custom_features, width=0.5):

    mean_pred, avg_hist_data = calc_mean_prediction(model, dataset, custom_features)

    def compute_mean_feature_importance(mean_pred, avg_hist_data):
        mean_abs_score = {}
        for k in avg_hist_data:
            try:
                mean_abs_score[k] = np.mean(np.abs(avg_hist_data[k] - mean_pred[k]))
            except:
                continue
        x1, x2 = zip(*mean_abs_score.items())
        return x1, x2

    ## TODO: rename x1 and x2
    x1, x2 = compute_mean_feature_importance(mean_pred, avg_hist_data)

    cols = custom_features
    fig = plt.figure(figsize=(5, 5))
    ind = np.arange(len(x1))
    x1_indices = np.argsort(x2)

    cols_here = [cols[i] for i in x1_indices]
    x2_here = [x2[i] for i in x1_indices]

    plt.bar(ind, x2_here, width, label='NAMs')
    plt.xticks(ind + width / 2, cols_here, rotation=90, fontsize='large')
    plt.ylabel('Mean Absolute Score', fontsize='x-large')
    plt.legend(loc='upper right', fontsize='large')
    plt.title(f'Overall Importance', fontsize='x-large')
    plt.show()

    return fig

