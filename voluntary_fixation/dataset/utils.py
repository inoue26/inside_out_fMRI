import numpy as np
import os
from typing import List, Tuple
import torch
import random
from voluntary_fixation.envs import SAVE_ROOT


def delayed_label(image_label:np.ndarray, delay:int)->Tuple[np.ndarray, np.ndarray]:
    # delay unit is idx. +1 means 1*TR delay from the image_label
    confound_label = image_label + delay
    if delay < 0:
        image_label = image_label[abs(delay):]# if delay -1, array([   1,    2,    3, ..., 3596, 3597, 3598])
        confound_label = confound_label[abs(delay):] # if delay -1, array([   0,    1,    2, ..., 3595, 3596, 3597])
    if delay > 0:
        image_label = image_label[:-delay]# if delay 1, array([   0,    1,    2, ..., 3595, 3596, 3597])
        confound_label = confound_label[:-delay] # if delay 1, array([   1,    2,    3, ..., 3596, 3597, 3598])
    assert (confound_label >= 0).all(), 'find some confound_label < 0'
    return image_label, confound_label


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms = True


def orthogonal_mat_against_vector(ortho_target_vector, matrix, z_score=True, scale=None):
    # matrix: (n_sample, n_dimension)
    # ortho_target_vector: (n_sample, 1)
    before_matrix = matrix.copy()
    ortho_target_vector = ortho_target_vector - ortho_target_vector.mean(0)
    ortho_target_vector = ortho_target_vector / np.linalg.norm(ortho_target_vector)
    if z_score:
        if scale is not None:
            matrix = (matrix - np.mean(matrix, axis=0)) / scale
        else:
            matrix = (matrix - np.mean(matrix, axis=0)) / matrix.std(axis=0)
    else:
        matrix = matrix - matrix.mean(axis=0)
        pass
    matrix = matrix - np.dot(ortho_target_vector, np.dot(matrix.T, ortho_target_vector).T)
    print('ORTHOGONAL: ', np.dot(matrix.T, ortho_target_vector).mean())

    abs_corrs = []
    corrs = []
    for i in range(matrix.shape[1]):
        abs_corrs.append(np.abs(np.corrcoef(matrix[:,i], ortho_target_vector[:,0])[0,1]))
        corrs.append((np.corrcoef(matrix[:,i], ortho_target_vector[:,0])[0,1]))
    print(np.mean(corrs), np.mean(abs_corrs))
    if np.isnan(matrix).any():
        print('has nan.')
        return before_matrix
    return matrix


