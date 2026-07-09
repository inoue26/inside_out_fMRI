import numpy as np
from typing import List, Tuple
from sklearn.decomposition import PCA
from himalaya.backend import set_backend
from himalaya.ridge import Ridge, RidgeCV
import torch
from bdpy.ml import add_bias

def regression_svd(x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray,
               device:str, alpha:float)->Tuple[np.ndarray,...]:
    '''Run regression
    '''

    # Add bias terms
    x_train = add_bias(x_train, axis=1) # (time x voxels) -> (time x voxels+1)
    x_test = add_bias(x_test, axis=1)

    force_cpu = False if device == 'cuda' else True
    model = Ridge(alpha=alpha, force_cpu=force_cpu)
    if device == 'cuda':
        x_train = torch.Tensor(x_train).to(device)
        x_test = torch.Tensor(x_test).to(device)
        y_train = torch.Tensor(y_train).to(device)
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    return y_pred_train, y_pred_test, model.coef_ # n_components+1 x n_target_voxel


def feature_prediction(x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, y_test:np.ndarray, alpha:List[float],
                       n_components:int=500, use_gpu:bool=True, regressor='ridge_svd',
                       normalize:bool=False, use_pca_y:bool=False, ret_coef:bool=False,
                       only_y:True=False)->Tuple[np.ndarray,...]:
    '''Run feature prediction with PCA

    Parameters
    ----------
    x_train, y_train : array_like [shape = (n_sample, n_vertices)]
        Source brain data and target brain data for training
    x_test, y_test : array_like [shape = (n_sample, n_vertices)]
        Source brain data and target brain data for test
    alpha, beta : list of float
        The regularization parameters for lasso regression. In original paper, they used alpha = 0 and beta = 0.
    n_components : int
        The number of voxels
    n_iter : int
        The number of iterations
    use_gpu : bool
        Whether to use GPU or not

    Returns
    -------
    predicted_label : array_like [shape = (n_sample, n_vertices)]
        Target brain data predicted from source brain data
    ture_label : array_like [shape = (n_sample, n_vertices)]
        True target brain data in test data
    pca_loadings:  array_like [shape = (n_components, n_vertices)]
        The loadings (weight matrix) of PCA
    '''
    source_n_vertecies = x_train.shape[1]
    target_n_vertecies = y_train.shape[1]

    if normalize:
        # Normalize brian data (x)
        norm_mean_x = np.mean(x_train, axis=0)
        norm_scale_x = np.std(x_train, axis=0, ddof=1)

        x_train = (x_train - norm_mean_x) / norm_scale_x
        x_test = (x_test - norm_mean_x) / norm_scale_x


    if normalize:
        norm_mean_y = np.mean(y_train, axis=0, keepdims=True)
        std_y = np.std(y_train, axis=0, ddof=1, keepdims=True)
        std_y[std_y==0] = 1
        norm_scale_y = std_y

        y_train = (y_train - norm_mean_y) / norm_scale_y

    # PCA
    if (source_n_vertecies > n_components) and not only_y:
        # PCA
        if len(x_train) < n_components:
            n_components =  min(*x_train.shape)
            print('WARNING::: n_components is too large.  Using n_components = {}'.format(n_components))
        pca = PCA(n_components=n_components, copy=True, whiten=False, svd_solver='full')
        pca.fit(x_train) # requiring x_train's shape to be (n_samples, n_features)

        x_train_principal = pca.transform(x_train)
        x_test_principal = pca.transform(x_test) # shape = (n_samples, n_components)
        pca_loadings = pca.components_  # shape = (n_components, n_features)
        assert x_train_principal.shape[1] == n_components
        assert x_test_principal.shape[1] == n_components

    else:
        x_train_principal = x_train
        x_test_principal = x_test
        pca_loadings = np.eye(source_n_vertecies)

    if use_pca_y:
        pca = PCA(n_components=n_components, copy=True, whiten=False, svd_solver='full')
        pca.fit(y_train) # requiring x_train's shape to be (n_samples, n_features)

        y_train = pca.transform(y_train)
        y_test = pca.transform(y_test) # shape = (n_sampl





    device='cuda' if use_gpu else 'cpu'
    backend_type = 'torch_cuda' if use_gpu else 'numpy'
    backend = set_backend(backend_type, on_error="warn")


    if regressor == 'ridge_svd':
        assert len(alpha) == 1
        print('use regressor: ridge_svd.  Ignoring beta and n_iter. \n using alpha: {} from {}'.format(alpha[0], alpha))
        y_predicted_train, y_predicted_test, coef_ = regression_svd(x_train_principal, y_train,
                                                             x_test_principal, device,
                                                             alpha=alpha[0])

    else:
        raise NotImplementedError('regressor should be ridge_svd')

    if normalize:
        # Denormalize predicted features
        y_predicted_test = y_predicted_test * norm_scale_y + norm_mean_y
        y_predicted_train = y_predicted_train * norm_scale_y + norm_mean_y
    if ret_coef:
        return y_predicted_train, y_train, y_predicted_test, y_test, pca_loadings, coef_
    else:
        return y_predicted_train, y_train, y_predicted_test, y_test, pca_loadings
