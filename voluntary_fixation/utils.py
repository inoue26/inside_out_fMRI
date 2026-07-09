import numpy as np


def calculate_q(p_seq): # FDR correction
    p_arr_   = np.asarray(p_seq).copy()
    p_arr = p_arr_[~np.isnan(p_arr_)]# np.nan_to_num(p_arr, nan=1)
    N       = len(p_arr)
    idx_arr = np.argsort(p_arr)
    q_arr   = p_arr[idx_arr] * N / (np.arange(N) + 1)
    q_arr   = np.minimum.accumulate(q_arr[::-1])[::-1]
    q_arr[idx_arr] = q_arr.copy()
    q_arr
    p_arr_[~np.isnan(p_arr_)] = q_arr
    q_arr = p_arr_
    return q_arr
