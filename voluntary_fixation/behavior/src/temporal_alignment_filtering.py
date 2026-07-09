import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from voluntary_fixation.envs import TR, RUN_VOLUMES



def convert_idx(src_idx, src_fps, tgt_fps):
    return np.floor(src_idx * tgt_fps / src_fps)

def idx2sec(idx, fps):
    return np.floor(idx / fps)

def find_idx_of_specific_onset(df, onset)->np.ndarray:
    onsets = df['onset'].values
    offsets = df['onset'] + df['duration']
    idx = np.where((onsets <= onset) & (offsets > onset))[0]
    return idx

def find_idx_of_specific_onset_offset(df, start, end, offset=False)->np.ndarray:
    onsets = df['onset'].values
    if offset:
        offsets = df['onset'] + df['duration']
        idx = np.where((start <= offsets) & (offsets<=end))[0]
        # idx = np.where((start <= onsets) & (offsets<=end))[0]

    else:
        idx = np.where((start <= onsets) & (onsets<=end))[0]
    return idx


def sacc_cnt_shift_relation_df(saliency_df:pd.DataFrame, original_eyetrack_df:pd.DataFrame,
                               gaze_shift_label: List[str],
                               sacc_th_q=0.5, have_duration:bool=False)->pd.DataFrame:

    SACC_df = original_eyetrack_df.query(f'label in {gaze_shift_label}')
    ssacc_quantiles = SACC_df['power'].quantile([sacc_th_q])
    # saliency_dfのonsetを基準にしてSACCが起きたかどうかをカウント
    saliency_eyetrack_df = {'SACC_cnt':[],'SACC_cnt_large':[], 'SACC_shift_vec_avg_x':[],
                            'SACC_shift_vec_avg_y':[], 'SACC_shift_norm':[]}
    if gaze_shift_label ==  ['SACC']:# [SACC]
        gaze_shift_lag_start = 0.05
        gaze_shift_lag_end = 0.15
    elif ('FIXA' in gaze_shift_label or 'PURS' in gaze_shift_label):
        if ('SACC' in gaze_shift_label):# [SACC, FIXA, PURS]
            gaze_shift_lag_start = 0.05
            gaze_shift_lag_end = 0.4
        else:# [FIXA, PURS]
            gaze_shift_lag_start = 0.1
            gaze_shift_lag_end = 0.4
    else:
        raise ValueError('gaze_shift_label should be either SACC, FIXA, PURS or a combination of them')

    for i,row in saliency_df.iterrows():
        # onset = row['onset'] + row['duration'] + 0.25 # David E. Warren, 2013  250msec latency from stimulus   # original:
        onset = row['onset'] + gaze_shift_lag_start# 5
        if have_duration:
            next_onset = row['onset'] + row['duration'] + gaze_shift_lag_end # David E. Warren, 2013  400msec latency from stimulus   # original: + 1/video_fps
        else:
            next_onset = row['onset']  + gaze_shift_lag_end
        original_eyetrack_idx = find_idx_of_specific_onset_offset(original_eyetrack_df, onset, next_onset, offset=True)
        # print('SACC cnt', len(original_eyetrack_idx))
        SACC_cnt = len(original_eyetrack_idx)
        if SACC_cnt == 0:
            SACC_shift_vec_avg_x = 0
            SACC_shift_vec_avg_y = 0
            SACC_shift_norm = 0
            SACC_cnt_large = 0
        else:
            SACC_shift_vec = original_eyetrack_df.iloc[original_eyetrack_idx][['diff_x', 'diff_y']]
            SACC_shift_vec_avg = SACC_shift_vec.mean()
            SACC_shift_vec_avg_x = SACC_shift_vec_avg['diff_x']
            SACC_shift_vec_avg_y = SACC_shift_vec_avg['diff_y']
            SACC_shift_norm = np.sqrt(SACC_shift_vec_avg_x**2 + SACC_shift_vec_avg_y**2)
            if SACC_shift_norm**2 > ssacc_quantiles[sacc_th_q]:
                SACC_cnt_large = 1
            else:
                SACC_cnt_large = 0

        saliency_eyetrack_df['SACC_cnt'].append(SACC_cnt)
        saliency_eyetrack_df['SACC_shift_vec_avg_x'].append(SACC_shift_vec_avg_x)
        saliency_eyetrack_df['SACC_shift_vec_avg_y'].append(SACC_shift_vec_avg_y)
        saliency_eyetrack_df['SACC_shift_norm'].append(SACC_shift_norm)
        saliency_eyetrack_df['SACC_cnt_large'].append(SACC_cnt_large)

    saliency_eyetrack_df = pd.DataFrame(saliency_eyetrack_df)
    saliency_eyetrack_df = pd.concat([saliency_df, saliency_eyetrack_df], axis=1)

    return saliency_eyetrack_df


def alignment2VOLUME(df:pd.DataFrame, run_id:int, agg:str='sum')->pd.DataFrame:
    num_sample_per_run = RUN_VOLUMES[run_id-1]
    columns = list(df.columns)
    new_df = {col:[] for col in columns}
    for i in range(num_sample_per_run):
        onset = i * TR
        offset = onset + TR
        idx = find_idx_of_specific_onset_offset(df, onset, offset)
        if agg == 'sum':
            agg_df = df.iloc[idx].sum()
        elif agg == 'mean':
            agg_df = df.iloc[idx].mean()
        else:
            raise ValueError('agg should be sum or mean')
        for col in columns:
            new_df[col].append(agg_df[col])
    new_df = pd.DataFrame(new_df)
    return new_df