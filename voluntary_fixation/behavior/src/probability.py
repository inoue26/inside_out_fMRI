import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from .temporal_alignment_filtering import sacc_cnt_shift_relation_df


def prob_sacc_on_multiple_saliency(saliency_df:pd.DataFrame, original_eyetrack_df:pd.DataFrame,
                                   saliency_qs:List[float], gaze_shift_label:List[str],
                                   sacc_th_q=0.5, have_duration:bool=False)->pd.DataFrame:
    """_summary_
    どの強さのsaliencyでsaccadeが起きる確率が変化するか算出する。
    Args:
        saliency_df (pd.DataFrame): _description_
        original_eyetrack_df (pd.DataFrame): _description_
        sacc_th_q (float, optional): _description_. Defaults to 0.5.

    Returns:
        pd.DataFrame: _description_
    """
    if have_duration:
        assert 'duration' in saliency_df.columns
        saliency_df['offset'] = saliency_df['onset'] + saliency_df['duration']
    else:
        print('saliency_df has no duration column')

    original_eyetrack_df['power'] = original_eyetrack_df['diff_x']**2 + original_eyetrack_df['diff_y']**2
    saliency_eyetrack_df = sacc_cnt_shift_relation_df(saliency_df, original_eyetrack_df, gaze_shift_label, sacc_th_q, have_duration=have_duration) # {'SACC_cnt','SACC_cnt_large', 'SACC_shift_vec_avg_x', 'SACC_shift_vec_avg_y', 'SACC_shift_norm'}
    assert len(saliency_eyetrack_df) == len(saliency_df)
    preset_columns = ['saliency_q_cnt', 'sal_label', 'q']

    sal_labels = ['max_shift_norm', 'avg_shift_norm']
    sacc_labels = ['SACC_cnt', 'SACC_cnt_large']

    ref_min_q = min(min(saliency_qs) , 0.1)
    if ref_min_q not in saliency_qs:
        saliency_qs.append(ref_min_q)
    preset_columns += sacc_labels
    ret_df = {col:[] for col in preset_columns}
    for i, sal_label in enumerate(sal_labels):
        saliency_quantiles = saliency_df[sal_label].quantile(saliency_qs)
        for q in saliency_qs:
            sal_q_df = saliency_eyetrack_df.query('{} > {}'.format(sal_label, saliency_quantiles[q]))
            ret_df['saliency_q_cnt'].append(len(sal_q_df))
            ret_df['sal_label'].append(sal_label)
            ret_df['q'].append(str(q))
            for sac_label in sacc_labels:
                # sal > qに誘引されたとされるsaccが一回でもあるかどうか
                ret_df[sac_label].append(len(sal_q_df.query(f'{sac_label} > 0')))
        # ref_
        sal_q_df = saliency_eyetrack_df.query('{} <= {}'.format(sal_label, saliency_quantiles[ref_min_q]))
        ret_df['saliency_q_cnt'].append(len(sal_q_df))
        ret_df['sal_label'].append(sal_label)
        ret_df['q'].append(f'ref:{ref_min_q}')
        for sac_label in sacc_labels:
            # sal <= qに誘引されたとされるsaccが一回でもあるかどうか
            ret_df[sac_label].append(len(sal_q_df.query(f'{sac_label} > 0')))


    ret_df = pd.DataFrame(ret_df)
    for sac_label in sacc_labels:
        ret_df[f'{sac_label}_on_sal'] = ret_df[sac_label] / ret_df['saliency_q_cnt']

    return ret_df





def sacc_ratio_by_saliency_shift_quantile_threshold(saliency_eyetrack_df:pd.DataFrame, qs:List[float])->dict:
    # qs　上位or下位の割合　 0.05, 0.1, 0.2, 0.3, 0.4, 0.5
    ret = {'label':[],'q':[], 'low_ratios':[], 'high_ratios':[], 'low_samples':[], 'high_samples':[]}
    for i, label in enumerate(['max_shift_norm', 'avg_shift_norm']):
        low_ratios = []
        high_ratios = []
        low_samples = []
        high_samples = []
        for q in qs:
            quantiles = saliency_eyetrack_df[label].quantile([q, 1-q])
            low_th = quantiles[q]
            high_th = quantiles[1-q]
            print(label, low_th)
            low_shift_SACC_cnt_ratio = saliency_eyetrack_df.query('{} <= {}'.format(label, low_th))['SACC_cnt_large'].mean()
            high_shift_SACC_cnt_ratio = saliency_eyetrack_df.query('{} >= {}'.format(label, high_th))['SACC_cnt_large'].mean()
            low_ratios.append(low_shift_SACC_cnt_ratio)
            high_ratios.append(high_shift_SACC_cnt_ratio)
            low_samples.append(len(saliency_eyetrack_df.query('{} <= {}'.format(label, low_th))))
            high_samples.append(len(saliency_eyetrack_df.query('{} >= {}'.format(label, high_th))))
        ret['q'] += qs
        ret['label'] += [label] * len(qs)
        ret['low_ratios'] += low_ratios # shift量が下位のSACC回数の平均
        ret['high_ratios'] += high_ratios # shift量が上位のSACC回数の平均
        ret['low_samples'] += low_samples # shift量が下位のサンプル数
        ret['high_samples'] += high_samples # shift量が上位のサンプル数

    return ret

def count_gaze_shift_with_saliency(saliency_eyetrack_df:pd.DataFrame, sal_q_upper:float):
    saliency_labels = ['max_shift_norm', 'avg_shift_norm']
    sacc_cnt = saliency_eyetrack_df['SACC_cnt'].to_numpy() > 0
    sacc_large_cnt = saliency_eyetrack_df['SACC_cnt_large'].to_numpy() > 0
    for i, label in enumerate(saliency_labels):
        th_upper = saliency_eyetrack_df[label].quantile(sal_q_upper)
        th_lower = saliency_eyetrack_df[label].quantile(1-sal_q_upper) #
        # sacc_cnt with upper saliency shift
        upper_indices = saliency_eyetrack_df[label].to_numpy() > th_upper
        sacc_with_upper_sal = sacc_cnt & upper_indices
        large_sacc_with_upper_sal = sacc_large_cnt & upper_indices
        # sacc_cnt with lower saliency shift
        lower_indices = saliency_eyetrack_df[label].to_numpy() > th_lower
        sacc_with_lower_sal = sacc_cnt & lower_indices
        large_sacc_with_lower_sal = sacc_large_cnt & lower_indices

        saliency_eyetrack_df[f'SACC_cnt_with_upper_sal_{label}'] = list(sacc_with_upper_sal)
        saliency_eyetrack_df[f'SACC_cnt_large_with_upper_sal_{label}'] = list(large_sacc_with_upper_sal)
        saliency_eyetrack_df[f'SACC_cnt_with_lower_sal_{label}'] = list(sacc_with_lower_sal)
        saliency_eyetrack_df[f'SACC_cnt_large_with_lower_sal_{label}'] = list(large_sacc_with_lower_sal)
    return saliency_eyetrack_df



# def sacc_with_saliency_shift(original_eyetrack_df:pd.DataFrame, saliency_df, qs:List[float],sal_th_q=0.5)->dict:
#     qs_ = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
#     ret = {'label':[],'q':[], 'low_ratios':[], 'high_ratios':[], 'low_samples':[], 'high_samples':[]}
#     for i, label in enumerate(['max_shift_norm', 'avg_shift_norm']):
#         low_ratios = []
#         high_ratios = []
#         low_samples = []
#         high_samples = []

#         saliency_quantiles = saliency_df[label].quantile([sal_th_q])
#         gaze_quantiles = original_eyetrack_df['power'].quantile(qs)
#         large_saliency_df = saliency_df.query('max_shift_norm > {}'.format(saliency_quantiles[sal_th_q]))
#         print('large saliency: ', len(large_saliency_df))
#         small_saliency_df = saliency_df.query('max_shift_norm == 0')
#         print('small saliency: ', len(small_saliency_df))
#         for q in qs:
#             large_saccade_df = original_eyetrack_df.query('power > {}'.format(gaze_quantiles[q]))
#             sacc_with_large_sal = []
#             sacc_with_small_sal = []
#             for i, iterrows in large_saccade_df.iterrows():
#                 onset = iterrows['onset']
#                 target_saliency_onset_start = onset - 0.4
#                 target_saliency_onset_end = onset - 0.25
#                 # 上位90%以上のsaliencyのendがこの間にあるかどうか
#                 sacc_before_large_saliency_df = large_saliency_df.query('{} < offset < {}'.format(target_saliency_onset_start, target_saliency_onset_end))
#                 if  len(sacc_before_large_saliency_df) > 0:
#                     # print(f'onset: {onset} size', len(sacc_before_large_saliency_df))
#                     sacc_with_large_sal.append(1)
#                 sacc_with_large_sal.append(0)
#                 sacc_before_small_saliency_df = small_saliency_df.query('{} < offset < {}'.format(target_saliency_onset_start, target_saliency_onset_end))
#                 if  len(sacc_before_small_saliency_df) > 0:
#                     # print(f'onset: {onset} size', len(sacc_before_small_saliency_df))
#                     sacc_with_small_sal.append(1)
#                 sacc_with_small_sal.append(0)
#             sacc_with_large_sal = np.array(sacc_with_large_sal)
#             sacc_with_small_sal = np.array(sacc_with_small_sal)
#             ret['q'].append(q)
#             ret['label'].append(label)
#             ret['low_ratios'].append(np.mean(sacc_with_small_sal)) # shift量が下位のSACC回数の平均
#             ret['high_ratios'].append(np.mean(sacc_with_large_sal)) # shift量が上位のSACC回数の平均
#             ret['low_samples'].append(np.sum(sacc_with_small_sal==1)) # shift量が下位のサンプル数
#             ret['high_samples'].append(np.sum(sacc_with_large_sal==1)) # shift量が上位のサンプル数

#     return ret