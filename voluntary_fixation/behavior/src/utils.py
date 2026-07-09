import numpy as np
import pandas as pd
from typing import List, Tuple

PAPER_FPS = 25
VIDEO_FPS = 30


def filter_saliency_df_paper_video_common_timing(saliency_df:pd.DataFrame, fps1:float=None, fps2:float=None)->Tuple[pd.DataFrame, int]:
    if fps1 is None:
        fps1 = PAPER_FPS
    if fps2 is None:
        fps2 = VIDEO_FPS

    new_FPS = np.gcd(fps2, fps1)
    video2common_interval = fps2 / new_FPS
    print(f'{fps2}(VIDEO)fps, {fps1}(PAPER)fps, {new_FPS}(new)fps, video2common_interval:{video2common_interval}')
    old_saliency_df_len = len(saliency_df)
    common_start_idx = range(0, old_saliency_df_len, int(video2common_interval))

    # merge saliency_df to common timing
    common_saliency_df_columns = ['max_coor_x', 'max_coor_y', 'avg_coor_x', 'avg_coor_y']
    common_saliency_df_columns_shift = ['max_shift_x', 'max_shift_y', 'avg_shift_x', 'avg_shift_y']
    other_columns = ['onset', 'duration']
    common_saliency_df = {col:[] for col in common_saliency_df_columns+common_saliency_df_columns_shift+other_columns}
    for idx in common_start_idx:
        middle_idx = min(int(idx + video2common_interval/2), old_saliency_df_len-1)
        for col in common_saliency_df_columns:
            common_saliency_df[col].append(saliency_df[col].iloc[middle_idx])
        end_idx = min(int(idx + video2common_interval), old_saliency_df_len-1)
        for col in common_saliency_df_columns_shift:
            assert 'shift' in col, f'{col}'
            common_saliency_df[col].append(saliency_df[col.replace('shift', 'coor')].iloc[end_idx] - saliency_df[col.replace('shift', 'coor')].iloc[idx])
        common_saliency_df['onset'].append(idx / fps2)
        common_saliency_df['duration'].append(1 / new_FPS)
    common_saliency_df = pd.DataFrame(common_saliency_df)

    common_saliency_df['max_shift_norm'] = np.sqrt(common_saliency_df['max_shift_x']**2 + common_saliency_df['max_shift_y']**2)
    common_saliency_df['avg_shift_norm'] = np.sqrt(common_saliency_df['avg_shift_x']**2 + common_saliency_df['avg_shift_y']**2)
    return common_saliency_df, new_FPS