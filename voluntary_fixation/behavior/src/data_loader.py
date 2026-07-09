import pandas as pd
from typing import List, Tuple
import numpy as np

def filter_eyetrack(eyetrack:pd.DataFrame, events:list)->pd.DataFrame:
    eyetrack = eyetrack[eyetrack['label'].isin(events)]
    return eyetrack


def make_vector(eyetrack:pd.DataFrame)->pd.DataFrame:
    eyetrack['diff_x'] = eyetrack['end_x'] - eyetrack['start_x']
    eyetrack['diff_y'] = eyetrack['end_y'] - eyetrack['start_y']
    eyetrack['velocity_x'] = eyetrack['diff_x'] / eyetrack['duration']
    eyetrack['velocity_y'] = eyetrack['diff_y'] / eyetrack['duration']
    return eyetrack


def read_csvs(saliency_csv_file_path:str, original_eyetrack_path:str, target_events:List[str],
              fps:float=None, video_width:int=None, video_height:int=None,
              )->Tuple[pd.DataFrame|None, pd.DataFrame|None, pd.DataFrame|None]:
    # read saliency csv
    if saliency_csv_file_path is not None:
        saliency_df = pd.read_csv(saliency_csv_file_path)
        saliency_df['onset'] = saliency_df['idx'] / fps
        pre_max_coor_x = [int(video_width/2)] + saliency_df['max_coor_x'].tolist()[:-1]
        saliency_df['max_shift_x'] = saliency_df['max_coor_x'] - pre_max_coor_x
        pre_max_coor_y = [int(video_height/2)] + saliency_df['max_coor_y'].tolist()[:-1]
        saliency_df['max_shift_y'] = saliency_df['max_coor_y'] - pre_max_coor_y
        saliency_df['max_shift_norm'] = np.sqrt(saliency_df['max_shift_x']**2 + saliency_df['max_shift_y']**2)
    else:
        saliency_df = None

    # read original eyetrack
    if original_eyetrack_path is not None:
        original_eyetrack_df = pd.read_table(original_eyetrack_path)
        original_eyetrack_df = filter_eyetrack(original_eyetrack_df, target_events)
        original_eyetrack_df = make_vector(original_eyetrack_df)
    else:
        original_eyetrack_df = None

    return saliency_df, original_eyetrack_df

