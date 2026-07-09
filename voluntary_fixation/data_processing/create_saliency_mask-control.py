import pandas as pd
import numpy as np
import os
from typing import List, Tuple
import cv2
import math
from tqdm import tqdm
from voluntary_fixation.envs import MOVIE_FPS, SAVE_ROOT, TR, RUN_VOLUMES, MOVIE_WIDTH, MOVIE_HEIGHT, COLOR_AREA_Y, SUBJECT_IDS, RUN_IDS
from voluntary_fixation.data_processing.create_mask import extract_coord_timing, circle, draw_eyetracking

resolution = (MOVIE_WIDTH, MOVIE_HEIGHT) # width x height

def extract_coord_timing(eyetrack:pd.DataFrame)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    onset = eyetrack['onset'].values
    duration = eyetrack['duration'].values
    start_x = eyetrack['start_x'].values
    start_y = eyetrack['start_y'].values
    end_x = eyetrack['end_x'].values
    end_y = eyetrack['end_y'].values
    xs = np.stack([start_x, end_x], axis=1)
    ys = np.stack([start_y, end_y], axis=1) # add offset  0 means top of colored area in a frame
    return onset, duration, xs, ys


def draw_eyetracking(onset:np.ndarray, duration:np.ndarray, xs:np.ndarray, ys:np.ndarray, time_resolution:int,
                     start_time:int, end_time:int)->np.ndarray:
    canvas_list = []
    circle_kernel = circle(101).astype(np.uint8)
    for t in range(start_time, end_time, time_resolution):
        target_indices_lower = onset >= t
        target_indices_upper = (onset + duration) < time_resolution + t
        target_indices = np.where(target_indices_lower & target_indices_upper)[0]

        canvas = np.zeros((resolution[1], resolution[0]), dtype=np.uint8)
        # draw eyetracking
        for idx in target_indices:
            x = xs[idx]
            y = ys[idx]
            cv2.line(canvas, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), color=1, thickness=1)
        # dilation with circle whose radius is 100
        canvas = cv2.dilate(canvas, circle_kernel, iterations = 1)
        # boolean for memory efficiency
        canvas = canvas.astype(bool)
        canvas_list.append(canvas)
    return np.stack(canvas_list, axis=0) # n_frame x height x width

def convert_saliency_df(saliency_df:pd.DataFrame, label:str)->pd.DataFrame:
    onset = saliency_df['idx'].values / MOVIE_FPS # sec
    end_x = saliency_df[label+'_coor_x'].to_list()
    end_y = saliency_df[label+'_coor_y'].to_list()
    start_x = [MOVIE_WIDTH/2] + end_x[:-1]
    start_y = [MOVIE_HEIGHT/2] + end_y[:-1]
    duration = [1/MOVIE_FPS] * len(onset)
    return pd.DataFrame({'onset':onset, 'duration':duration, 'start_x':start_x, 'start_y':start_y, 'end_x':end_x, 'end_y':end_y})


def run(saliency_dir:str, run_ids:List[int], label:str, save_dir:str, time_resolution:int, start:int)->pd.DataFrame:
    mask_file_pattern = 'mask_run-{run_id}.npy'
    for run_id in tqdm(run_ids):
        mask_path = os.path.join(save_dir, mask_file_pattern.format(run_id=run_id))
        if os.path.exists(mask_path):
            print('already exists: ', mask_path)
            continue
        # load eyetrack data
        saliency_df = pd.read_csv(os.path.join(saliency_dir, f'segment{run_id}.csv'))
        # add: onset, duration,start_x, start_y, end_x, end_y
        saliency_df = convert_saliency_df(saliency_df, label)
        # get corrd and timing
        onset, duration, xs, ys = extract_coord_timing(saliency_df)
        # validation
        assert onset.max() <= (RUN_VOLUMES[run_id-1] * TR + 2), f'should {onset.max()}<= {RUN_VOLUMES[run_id-1] * TR}'
        # draw eyetracking
        canvases = draw_eyetracking(onset, duration, xs, ys, time_resolution, start, RUN_VOLUMES[run_id-1] * TR)

        np.save(mask_path, canvases)
        print(f'shape: {canvases.shape}, save as ', mask_path)




if __name__ == '__main__':
    saliency_dir = os.path.join(SAVE_ROOT, 'saliency', 'deepgaze2e_predict')
    SAVE_ROOT_FORMAT = os.path.join(saliency_dir, 'mask', 'resolution{time_resolution}-start{start}')
    sbjs = SUBJECT_IDS # ['01', '02', '03', '04', '06', '10', '14', '15', '16', '17', '18', '19']
    run_ids = RUN_IDS # [1, 2, 3, 4, 5, 6, 7, 8]
    label = 'max' # avg
    time_resolution = 2
    start = 0

    save_dir = SAVE_ROOT_FORMAT.format(time_resolution=time_resolution, start=start)
    os.makedirs(save_dir, exist_ok=True)
    run(saliency_dir, run_ids, label, save_dir, time_resolution, start)