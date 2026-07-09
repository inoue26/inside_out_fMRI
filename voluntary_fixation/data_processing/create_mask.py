import pandas as pd
import numpy as np
import os
from typing import List, Tuple
import cv2
import math
from tqdm import tqdm
from voluntary_fixation.envs import EYEMOVE_ROOT, SAVE_ROOT, TR, RUN_VOLUMES, MOVIE_WIDTH, MOVIE_HEIGHT, COLOR_AREA_Y, SUBJECT_IDS, RUN_IDS

resolution = (MOVIE_WIDTH, MOVIE_HEIGHT) # width x height

def extract_coord_timing(eyetrack:pd.DataFrame)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    onset = eyetrack['onset'].values
    duration = eyetrack['duration'].values
    start_x = eyetrack['start_x'].values
    start_y = eyetrack['start_y'].values
    end_x = eyetrack['end_x'].values
    end_y = eyetrack['end_y'].values
    xs = np.stack([start_x, end_x], axis=1)
    ys = np.stack([start_y, end_y], axis=1) + COLOR_AREA_Y[0] # add offset  0 means top of colored area in a frame
    return onset, duration, xs, ys

def get_label_df(eyetrack_dir, sbj:str, run_id:int, label:List[str])->pd.DataFrame:
    eyetrack_fn = f'sub-{sbj}/sub-{sbj}_task-movie_run-{run_id}_events.tsv'
    eyetrack_path = os.path.join(eyetrack_dir, eyetrack_fn)
    eyetrack = pd.read_table(eyetrack_path)
    eyetrack = eyetrack[eyetrack['label'].isin(label)]
    return eyetrack

def circle(r):
    c = math.ceil(r - 1)
    s = c * 2 + 1
    return np.clip(r - np.sqrt(np.sum((np.stack((
        np.tile(np.arange(s), (s, 1)),
        np.repeat(np.arange(s), s).reshape((-1, s))
    )) - c) ** 2, axis=0)), 0, 1)

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


def run(eyetrack_dir:str, sbjs:List[str], run_ids:List[int], labels:List[str], save_dir:str, time_resolution:int, start:int)->pd.DataFrame:
    mask_file_pattern = 'mask_sub-{sbj}_run-{run_id}.npy'
    for sbj in sbjs:
        for run_id in tqdm(run_ids):
            mask_path = os.path.join(save_dir, mask_file_pattern.format(sbj=sbj, run_id=run_id))
            if os.path.exists(mask_path):
                print('already exists: ', mask_path)
                continue
            # load eyetrack data
            eyetrack = get_label_df(eyetrack_dir, sbj, run_id, labels)
            # get corrd and timing
            onset, duration, xs, ys = extract_coord_timing(eyetrack)
            # validation
            assert onset.max() <= (RUN_VOLUMES[run_id-1] * TR + 2), f'should {onset.max()}<= {RUN_VOLUMES[run_id-1] * TR}'
            # draw eyetracking
            canvases = draw_eyetracking(onset, duration, xs, ys, time_resolution, start, RUN_VOLUMES[run_id-1] * TR)

            np.save(mask_path, canvases)
            print(f'shape: {canvases.shape}, save as ', mask_path)




if __name__ == '__main__':
    SAVE_ROOT_FORMAT = os.path.join(SAVE_ROOT, 'mask/resolution{time_resolution}-start{start}')
    sbjs = SUBJECT_IDS # ['01', '02', '03', '04', '06', '10', '14', '15', '16', '17', '18', '19']
    run_ids = RUN_IDS # [1, 2, 3, 4, 5, 6, 7, 8]
    labels = ['PURS', 'FIXA'] # SACC
    time_resolution = 2
    start = 0

    save_dir = SAVE_ROOT_FORMAT.format(time_resolution=time_resolution, start=start)
    os.makedirs(save_dir, exist_ok=True)
    run(EYEMOVE_ROOT, sbjs, run_ids, labels, save_dir, time_resolution, start)