import pandas as pd
import numpy as np
import os
from typing import List, Tuple
import cv2
import math
from tqdm import tqdm
from voluntary_fixation.envs import SEGMENT_MOVIE_DIR, SAVE_ROOT, TR, RUN_VOLUMES, MOVIE_WIDTH, MOVIE_HEIGHT, MOVIE_FPS

resolution = (MOVIE_WIDTH, MOVIE_HEIGHT) # width x height

def run(run_ids:List[int], time_resolution:int, start_offset:int, movie_fps:int, save_dir:str):
    for run_id in run_ids:
        npy_save_file = os.path.join(save_dir, f'segment{run_id}.npy')
        if os.path.exists(npy_save_file):
            print('already exists: ', npy_save_file)
            continue
        mkv_path = os.path.join(SEGMENT_MOVIE_DIR, f'segment{run_id}.mkv')
        assert os.path.exists(mkv_path), 'movie file not found: %s' % mkv_path
        cap = cv2.VideoCapture(mkv_path)
        print('============================================')
        print('RUN_ID: ', run_id)
        print('\tWIDTH: ', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('\tHEIGHT: ', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('\tFPS: ', cap.get(cv2.CAP_PROP_FPS))
        print('\tNUM_FRAMES: ', cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('============================================')
        target_frame_ids = list(range(int(start_offset*movie_fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(time_resolution*movie_fps)))
        # VOLUME とframeでずれる
        assert len(target_frame_ids) == RUN_VOLUMES[run_id-1] or len(target_frame_ids) == RUN_VOLUMES[run_id-1]+1 or len(target_frame_ids) == RUN_VOLUMES[run_id-1]-1, f'{len(target_frame_ids)} != {RUN_VOLUMES[run_id-1]}'
        frame_list = []
        target_length = RUN_VOLUMES[run_id-1] # if start_offset == 0 else RUN_VOLUMES[run_id-1] - 1
        for f_id in tqdm(target_frame_ids):
            if len(frame_list) > RUN_VOLUMES[run_id-1]-1: # 1-indexed
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_id)
            ret, frame = cap.read()
            assert ret, 'frame not found: %s' % f_id
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame)
        if len(frame_list) == target_length-1:
            frame_list.append(frame_list[-1])
            print('\nWARNING:: A dummy frame added. Eventually, this frame is excluded in analysis as overlapped frame.')
        frame_list = np.stack(frame_list, axis=0)
        assert len(frame_list) == target_length, f'{len(frame_list)} != {target_length}'
        np.save(npy_save_file, frame_list)
        print(f'shape: {frame_list.shape}\tsaved: ', npy_save_file)



if __name__ == '__main__':
    SAVE_ROOT_FORMAT = os.path.join(SAVE_ROOT, 'frames/resolution{time_resolution}-start{start}')
    run_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    labels = ['PURS', 'FIXA'] # SACC
    time_resolution = TR
    movie_fps =MOVIE_FPS
    start = 2 # [sec] if set0, each frame corresponds to the start of each TR(/ Slice Timing). if set 1, each frame corresponds to the middle of each TR(/ Slice Timing). if set2, each frame corresponds to the end of each TR(/ Slice Timing).

    save_dir = SAVE_ROOT_FORMAT.format(time_resolution=time_resolution, start=start)
    os.makedirs(save_dir, exist_ok=True)

    run(run_ids, time_resolution, start, movie_fps, save_dir)