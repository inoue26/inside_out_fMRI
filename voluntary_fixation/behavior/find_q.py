import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from voluntary_fixation.behavior.src.probability import prob_sacc_on_multiple_saliency
from voluntary_fixation.behavior.src.data_loader import read_csvs
from voluntary_fixation.envs import TR, RUN_IDS, SUBJECT_IDS, SAVE_ROOT, NUM_ROIS, EYEMOVE_ROOT, RUN_VOLUMES, MOVIE_FPS, MOVIE_HEIGHT, MOVIE_WIDTH



def data_processing(salieny_csv_root:str, original_eyetrack_root:str, gaze_shift_label:List[str]):
    saliency_csv_file_path_pattern = 'segment{run_id}.csv'
    original_eyetrack_path_pattern = 'sub-{sub}/sub-{sub}_task-movie_run-{run_id}_events.tsv'

    prob_dfs:List[pd.DataFrame] = []
    for sbj in tqdm(SUBJECT_IDS):
        for run_id in RUN_IDS:
            saliency_csv_file_path = os.path.join(salieny_csv_root, saliency_csv_file_path_pattern.format(run_id=run_id))
            original_eyetrack_path = os.path.join(original_eyetrack_root, original_eyetrack_path_pattern.format(sub=sbj, run_id=run_id))

            saliency_df, original_eyetrack_df = read_csvs(saliency_csv_file_path,
                                                            original_eyetrack_path, gaze_shift_label,
                                                            video_width=MOVIE_WIDTH,
                                                            video_height=MOVIE_HEIGHT, fps=MOVIE_FPS)
            # calc sacc probability on saliency
            prob_df = prob_sacc_on_multiple_saliency(saliency_df, original_eyetrack_df,
                                saliency_qs, gaze_shift_label, sacc_th_q=sacc_th_q, have_duration=False)

            prob_df['sbj'] = [sbj] * len(prob_df)
            prob_df['run_id'] = [run_id] * len(prob_df)

            prob_dfs.append(prob_df)
    prob_dfs = pd.concat(prob_dfs, axis=0)
    return prob_dfs


def plot_q(prob_dfs:pd.DataFrame, png_savepath:str):
    sal_labels = ['max_shift_norm', 'avg_shift_norm']
    sacc_labels = ['SACC_cnt', 'SACC_cnt_large']
    # all sbj :
    df_qs = prob_dfs['q'].unique()
    df_qs = np.sort(df_qs)

    xticks_q = [q for q in df_qs if not('ref' in q) ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for c, sal_label in enumerate(sal_labels):
        target_df = prob_dfs[prob_dfs['sal_label'] == sal_label]
        score_mean_qs = []
        score_std_qs = []
        for q in df_qs:
            q_df = target_df[target_df['q'] == q]
            if 'ref' in q:
                ref_scores = q_df.mean() # agg across sbj
                continue
            scores = q_df.mean()
            stds = q_df.std()
            score_mean_qs.append(scores)
            score_std_qs.append(stds)
        for r, sacc_label in enumerate(sacc_labels):
            means = [m[f'{sacc_label}_on_sal'] for m in score_mean_qs]
            stds = [m[f'{sacc_label}_on_sal'] for m in score_std_qs]

            axes[r, c].errorbar(xticks_q, means, yerr=stds, label=sal_label)
            axes[r,c].hlines(ref_scores[f'{sacc_label}_on_sal'], 0, len(xticks_q), linestyles='dashed', label='ref')
            axes[r,c].set_title(sacc_label)
            axes[r,c].set_xlabel('q')
            axes[r,c].set_ylabel(f'{sacc_label}_on_sal')
            axes[r,c].legend()
            axes[r,c].set_xticks(xticks_q)
    plt.savefig(png_savepath)
    print('save as ', png_savepath)



if __name__ == '__main__':

    salieny_csv_root = os.path.join(SAVE_ROOT, 'saliency', 'deepgaze2e_predict')# '../../../../results/srm/saliency_prediction2/deepgaze2e'
    original_eyetrack_root = EYEMOVE_ROOT
    gaze_shift_label =  ['FIXA', 'PURS'] # ['FIXA', 'PURS', 'SACC'] # ['SACC']

    save_root = os.path.join(SAVE_ROOT, 'behavior', 'find_q')# '../../results/srm/saliency_gaze/eyemove_saliency'
    os.makedirs(save_root, exist_ok=True)

    saliency_qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]# [0.5, 1, 2, 3, 4]
    colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y', 'orange', 'purple', 'brown']
    sacc_th_q = 0.5


    csv_path = os.path.join(save_root, f'gaze_shift_with_saliency-{"_".join(gaze_shift_label)}_q{sacc_th_q}.csv')
    prob_dfs = data_processing(salieny_csv_root, original_eyetrack_root, gaze_shift_label)
    prob_dfs.to_csv(csv_path, index=False)
    print('save as ', csv_path)

    png_savepath = os.path.join(save_root, f'gaze_shift_with_saliency-{"_".join(gaze_shift_label)}_q{sacc_th_q}.png')
    plot_q(prob_dfs, png_savepath)