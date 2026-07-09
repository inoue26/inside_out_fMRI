import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from voluntary_fixation.envs import TR, MOVIE_FPS, EYEMOVE_ROOT, SUBJECT_IDS, RUN_IDS, SAVE_ROOT
import cv2
from voluntary_fixation.data_processing.create_mask import get_label_df


labels = ['PURS', 'FIXA']
# labels = ['SACC']



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
for fixation_lag_lower in [0.05, 0.1, 0.15, 0.2, 0.25]:#, 0.3, 0.35]:
    for duration in [0.1, 0.15, 0.2, 0.25, 0.3]:
        print('================================================')
        print('Lag: ', fixation_lag_lower, 'Duration: ', duration)
        fixation_lag_upper = fixation_lag_lower + duration
        sal_x_max = []
        sal_y_max = []
        sal_x_avg = []
        sal_y_avg = []
        fixa_x = []
        fixa_y = []
        for sbj in SUBJECT_IDS:
            for run_id in RUN_IDS:
                print('sbj: ', sbj, 'run: ', run_id)
                saliency_path = os.path.join(SAVE_ROOT, f'saliency/deepgaze2e_predict/segment{run_id}.csv')
                saliency_df = pd.read_csv(saliency_path)
                fixations_df = get_label_df(EYEMOVE_ROOT, sbj, run_id, labels)
                fixations_df['offset'] = fixations_df['onset'] + fixations_df['duration']
                for i, row in saliency_df.iterrows():
                    sal_onset = row['idx'] / MOVIE_FPS
                    fixation_start = sal_onset + fixation_lag_lower
                    fixation_end = sal_onset + fixation_lag_upper
                    fixations_with_sal = fixations_df[(fixations_df['offset'] >= fixation_start) & (fixations_df['offset'] <= fixation_end)]
                    if len(fixations_with_sal) == 0:
                        continue
                    sal_x_max.append(row['max_coor_x'])
                    sal_y_max.append(row['max_coor_y'])
                    sal_x_avg.append(row['avg_coor_x'])
                    sal_y_avg.append(row['avg_coor_y'])
                    fixa_x.append(fixations_with_sal['end_x'].mean())
                    fixa_y.append(fixations_with_sal['end_y'].mean())
        # print('FIX: ', len(fixa_y), 'SAL: ', len(sal_y))
        print('\t Max', np.corrcoef(fixa_x, sal_x_max)[0,1], np.corrcoef(fixa_y, sal_y_max)[0,1])
        print('\t Avg', np.corrcoef(fixa_x, sal_x_avg)[0,1], np.corrcoef(fixa_y, sal_y_avg)[0,1])
        corr_mean = np.mean([np.corrcoef(fixa_x, sal_x_max)[0,1], np.corrcoef(fixa_y, sal_y_max)[0,1]])
        corr_mean_avg = np.mean([np.corrcoef(fixa_x, sal_x_avg)[0,1], np.corrcoef(fixa_y, sal_y_avg)[0,1]])
        axes[0].plot([fixation_lag_lower, fixation_lag_upper], (corr_mean, corr_mean))
        axes[1].plot([fixation_lag_lower, fixation_lag_upper], (corr_mean_avg, corr_mean_avg))
axes[0].set_title('Max')
axes[1].set_title('Avg')

savefig_dir = os.path.join(SAVE_ROOT, 'behavior', 'saliency_fixation_lag')
os.makedirs(savefig_dir, exist_ok=True )
figpath = os.path.join(savefig_dir, 'all_sbj-all_run-{}.png'.format('_'.join(labels)))
plt.savefig(figpath)
print(figpath)
