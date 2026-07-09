import os
# import glob

from preprocessing_utils import preprocess
from paths import DATASETS_OUT, BOLD_DIR

TR = 2.0


def main():
    base_dir = DATASETS_OUT
    os.makedirs(BOLD_DIR, exist_ok=True)
    run_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    # run_ids = [1]

    groups = [1]
    # avoid participant 5, 9, 20 for functional data and 1 if using physiological data (unavailable)
    # subject_ids_grouped = [['01']]
    # subjct_ids_grouped = [['01', '02', '03', '04', '05', '06', '09', '10', '14', '15', '16', '17', '18', '19', '20']]
    subject_ids_grouped = [['01', '02', '03', '04', '06', '10', '14', '15', '16', '17', '18', '19']]

    for group, subject_ids in zip(groups, subject_ids_grouped):
        group_dir = os.path.join(base_dir, 'studyforrest-data-all_out', 'fmriprep-latest', 'fmriprep')
        for subject_id in subject_ids:
            subject_dir = os.path.join(group_dir, f'sub-{subject_id}', 'ses-movie', 'func')
            for run_id in run_ids:

                confound = os.path.join(subject_dir, f'sub-{subject_id}_ses-movie_task-movie_run-{run_id}_desc-confounds_timeseries.tsv')
                bold = os.path.join(subject_dir, f'sub-{subject_id}_ses-movie_task-movie_run-{run_id}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')

                output_bold = os.path.join(BOLD_DIR, f'sub-{subject_id}_run-{run_id}_space-MNI152NLin2009cAsym_bold.nii.gz')

                print('preprocessing:', output_bold)
                _ = preprocess(bold, confound, TR, output_bold=output_bold)


if __name__ == '__main__':
    main()
