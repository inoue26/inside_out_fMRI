import numpy as np
import os
import pandas as pd
from voluntary_fixation.envs import TR, SUBJECT_IDS, RUN_IDS, SAVE_ROOT, RUN_VOLUMES
import torch
import os
from torchmetrics.classification import BinaryJaccardIndex
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from typing import List, Tuple

BICUBIC = InterpolationMode.BICUBIC


def metric_binary_iou(pred:torch.Tensor, gt:torch.Tensor, threshold=0.5, device='cuda'):
    # pred, gt: (1, 720, 1280)
    return BinaryJaccardIndex(threshold=threshold).to(device)(pred, gt)

def kl_div(dist1:np.ndarray, dist2:np.ndarray):
    kl_div = np.sum(dist1*np.log(dist1/dist2))
    return kl_div

def run(sbj_id:str, run_id:int, time_resolution:int, offset:int, metric:str='iou')->Tuple[List[int], List[int], List[float]]:
    # get mask
    mask_path = os.path.join(SAVE_ROOT, 'mask', f'resolution{time_resolution}-start{offset}', f'mask_sub-{sbj_id}_run-{run_id}.npy')
    masks = np.load(mask_path) # n_samples x 720 x 1280
    masks = torch.from_numpy(masks.astype(np.float32)).unsqueeze(1).cuda() # n_samples x 1 x 720 x 1280

    pre_mask = None
    ious = []
    idx_in_run = []
    idx_in_exp = []
    idx_offsets = np.cumsum([0] + RUN_VOLUMES)
    for i, mask in enumerate(masks):
        idx_offset = idx_offsets[run_id-1]
        idx_in_run.append(i)
        idx_in_exp.append(i + idx_offset)
        # iou
        if pre_mask is None:
            ious.append(np.nan)
            pre_mask = mask
        else:
            if metric=='iou':
                iou = metric_binary_iou(pre_mask, mask, device='cuda')
                ious.append(iou.cpu().numpy())
            elif metric=='kl':
                dist1 = mask + 1e-6
                dist1 = dist1 / dist1.sum()
                dist2 = pre_mask + 1e-6
                dist2 = dist2 / dist2.sum()
                kl = kl_div(dist1.cpu().numpy(), dist2.cpu().numpy())
                ious.append(kl)
            else:
                raise ValueError('metric should be iou or kl')
            pre_mask = mask # これを忘れていた
    # df = pd.DataFrame({'idx_in_run':idx_in_run, 'idx_in_exp':idx_in_exp, 'iou':ious})
    return idx_in_run, idx_in_exp, ious


if __name__ == '__main__':
    time_resolution = TR
    offset = 0
    metric = 'iou'
    save_dir = os.path.join(SAVE_ROOT, 'mask', f'resolution{time_resolution}-start{offset}', metric)
    os.makedirs(save_dir, exist_ok=True)
    for sbj_id in SUBJECT_IDS:
        print('sbj: ', sbj_id)
        ious_sbj = []
        idx_in_run_sbj = []
        idx_in_exp_sbj = []
        runs = []
        for run_id in tqdm(RUN_IDS):
            idx_in_run, idx_in_exp, ious = run(sbj_id, run_id, time_resolution, offset, metric=metric)
            idx_in_run_sbj+=idx_in_run
            idx_in_exp_sbj+=idx_in_exp
            ious_sbj+=ious
            runs += [run_id]*len(ious)
        df = pd.DataFrame({'idx_in_run':idx_in_run_sbj, 'idx_in_exp':idx_in_exp_sbj, 'iou':ious_sbj, 'run_id': runs})
        csv_path = os.path.join(save_dir, f'sub-{sbj_id}.csv')
        df.to_csv(csv_path)
        print('save as ', csv_path)