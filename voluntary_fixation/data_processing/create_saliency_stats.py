import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import pandas as pd
import deepgaze_pytorch
from scipy.stats import multivariate_normal
from voluntary_fixation.dataset.image_dataset import MovieDataset
from voluntary_fixation.envs import SAVE_ROOT, SEGMENT_MOVIE_DIR, DEVICE, RUN_IDS
from typing import List, Tuple


def pooling(x, pooling_size=8):
    x = torch.from_numpy(x).unsqueeze(0)
    x = F.avg_pool2d(x, pooling_size)
    return x


def get_average_coor(pred:torch.Tensor, target_width:int, target_height:int, k:float=3):
    pred_flat = pred.view(target_height*target_width)
    pred_mean = pred_flat.mean(dim=0)
    pred_sigma = pred_flat.std(dim=0)
    threhsold =  pred_mean + k * pred_sigma
    pred_large_idx = torch.where(pred_flat >threhsold)[0] # tensor([197744, 197745, 197746,  ..., 491105, 491106, 491107], device='cuda:2') torch.Size([18964])
    if pred_large_idx.shape[0] == 0:
        mean_coor = torch.tensor([target_width/2, target_height/2]).to(DEVICE)
    else:
        pred_large_idx_y = pred_large_idx // target_width
        pred_large_idx_x = pred_large_idx % target_width
        pred_large_coords = torch.cat([pred_large_idx_x.unsqueeze(1), pred_large_idx_y.unsqueeze(1)], dim=1) # n_pixels x 2 # torch.Size([18964, 2])
        pred_large_weight = pred[pred > threhsold].unsqueeze(-1) # n_pixels torch.Size([18964, 1])
        pred_large_weight /= torch.sum(pred_large_weight)
        mean_coor = torch.sum(pred_large_coords * pred_large_weight, dim=0)
    return mean_coor


def get_masked_squared_error(pred:torch.Tensor, pre_pred:torch.Tensor, target_width:int, target_height:int,  k:float=3, direction='both'):
    pre_pred_flat = pre_pred.view(target_height*target_width)
    pred_flat = pred.view(target_height*target_width)

    pre_pred_mean = pre_pred_flat.mean(dim=0)
    pre_pred_sigma = pre_pred_flat.std(dim=0)
    pre_threhsold =  pre_pred_mean + k * pre_pred_sigma
    pre_pred_large_idx = torch.where(pre_pred_flat >pre_threhsold)[0]

    masked_squared_error = torch.sqrt(torch.sum((pre_pred_flat[pre_pred_large_idx] - pred_flat[pre_pred_large_idx])**2))
    if direction == 'both':
        pred_mean = pred_flat.mean(dim=0)
        pred_sigma = pred_flat.std(dim=0)
        threhsold =  pred_mean + k * pred_sigma
        pred_large_idx = torch.where(pred_flat >threhsold)[0]

        masked_squared_error += torch.sqrt(torch.sum((pre_pred_flat[pred_large_idx] - pred_flat[pred_large_idx])**2))
    return masked_squared_error


def gauss2d(mu:List[int], sigma:List[float], w:int, h:int)->np.ndarray:

    x = np.arange(w)
    y = np.arange(h)

    x, y = np.meshgrid(x, y)

    x_ = x.flatten()
    y_ = y.flatten()
    xy = np.vstack((x_, y_)).T

    normal_rv = multivariate_normal(mu, sigma)
    z = normal_rv.pdf(xy)
    z = z.reshape(w, h, order='F')

    return z.T




def predict(dataset:torch.utils.data.Dataset, batch_size:int=20):
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE).eval()
    target_height = int(dataset.target_height)
    target_width = int(dataset.target_width)
    centerbias_template =  np.zeros((target_height, target_width))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pre_pred_of_first = gauss2d([np.floor(target_width/2), np.floor(target_height/2)],
                                [target_width, target_height], target_width, target_height)
    pre_pred_of_first = torch.tensor(pre_pred_of_first).to(DEVICE)

    record = {'idx':[], 'sq_error': [], 'masked_sq_error':[],
              'max_coor_x':[], 'max_coor_y': [], 'max_shift_norm': [], 'max_shift_x': [], 'max_shift_y': [],
              'avg_coor_x':[], 'avg_coor_y': [], 'avg_shift_norm': [], 'avg_shift_x': [], 'avg_shift_y': []}
    print('num_batches:: ', len(dataloader))
    for i, batch_image in tqdm(enumerate(dataloader)):
        # batch_image = batch_image.astype(np.uint8)
        centerbias = zoom(centerbias_template, (batch_image.shape[1]/centerbias_template.shape[0], batch_image.shape[2]/centerbias_template.shape[1]), order=0, mode='nearest')
        # renormalize log density
        centerbias -= logsumexp(centerbias)
        batch_tensors = torch.tensor(batch_image.permute(0, 3, 1, 2)).to(DEVICE) # torch.Size([20, 3, 720, 1280])
        centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)
        with torch.no_grad():
            log_density_prediction = model(batch_tensors, centerbias_tensor)

        preds = log_density_prediction.squeeze_()
        if preds.ndim == 2:
            preds = preds.unsqueeze(0)
        preds = torch.nn.functional.softmax(preds.view(batch_size, -1)).view(preds.shape)
        # pre_preds = torch.cat([pre_pred, pred], dim=0)
        for b in range(len(preds)):
            if b == 0:
                pre_pred = pre_pred_of_first
                pred = preds[b]
            else:
                pre_pred = preds[b-1]
                pred = preds[b]
            record['idx'].append(i*batch_size + b)
            # squared error
            sq_error = torch.sqrt(torch.sum((pred - pre_pred)**2))
            record['sq_error'].append(sq_error.item())
            # masked squared error
            masked_sq_error = get_masked_squared_error(pred, pre_pred, target_width, target_height, k=3, direction='both')
            record['masked_sq_error'].append(masked_sq_error.item())
            # avg of optical flow
            # flow = cv2.calcOpticalFlowFarneback(pre_pred.cpu().numpy(),pred.cpu().numpy(), None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # max shift
            idx_pre = torch.argmax(pre_pred)
            max_x_pre = idx_pre % target_width
            max_y_pre = idx_pre // target_width
            idx_next = torch.argmax(pred)
            max_x_next = idx_next % target_width
            max_y_next = idx_next // target_width
            max_shift_x = max_x_pre - max_x_next
            max_shift_y = max_y_pre - max_y_next
            max_shift = torch.sqrt((max_shift_x)**2 + (max_shift_y)**2)
            record['max_coor_x'].append(max_x_next.item()) #
            record['max_coor_y'].append(max_y_next.item()) #
            record['max_shift_norm'].append(max_shift.item()) #
            record['max_shift_x'].append(max_shift_x.item()) #
            record['max_shift_y'].append(max_shift_y.item()) #

            # avg shift
            pre_avg_coor = get_average_coor(pre_pred, target_width, target_height, k=3)# 2 x_coor, y_coor
            next_avg_coor = get_average_coor(pred, target_width, target_height, k=3)# 2
            avg_shift_x = pre_avg_coor[0] - next_avg_coor[0]
            avg_shift_y = pre_avg_coor[1] - next_avg_coor[1]
            avg_shift = torch.sqrt(torch.sum((pre_avg_coor - next_avg_coor)**2))
            record['avg_shift_norm'].append(avg_shift.item())
            record['avg_shift_x'].append(avg_shift_x.item())
            record['avg_shift_y'].append(avg_shift_y.item())
            record['avg_coor_x'].append(next_avg_coor[0].item())
            record['avg_coor_y'].append(next_avg_coor[1].item())

        pre_pred_of_first = preds[-1]

    return record







if __name__ == '__main__':

    run_ids = RUN_IDS #[1,2,3,4,5,6,7,8]
    # run_ids = [5]
    # run_ids = [7,8]
    # run_ids = [1,2,3]

    save_dir = os.path.join(SAVE_ROOT, 'saliency', 'deepgaze2e_predict')
    os.makedirs(save_dir, exist_ok=True)
    for run_id in run_ids:
        movie_path = os.path.join(SEGMENT_MOVIE_DIR, f'segment{run_id}.mkv')
        assert os.path.exists(movie_path), 'movie file not found: %s' % movie_path
        target_width = None # 1280
        target_height = None # 720

        dataset = MovieDataset(movie_path, target_width, target_height)
        record = predict(dataset, batch_size=20)
        record_df = pd.DataFrame(record)
        csv_path = os.path.join(save_dir, f'segment{run_id}.csv')
        record_df.to_csv(csv_path, index=False)
        print('save as ', csv_path)
