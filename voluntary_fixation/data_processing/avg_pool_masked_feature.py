import argparse, os
import numpy as np
import cv2
from voluntary_fixation.envs import SAVE_ROOT, RUN_VOLUMES, SUBJECT_IDS, RUN_IDS, TR
from tqdm import tqdm
from typing import Tuple
import random
random.seed(1126)

def masked_avg_pooling(feature:np.ndarray, mask:np.ndarray)->Tuple[np.ndarray, bool]:
    # exclude redundant patch
    image_feat = feature[:, 1:, :] # num_target_hidden_states x num_patches x embedding_dims
    resized_mask = cv2.resize(mask.astype(np.int8), (16, 16), interpolation=cv2.INTER_NEAREST) # (16, 16) # binary mask
    resized_mask_tmp = resized_mask.flatten()
    target_indices = np.where(resized_mask_tmp>0)[0] # binary mask

    if target_indices.shape[0] == 0: # when frame has no fixation.
        print('WARNING: target_indices.shape[0] == 0. Take Global Avg Pool')
        image_feat = image_feat.mean(keepdims=True, axis=1) # Avg Pool : num_layer x 256 x embedding_dims -> num_layer x 1 x embedding_dims
        flag = False
    else:
        image_feat = image_feat[:, target_indices, :].mean(keepdims=True, axis=1) # Avg Pool : num_layer x 256 x embedding_dims -> num_layer x 1 x embedding_dims
        flag = True
    assert image_feat.shape == (feature.shape[0], 1, feature.shape[2])
    return image_feat, flag # num_layer x 1 x embedding_dims


def run(sbj_id:str, run_id:int, hidden_state:bool, modality:str, time_resolution:int, frame_offset:int, mask_offset:int):
    if modality in ['image' , 'whole_image', 'sampled_whole_image']:
        feature_dir =  os.path.join(SAVE_ROOT, 'features', 'image' + f'-tr{TR}-fo{frame_offset}-mo{mask_offset}')
    elif modality in ['masked_image', 'both', 'reverse_masked_image', 'saliency_masked_image',
                      'sampled_reverse_masked_image', 'shuffled_masked_image', 'shuffled_masked_image2', 'previous_masked_image',
                      'shift_reverse_masked_image','shift_reverse_masked_image025','shift_reverse_masked_image075', 'shift_reverse_masked_image_shuffle']:
        feature_dir =  os.path.join(SAVE_ROOT, 'features', modality + f'-tr{TR}-fo{frame_offset}-mo{mask_offset}', sbj_id)
    else:
        raise ValueError('modality must be image, masked_image, reverse_masked_image, or both')
    if hidden_state:
        feature_path = os.path.join(feature_dir, f'fg_b_{run_id}-hs.npy')
    else:
        feature_path = os.path.join(feature_dir, f'fg_b_{run_id}.npy')

    masks_path = os.path.join(SAVE_ROOT, 'mask', f'resolution{time_resolution}-start{mask_offset}',f'mask_sub-{sbj_id}_run-{run_id}.npy')
    if modality == 'saliency_masked_image':
        masks_path = os.path.join(SAVE_ROOT, 'saliency', 'deepgaze2e_predict', 'mask',
                                  f'resolution{time_resolution}-start{mask_offset}', f'mask_run-{run_id}.npy')
    elif modality == 'sampled_reverse_masked_image':
        masks_path =  os.path.join(SAVE_ROOT, 'sampled_reverse_mask', f'resolution{time_resolution}-start{mask_offset}',f'mask_sub-{sbj_id}_run-{run_id}.npy')
    elif modality == 'shift_reverse_masked_image':
        masks_path =  os.path.join(SAVE_ROOT, 'shift_reverse_mask', f'resolution{time_resolution}-start{mask_offset}',f'mask_sub-{sbj_id}_run-{run_id}.npy')
    elif modality == 'shift_reverse_masked_image025':
        masks_path =  os.path.join(SAVE_ROOT, 'shift_reverse_mask025', f'resolution{time_resolution}-start{mask_offset}',f'mask_sub-{sbj_id}_run-{run_id}.npy')
    elif modality == 'shift_reverse_masked_image075':
        masks_path =  os.path.join(SAVE_ROOT, 'shift_reverse_mask075', f'resolution{time_resolution}-start{mask_offset}',f'mask_sub-{sbj_id}_run-{run_id}.npy')
    elif modality == 'shift_reverse_masked_image_shuffle':
        masks_path =  os.path.join(SAVE_ROOT, 'shift_reverse_mask_shuffle', f'resolution{time_resolution}-start{mask_offset}',f'mask_sub-{sbj_id}_run-{run_id}.npy')

    elif modality == 'shuffled_masked_image':
        masks_path =  os.path.join(SAVE_ROOT, 'shuffled_mask', f'resolution{time_resolution}-start{mask_offset}',f'mask_sub-{sbj_id}_run-{run_id}.npy')
    elif modality == 'shuffled_masked_image2':
        masks_path =  os.path.join(SAVE_ROOT, 'shuffled_mask2', f'resolution{time_resolution}-start{mask_offset}',f'mask_sub-{sbj_id}_run-{run_id}.npy')
    elif modality == 'previous_masked_image':
        masks_path =  os.path.join(SAVE_ROOT, 'previous_mask', f'resolution{time_resolution}-start{mask_offset}',f'mask_sub-{sbj_id}_run-{run_id}.npy')

    features = np.load(feature_path) # n_samples x num_target_hidden_states x 257 x 1408 or n_samples x 257 x 1408
    masks = np.load(masks_path) # n_samples x height x width
    if modality == 'whole_image':
        masks = np.ones_like(masks).astype(np.bool)
    if modality == 'reverse_masked_image':
        masks = ~masks
    if modality == 'sampled_whole_image':
        new_masks = np.zeros_like(masks).astype(np.uint8)
        for m, mask in enumerate(masks):
            num_non_mask = np.sum(mask == 1) # eye fixation
            num_patches = mask.shape[0] * mask.shape[1]
            sampled_mask_indices = random.sample(range(num_patches), num_non_mask)
            new_mask = np.zeros(num_patches).astype(np.uint8)
            new_mask[sampled_mask_indices] = 1
            new_masks[m] = new_mask.reshape(mask.shape)
        masks = new_masks.astype(np.uint8)
    if not hidden_state:
        features = features[:,np.newaxis,:,:]# n_samples x num_target_hidden_states(1) x 257 x 1408
    assert len(features) == RUN_VOLUMES[run_id-1], 'feat:{} vs volumes:{}'.format(len(features), RUN_VOLUMES[run_id-1])
    assert len(features) == len(masks), 'feat:{} vs mask:{}'.format(len(features), len(masks))

    pooled_features = []
    tracking_exist_flags = []
    for feat, mask in zip(features, masks):
        pooled_feat, flag = masked_avg_pooling(feat, mask)
        pooled_features.append(pooled_feat)
        tracking_exist_flags.append(flag)

    pooled_features = np.concatenate(pooled_features, axis=1) # num_layer x num_samples x embedding_dims
    tracking_exist_flags = np.array(tracking_exist_flags)
    save_dir = os.path.join(SAVE_ROOT, 'features', f'pooled_{modality}-tr{TR}-fo{frame_offset}-mo{mask_offset}', sbj_id)
    os.makedirs(save_dir, exist_ok=True)
    if hidden_state:
        savepath = os.path.join(save_dir, f'fg_b_{run_id}-hs.npy')
    else:
        savepath = os.path.join(save_dir, f'fg_b_{run_id}.npy')
    np.save(savepath, pooled_features)
    print('save as : ', savepath, pooled_features.shape)
    savepath = os.path.join(save_dir, f'fg_b_{run_id}-flag.npy')
    np.save(savepath, tracking_exist_flags)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        required=True,
        type=str,
        help="run image, masked_image, shuffled_masked_image2, saliency_masked_image"
    )
    parser.add_argument(
        "--hidden_state",
        action='store_true',
        help="get hidden state of vision encoder"
    )
    parser.add_argument(
        "--frame_offset",
        type=int,
        default=0,
        help="offset [sec] of frames for VOLUME [0-2]."
    )
    parser.add_argument(
        "--mask_offset",
        type=int,
        default=0,
        help="offset [sec] of masks for VOLUME [0-2]."
    )

    # Set Parameters
    opt = parser.parse_args()
    for sbj_id in SUBJECT_IDS:
        for run_id in tqdm(RUN_IDS):
            run(sbj_id, run_id, opt.hidden_state, opt.modality, TR, opt.frame_offset, opt.mask_offset)
