import argparse, os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from voluntary_fixation.data_processing.blip import get_blip_model
from transformers import InstructBlipProcessor
from voluntary_fixation.envs import SAVE_ROOT, DEVICE, RUN_VOLUMES, SUBJECT_IDS, RUN_IDS, TR
import random
random.seed(1126)

def load_model_from_config():
    class BLIPConfig():
        pretrained_model_name_or_path="Salesforce/instructblip-flan-t5-xl"
        # model parallel training が可能なモデルについては、device_map: auto とすることで自動的にモデルを分割して複数GPUに読み込んでくれる
        device_map='auto'
        # 8bit/4bit training may be incompatible with V100.
        # load_in_8bit: true
        trust_remote_code=True
        # torch_dtype: torch.float16

    blip_config = BLIPConfig()
    model, preprocess = get_blip_model(**blip_config.__dict__)
    # これでvision encoderのhidden state 40層分が取得可能
    model.vision_model.config.output_hidden_states = True
    return model,  preprocess


def get_features(model:torch.nn.Module, preprocess:InstructBlipProcessor, run_id:int, sub_id:str, time_resolution:int, frame_offset:int, mask_offset:int,
                 save_dir:str, modality:str='masked_image', hidden_state=False):
    if hidden_state:
        savepath = os.path.join(save_dir, 'fg_b_{}-hs.npy'.format(run_id))
        print('GETTING HIDDEN STATE + LAST STATE')
    else:
        savepath = os.path.join(save_dir, 'fg_b_{}.npy'.format(run_id))

    # if os.path.exists(savepath):
    #     print('already exists: ', savepath)
    #     return
    assert run_id <= len(RUN_VOLUMES)
    model.eval()

    frames_path = os.path.join(SAVE_ROOT, 'frames', f'resolution{time_resolution}-start{frame_offset}', 'segment{}.npy'.format(run_id))
    masks_path = os.path.join(SAVE_ROOT, 'mask', f'resolution{time_resolution}-start{mask_offset}','mask_sub-{}_run-{}.npy'.format(sub_id, run_id))
    if modality == 'saliency_masked_image':
        masks_path = os.path.join(SAVE_ROOT, 'saliency', 'deepgaze2e_predict', 'mask',
                                  f'resolution{time_resolution}-start{mask_offset}', f'mask_run-{run_id}.npy')
    frames = np.load(frames_path) # n_samples x height x width x 3
    masks = np.load(masks_path) # n_samples x height x width

    masks = masks[:,:,:,np.newaxis]
    assert len(frames) == len(masks), '{} vs {}'.format(len(frames), len(masks))
    assert masks.shape[1:3] == frames.shape[1:3], '{} vs {}'.format(masks.shape[1:3], frames.shape[1:3])
    print('frames.shape: ', frames.shape)
    print('masks.shape: ', masks.shape)


    assert RUN_VOLUMES[run_id-1] == len(frames), '{} vs {}'.format(RUN_VOLUMES[run_id-1], len(frames))

    if modality == 'image':
        # del masks
        pass
    elif modality == 'masked_image' or modality == 'saliency_masked_image':
        masked_images = frames * masks
        assert masked_images.dtype == np.uint8
        del frames
    elif modality == 'reverse_masked_image':
        masked_images = frames * ~masks
        assert masked_images.dtype == np.uint8
        del frames
    elif modality == 'sampled_reverse_masked_image':

        new_masks = np.zeros_like(masks).astype(np.uint8)
        for m, mask in enumerate(masks):
            num_non_mask = np.sum(mask == 1) # eye fixation
            num_mask = np.sum(mask == 0)

            if num_mask > num_non_mask: # 視線領域より、非視線領域の方が多い場合：通常こちら
                sampled_mask_indices = random.sample(range(num_mask), num_non_mask)
            else:
                sampled_mask_indices = list(range(num_mask))
            masked_indices = np.where(mask == 0) # other than eye fixation
            masked_indices = (masked_indices[0][sampled_mask_indices], masked_indices[1][sampled_mask_indices])
            new_masks[m][masked_indices] = 1
        new_masks_path = os.path.join(SAVE_ROOT, 'sampled_reverse_mask', f'resolution{time_resolution}-start{mask_offset}','mask_sub-{}_run-{}.npy'.format(sub_id, run_id))
        os.makedirs(os.path.dirname(new_masks_path), exist_ok=True)
        np.save(new_masks_path, new_masks)
        masked_images = frames * new_masks
        assert masked_images.dtype == np.uint8
        del frames
    elif modality == 'shift_reverse_masked_image':
        N, H, W, C = masks.shape
        new_masks = np.zeros_like(masks).astype(np.uint8)
        for m, mask in enumerate(masks):
            new_mask = np.roll(mask, (H//2, W//2), axis=(0,1))
            new_masks[m]=new_mask
        new_masks_path = os.path.join(SAVE_ROOT, 'shift_reverse_mask', f'resolution{time_resolution}-start{mask_offset}','mask_sub-{}_run-{}.npy'.format(sub_id, run_id))
        os.makedirs(os.path.dirname(new_masks_path), exist_ok=True)
        np.save(new_masks_path, new_masks)
        masked_images = frames * new_masks
        assert masked_images.dtype == np.uint8
        del frames
    elif modality == 'shift_reverse_masked_image025':
        N, H, W, C = masks.shape
        new_masks = np.zeros_like(masks).astype(np.uint8)
        for m, mask in enumerate(masks):
            new_mask = np.roll(mask, (H//4, W//4), axis=(0,1))
            new_masks[m]=new_mask
        new_masks_path = os.path.join(SAVE_ROOT, 'shift_reverse_mask025', f'resolution{time_resolution}-start{mask_offset}','mask_sub-{}_run-{}.npy'.format(sub_id, run_id))
        os.makedirs(os.path.dirname(new_masks_path), exist_ok=True)
        np.save(new_masks_path, new_masks)
        masked_images = frames * new_masks
        assert masked_images.dtype == np.uint8
        del frames
    elif modality == 'shift_reverse_masked_image075':
        N, H, W, C = masks.shape
        new_masks = np.zeros_like(masks).astype(np.uint8)
        for m, mask in enumerate(masks):
            new_mask = np.roll(mask, ((H//4)*3, (W//4)*3), axis=(0,1))
            new_masks[m]=new_mask
        new_masks_path = os.path.join(SAVE_ROOT, 'shift_reverse_mask075', f'resolution{time_resolution}-start{mask_offset}','mask_sub-{}_run-{}.npy'.format(sub_id, run_id))
        os.makedirs(os.path.dirname(new_masks_path), exist_ok=True)
        np.save(new_masks_path, new_masks)
        masked_images = frames * new_masks
        assert masked_images.dtype == np.uint8
        del frames

    elif modality == 'shift_reverse_masked_image_shuffle':
        N, H, W, C = masks.shape
        new_masks = np.zeros_like(masks).astype(np.uint8)
        for m, mask in enumerate(masks):
            hh = random.choice([-3,-2,-1,1,2,3])
            ww = random.choice([-3,-2,-1,1,2,3])
            new_mask = np.roll(mask, ((H//4)*hh, (W//4)*ww), axis=(0,1))
            new_masks[m]=new_mask
        new_masks_path = os.path.join(SAVE_ROOT, 'shift_reverse_mask_shuffle', f'resolution{time_resolution}-start{mask_offset}','mask_sub-{}_run-{}.npy'.format(sub_id, run_id))
        os.makedirs(os.path.dirname(new_masks_path), exist_ok=True)
        np.save(new_masks_path, new_masks)
        masked_images = frames * new_masks
        assert masked_images.dtype == np.uint8
        del frames

    elif modality == 'shuffled_masked_image':
        shuffle_indices = np.random.permutation(np.arange(0, masks.shape[0]))
        sample_masks = masks[shuffle_indices]
        new_masks_path = os.path.join(SAVE_ROOT, 'shuffled_mask', f'resolution{time_resolution}-start{mask_offset}','mask_sub-{}_run-{}.npy'.format(sub_id, run_id))
        os.makedirs(os.path.dirname(new_masks_path), exist_ok=True)
        np.save(new_masks_path, sample_masks)
        masked_images = frames * sample_masks
        assert masked_images.dtype == np.uint8
        del frames
    elif modality == 'shuffled_masked_image2':
        shuffle_indices = np.random.permutation(np.arange(0, masks.shape[0]))
        sample_masks = masks[shuffle_indices]
        # empty eyetrackの処理
        empty_indices = np.where(np.sum(sample_masks, axis=(1,2)) == 0)[0]
        kk = 0
        while len(empty_indices) > 0:
            sample_masks[empty_indices] = sample_masks[empty_indices-1]
            empty_indices = np.where(np.sum(sample_masks, axis=(1,2)) == 0)[0]
            print(f'find {len(empty_indices)} empty indices: ', kk)
            kk+=1
        new_masks_path = os.path.join(SAVE_ROOT, 'shuffled_mask2', f'resolution{time_resolution}-start{mask_offset}','mask_sub-{}_run-{}.npy'.format(sub_id, run_id))
        os.makedirs(os.path.dirname(new_masks_path), exist_ok=True)
        np.save(new_masks_path, sample_masks)
        masked_images = frames * sample_masks
        assert masked_images.dtype == np.uint8
        del frames

    elif modality == 'previous_masked_image':
        rolled_indices = np.roll(np.arange(0, masks.shape[0]), 1)
        sample_masks = masks[rolled_indices]
        # empty eyetrackの処理
        empty_indices = np.where(np.sum(sample_masks, axis=(1,2)) == 0)[0]
        kk = 0
        while len(empty_indices) > 0:
            sample_masks[empty_indices] = sample_masks[empty_indices-1]
            empty_indices = np.where(np.sum(sample_masks, axis=(1,2)) == 0)[0]
            print(f'find {len(empty_indices)} empty indices: ', kk)
            kk+=1
        new_masks_path = os.path.join(SAVE_ROOT, 'previous_mask', f'resolution{time_resolution}-start{mask_offset}','mask_sub-{}_run-{}.npy'.format(sub_id, run_id))
        os.makedirs(os.path.dirname(new_masks_path), exist_ok=True)
        np.save(new_masks_path, sample_masks)
        masked_images = frames * sample_masks
        assert masked_images.dtype == np.uint8
        del frames

    elif modality == 'both':
        masked_images = frames * masks
        assert masked_images.dtype == np.uint8
        # del masks
        pass
    else:
        raise ValueError('modality must be image, masked_image, or both')

    features = []
    for i in tqdm(np.arange(0, RUN_VOLUMES[run_id-1])):
        # image = target_imgs[i]
        # mask = run_focus_mask[i]
        # masked_target = mask * image

        if modality == 'image':
            images = Image.fromarray(frames[i])
            prompt = "Can yon tell me about this image in detail?"
        elif  modality in ['masked_image', 'reverse_masked_image',
                          'saliency_masked_image', 'sampled_reverse_masked_image',
                          'shuffled_masked_image', 'shuffled_masked_image2', 'previous_masked_image',
                          'shift_reverse_masked_image','shift_reverse_masked_image025','shift_reverse_masked_image075',
                          'shift_reverse_masked_image_shuffle']:
            images = Image.fromarray(masked_images[i])
            prompt = "Can yon tell me about this image in detail?"
        elif modality == 'both':
            images = Image.fromarray(np.concatenate([frames[i], masked_images[i]], axis=1))
            prompt = 'The photo on the right is a cutout of the area of attention from the photo on the left. Can you describe the right photo in detail?'
        else:
            raise ValueError('modality must be image, masked_image, reverse_masked_image, or both')

        inputs = preprocess(images=images, text=prompt, return_tensors="pt")

        text = prompt
        answer='dummy'
        # LAVISに沿って上書き
        preprocess.tokenizer.truncation_side = 'left'
        input_dict = preprocess.tokenizer(
                text,
                padding="longest",
                truncation=True,
                max_length=40,
                return_tensors="pt",
            )
        inputs['input_ids'] = input_dict.input_ids# .squeeze()
        inputs['attention_mask'] = input_dict.attention_mask# .squeeze()

        preprocess.tokenizer.truncation_side = 'right'
        labels_dict = preprocess.tokenizer(
                answer,
                padding="longest",
                truncation=True,
                max_length=40,
                return_tensors="pt",
            )
        targets = labels_dict.input_ids.masked_fill(
                labels_dict.input_ids == preprocess.tokenizer.pad_token_id, -100
            )
        inputs["labels"] = targets# .squeeze()
        inputs = inputs.to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        if hidden_state:
            vision_features = [outputs.vision_outputs.hidden_states[hi] for hi in [0, 9, 19, 29, 39]]
            vision_features = torch.cat(vision_features, axis=0).unsqueeze(0) # batch_size(1) x num_target_hidden_states x 16 x 16
        else:
            vision_features = outputs.vision_outputs.last_hidden_state
        qformer_features = outputs.qformer_outputs.last_hidden_state

        if modality == 'image':
            features.append(vision_features)
        elif modality in ['masked_image', 'reverse_masked_image',
                          'saliency_masked_image', 'sampled_reverse_masked_image',
                          'shuffled_masked_image', 'shuffled_masked_image2', 'previous_masked_image',
                          'shift_reverse_masked_image','shift_reverse_masked_image025','shift_reverse_masked_image075',
                          'shift_reverse_masked_image_shuffle']:
            features.append(vision_features)
        elif modality == 'both':
            features.append(qformer_features)
        else:
            raise ValueError('modality must be image, masked_image, reverse_masked_image, or both')
    features = torch.cat(features, dim=0)
    print('features.shape: ', features.shape)
    features = features.cpu().numpy()
    np.save(savepath, features)
    print('save to ', savepath)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        required=True,
        type=str,
        help="run image, masked_image, both, reverse_masked_image, saliency_masked_image, sampled_reverse_masked_image, shuffled_masked_image, shuffled_masked_image2, previous_masked_image"
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

    model, preprocess = load_model_from_config() # model: BLIP, preprocess: InstructBlipProcessor (resize to 224, patch_size=14 (num_patch=16x16), ...)
    if opt.modality == 'image':
        sub_id = SUBJECT_IDS[0]
        save_dir = os.path.join(SAVE_ROOT, 'features', opt.modality + f'-tr{TR}-fo{opt.frame_offset}-mo{opt.mask_offset}')
        os.makedirs(save_dir, exist_ok=True)
        for run_id in RUN_IDS:
            print('No subject specific -- Run: {run_id}'.format(sub_id=sub_id, run_id=run_id))
            get_features(model, preprocess, run_id, sub_id, TR, opt.frame_offset, opt.mask_offset,
                        save_dir, modality=opt.modality, hidden_state=opt.hidden_state)
    elif opt.modality in ['masked_image', 'both', 'reverse_masked_image', 'saliency_masked_image',
                          'sampled_reverse_masked_image', 'shuffled_masked_image', 'shuffled_masked_image2', 'previous_masked_image',
                          'shift_reverse_masked_image','shift_reverse_masked_image025','shift_reverse_masked_image075',
                          'shift_reverse_masked_image_shuffle']:
        for sub_id in SUBJECT_IDS:
            save_dir = os.path.join(SAVE_ROOT, 'features', opt.modality + f'-tr{TR}-fo{opt.frame_offset}-mo{opt.mask_offset}', sub_id)
            os.makedirs(save_dir, exist_ok=True)
            for run_id in RUN_IDS:
                print('Sub: {sub_id} -- Run: {run_id}'.format(sub_id=sub_id, run_id=run_id))
                get_features(model, preprocess, run_id, sub_id, TR, opt.frame_offset, opt.mask_offset,
                            save_dir, modality=opt.modality, hidden_state=opt.hidden_state)
    else:
        raise ValueError('modality must be image, masked_image, saliency_masked_image, reverse_masked_image or both')



if __name__ == "__main__":
    DEBUG=False
    # python voluntary_fixation/data_processing/create_masked_visual_features.py  --modality image --hidden_state
    # python voluntary_fixation/data_processing/create_masked_visual_features.py  --modality shift_reverse_masked_image --hidden_state
    # python voluntary_fixation/data_processing/create_masked_visual_features.py  --modality shift_reverse_masked_image025 --hidden_state
    # python voluntary_fixation/data_processing/create_masked_visual_features.py  --modality shift_reverse_masked_image075 --hidden_state
    # python voluntary_fixation/data_processing/create_masked_visual_features.py  --modality shift_reverse_masked_image_shuffle --hidden_state


    main()