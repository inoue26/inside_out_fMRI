from nvae_dir.datasets import FGReconstSubjectRawDict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from torchmetrics.classification import BinaryJaccardIndex
from nvae_dir.utils import unmap_pixels
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryJaccardIndex
import random
import numpy as np
from scipy.misc import face
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
from functools import partial
import cv2
import deepgaze_pytorch
from voluntary_fixation.envs import MOVIE_FPS, MOVIE_WIDTH, MOVIE_HEIGHT



def pooling(x, pooling_size=8):
    x = torch.from_numpy(x).unsqueeze(0)
    x = F.avg_pool2d(x, pooling_size)
    # x = x.squeeze(0).cpu().numpy()
    return x

class ImageDataset(torch.utils.data.Dataset):
    pass

class MovieDataset(torch.utils.data.Dataset):
    def __init__(self, movie_path:str, target_width:int, target_height:int):
        print('movie_path: ', movie_path)
        self.cap = cv2.VideoCapture(movie_path)

        self.original_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.original_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        assert self.original_height == MOVIE_HEIGHT, f'{self.original_height} != {MOVIE_HEIGHT}'
        assert self.original_width == MOVIE_WIDTH, f'{self.original_width} != {MOVIE_WIDTH}'
        assert self.fps == MOVIE_FPS, f'{self.fps} != {MOVIE_FPS}'

        print('video info: \n resolution: {} x {} \n fps: {} \n frame_num: {}'.format(self.original_height,
                                                                                    self.original_width,
                                                                                    self.fps,
                                                                                    self.num_frames))
        self.target_width = target_width if target_width is not None else self.original_width
        self.target_height = target_height if target_height is not None else self.original_height
        if target_width is None:
            self.resize = None
        else:
            self.resize = partial(cv2.resize, dsize=(self.target_width, self.target_height), interpolation=cv2.INTER_CUBIC)
        self.max_seconds = np.ceil(self.num_frames / self.fps)

    def __len__(self):
        return int(self.num_frames)

    def __getitem__(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError('frame not found')

        if self.resize is not None:
            frame = self.resize(src=frame)
        return frame