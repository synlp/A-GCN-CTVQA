from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class M3DVQA(Dataset):
    def __init__(self, csv_path, root, preprocessing):
        self.rows = pd.read_csv(csv_path, keep_default_na=False)
        required = {'Image Path', 'Question', 'Answer', 'Question Type'}
        if not required.issubset(self.rows.columns):
            raise ValueError(f'M3D-VQA CSV requires {sorted(required)}')
        self.root = Path(root)
        self.preprocessing = preprocessing
        for key in ('slice_stride', 'image_size', 'resize_mode', 'mean', 'std'):
            if key not in preprocessing or preprocessing[key] is None:
                raise ValueError(f'Explicit preprocessing setting required: {key}')
        if preprocessing['slice_stride'] < 1 or preprocessing['image_size'] < 1:
            raise ValueError('Slice stride and image size must be positive')
        if len(preprocessing['mean']) != 3 or len(preprocessing['std']) != 3 or min(preprocessing['std']) <= 0:
            raise ValueError('Three channel means and positive standard deviations are required')
        if preprocessing['resize_mode'] not in ('nearest', 'bilinear', 'bicubic'):
            raise ValueError('Unsupported resize_mode')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        volume = np.load(self.root / row['Image Path'], allow_pickle=False)
        if volume.ndim != 4 or volume.shape[0] != 1 or not np.isfinite(volume).all() or volume.min() < 0 or volume.max() > 1:
            raise ValueError('Expected normalized M3D array with shape (1,D,H,W), range [0,1]')
        p = self.preprocessing
        pixels = torch.as_tensor(volume[:, ::p['slice_stride']].copy(), dtype=torch.float32).permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
        kwargs = {} if p['resize_mode'] == 'nearest' else {'align_corners': False}
        pixels = F.interpolate(pixels, size=(p['image_size'], p['image_size']), mode=p['resize_mode'], **kwargs)
        pixels = (pixels - torch.tensor(p['mean'])[None, :, None, None]) / torch.tensor(p['std'])[None, :, None, None]
        return {'pixels': pixels, 'question': str(row['Question']), 'answer': str(row['Answer']), 'category': str(row['Question Type']), 'index': index}
