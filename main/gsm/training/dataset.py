# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
from pathlib import Path
import cv2
from utils.image_utils import get_bg_color
import scipy.io as sio
try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        # assert self.image_shape[1] == self.image_shape[2]
        return self._resolution #self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        label_file      = "dataset.mat", # Relative path of the label file
        resolution      = None, # Ensure specific resolution, None = highest available.
        gaussian_weighted_sampler = None,
        sample_std      = 15,
        bg_color        = "white",
        old_code        = False,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._old_code = old_code
        self._zipfile = None
        self._resolution = resolution
        self._label_file = label_file
        self.gaussian_weighted_sampler = gaussian_weighted_sampler
        self._bg_color = np.uint8(get_bg_color(bg_color).numpy() * 255.0)
        PIL.Image.init()
        if os.path.isdir(self._path):
            self._type = 'dir'
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
        else:
            raise IOError('Path must point to a directory or zip')

        if label_file.endswith('.mat'):
            labels = sio.loadmat(self._open_file('dataset.mat'))['labels']
            self._all_fnames = [str(x[0][0]) for x in labels]
        elif label_file.endswith('.json'):
            with open(os.path.join(self._path, label_file)) as f:
                labels = json.load(f)['labels']
            self._all_fnames = list(labels.keys())

        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

        self.weights = None
        self.gaussian_weighted_sampler = gaussian_weighted_sampler
        if self.gaussian_weighted_sampler:
            self.weights = self.pdf_weights(sample_std)

    def get_label(self, idx):
        label = super().get_label(idx)
        if label.shape[0] == 0:
            return label
        # TODO: Change intrinsics
        if not self._old_code:
            aspect_ratio = self._raw_shape[-2] / self._raw_shape[-1]
            label[..., 16] /= aspect_ratio
        return label

    def pdf_weights(self, std):
        labels = None
        if 'dataset.mat' in self._all_fnames or True:
            # This is faster and hence preferred.
            import scipy.io as sio
            labels = sio.loadmat(self._open_file('dataset.mat'))['labels']
            labels = [(str(x[0][0]), x[1][0]) for x in labels]
        elif 'dataset.json' in self._all_fnames:
            with self._open_file('dataset.json') as f:
                labels = json.load(f)['labels']
        else:
            return None

        gol = []
        for label in labels:
            go = label[1][25:][:3]
            gol.append(go)
        gol = np.array(gol)

        angles = np.clip(gol[:,1], -1, 1) %(2 * np.pi)
        frac_num = 360
        frac = 2 * np.pi / frac_num
        num_list = []
        index_list = []
        for i in range(frac_num):
            index = np.logical_and(angles >= i * frac, angles < (i + 1) * frac)
            num_list.append(index.sum())
            index_list.append(index)
        weights = np.zeros_like(angles).reshape(-1)
        all_samples = sum(num_list)

        pdf_list = []
        for i, index, num in zip(range(len(index_list)), index_list, num_list):
            if i < 180:
                shift_i = 360 + i - 180
            else:
                shift_i = i - 180
            # print(i, shift_i, num)
            pdf = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((shift_i-180)/std)**2 / 2)
            if shift_i < 180-20 or shift_i > 180+20:
                pdf = max(pdf, num / all_samples)
            pdf_list.append(pdf)
            if num > 0:
                weights[index] = pdf * 1000 / num
        return weights


    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        mask = self._load_raw_mask(self._raw_idx[idx])
        label = self.get_label(idx)
        # not square
        if image.shape[-1] != image.shape[-2]:
            # SHHQ is tall image
            assert image.shape[-2] > image.shape[-1]
            pad = (image.shape[-2] - image.shape[-1])//2
            p1d = (pad, pad)
            image = np.pad(image, ((0,0),(0,0),p1d), "constant", constant_values=((0,0),(0,0),(0,0)))
        if mask.shape[-1] != mask.shape[-2]:
            assert mask.shape[-2] > mask.shape[-1]
            pad = (mask.shape[-2] - mask.shape[-1])//2
            p1d = (pad, pad)
            mask = np.pad(mask, ((0,0),(0,0), p1d), "constant", constant_values=((0,0),(0,0),(0,0)))

        if self._resolution is not None and self._resolution != image.shape[0]:
            image = cv2.resize(image.transpose(1, 2, 0), (self._resolution, self._resolution)).transpose(2, 0, 1)
            mask = cv2.resize(mask.transpose(1, 2, 0), (self._resolution, self._resolution))
            mask = mask[..., None].transpose(2, 0, 1)

        image = image * mask + self._bg_color.reshape(3, 1, 1) * (1 - mask)
        image = image.astype(np.uint8)
        assert isinstance(image, np.ndarray)
        # assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8

        return image.copy(), mask.copy(), label

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC

        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_mask(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        fname_mask = fname.replace('images', 'masks')
        with self._open_file(fname_mask) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image_mask = pyspng.load(f.read())
            else:
                image_mask = np.array(PIL.Image.open(f))
            if len(image_mask.shape) == 3:
                image_mask = image_mask.mean(axis=-1)
            if len(image_mask.shape) == 2:
                image_mask = image_mask[...,None]
            if 'deepfashion' in self._path.lower():
                image_mask[image_mask > 0] = 255.0
            if (np.unique(image_mask) > 1).sum() >= 1:
                image_mask = image_mask.astype("float32") / 255.0
            if  image_mask.max() == 255:
                image_mask = image_mask.astype("float32") / 255.0
            if image_mask.shape[-1] > 1:
                image_mask = image_mask[..., 0:1]
        # if image_mask.dtype !=np.float32:
        #     print(image_mask.dtype)
        image_mask = image_mask.astype('float32')
        image_mask = image_mask.transpose(2, 0, 1) # HWC => CHW
        assert image_mask.max() <= 1.0 
        assert  image_mask.min() >= 0.0 
        assert image_mask.dtype == np.float32
        return image_mask

    def _load_raw_labels(self):
        labels = None
        if self._label_file.endswith('.mat'):
            # This is faster and hence preferred.
            import scipy.io as sio
            labels = sio.loadmat(self._open_file('dataset.mat'))['labels']
            labels = [(str(x[0][0]), x[1][0]) for x in labels]
            # print(labels)
        elif self._label_file.endswith('.json'):
            with self._open_file('dataset.json') as f:
                labels = json.load(f)['labels']
        else:
            return None
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

        return labels

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
