# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
# import dnnlib
import warnings

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
        envmap_dir  = None,     # directory to load the DECA envmap sh.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        # self._envmap_pth = os.path.join(envmap_dir, "deca_preds.npy")
        self._envmap_pth = None
        self._light_shs = None

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

    def _get_raw_lights(self):
        if self._light_shs is None:
            self._light_shs = self._load_raw_lights()
        return self._light_shs

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_lights(self): # to be overridden by subclass
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
        image, mask = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        # return image.copy(), mask.copy(), self.get_label(idx), self.get_light(idx)
        return image.copy(), mask.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_light(self, idx):
        light = self._get_raw_lights()[self._raw_idx[idx]]
        return light.copy()

    # def get_details(self, idx):
    #     d = dnnlib.EasyDict()
    #     d.raw_idx = int(self._raw_idx[idx])
    #     d.xflip = (int(self._xflip[idx]) != 0)
    #     d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
    #     return d

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
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

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
        resolution      = None, # Ensure specific resolution, None = highest available.
        img_subpath     = None, # subpath under path to look for image pngs
        mask_subpath    = None, # subpath under path to look for masks pngs
        rnd_bg          = "none", # set background to random color
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self._has_mask = mask_subpath is not None
        self._mask_subpath = mask_subpath
        self._img_subpath = img_subpath
        self._rnd_bg = rnd_bg

        if self._rnd_bg != "none" and not self._has_mask:
            raise ValueError("`mask_subpath` must be provided if `rnd_bg != none`")

        if os.path.isdir(self._path):
            self._type = 'dir'
            img_path = self._path
            if img_subpath is not None:
                img_path = os.path.join(self._path, img_subpath)
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(img_path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            if self._has_mask:
                raise NotImplementedError("Zipped dataset with masks is not supported yet.")
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
            if img_subpath is not None:
                self._all_fnames = {x for x in self._all_fnames if x[:len(img_subpath)] == img_subpath}
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0)[0].shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        self._label_fname = "dataset.json"
        if img_subpath is not None:
            self._label_fname = os.path.join(img_subpath, "dataset.json")
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

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
        if self._has_mask:
            mask_fname = os.path.join(self._mask_subpath, os.path.relpath(fname, start=self._img_subpath))
            with self._open_file(mask_fname) as f:
                if pyspng is not None and self._file_ext(fname) == '.png':
                    mask_image = pyspng.load(f.read())
                else:
                    mask_image = np.array(PIL.Image.open(f))
                if mask_image.ndim == 2:
                    mask_image = mask_image[:, :, np.newaxis] # HW => HWC
                mask_image = mask_image.astype('float32') / 255.0
            if self._rnd_bg == "per-image":
                image = image * mask_image + (np.random.randint(256, size=[1, 1, 3])) * (1-mask_image)
                image = image.astype('uint8')
            elif self._rnd_bg == "per-pixel":
                image = image * mask_image + (np.random.randint(256, size=image.shape)) * (1-mask_image)
                image = image.astype('uint8')
            elif self._rnd_bg == "static":
                image = image * mask_image + 255 * (1 - mask_image)
                image = image.astype('uint8')
        else:
            mask_image = np.ones_like(image[..., :1], dtype=np.float32)

        image = image.transpose(2, 0, 1) # HWC => CHW
        mask_image = mask_image.transpose(2, 0, 1)
        return image, mask_image

    def _load_raw_labels(self):
        if self._label_fname not in self._all_fnames:
            return None
        with self._open_file(self._label_fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        idx_no_labels = [
                (idx, fname) for idx, fname in enumerate(self._image_fnames)
                if os.path.join(os.path.basename(os.path.dirname(fname)), os.path.basename(fname)) not in labels]
        if idx_no_labels:
            warnings.warn("Couldn't find label for {}".format(", ".join(fname for _, fname in idx_no_labels)))
            # find the first index that's not in idx_no_labels
            for idx_exist in self._raw_idx:
                if idx_exist not in idx_no_labels:
                    break
            for idx, _ in idx_no_labels:
                self._image_fnames[idx] = self._image_fnames[idx_exist]
        labels = [labels[os.path.join(os.path.basename(os.path.dirname(fname)), os.path.basename(fname))]
                for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def _load_raw_lights(self):
        light_shs_dict = np.load(self._envmap_pth, allow_pickle=True).item()
        light_shs = [light_shs_dict[os.path.join(os.path.basename(os.path.dirname(fname)), os.path.basename(fname))]
                for fname in self._image_fnames]
        light_shs = np.array(light_shs).astype(np.float32)
        return light_shs


#----------------------------------------------------------------------------

class EnvmapSHDataset(torch.utils.data.Dataset):
    def __init__(self,
        root_dir,   # Root dir of the light dataset.
        split,      # The split of the dataset ['train', 'test'].
        deg,        # The highest degree of the SH.
    ):
        if split not in ['train', 'test']:
            raise ValueError(f'split has to be either \'train\' or \'test\', {split} given.')
        self.split = split
        self.deg = deg
        load_pth = os.path.join(root_dir, f"deg_{deg}", f"{split}_sh.npy")
        raw_sh = np.load(load_pth, allow_pickle=True).item()
        self.sh_keys = sorted(list(raw_sh.keys()))
        self.sh_params = np.array([raw_sh[k] for k in self.sh_keys])
        self.n_envmaps = self.sh_params.shape[0]
        if self.split == 'train':
            deca_load_pth = os.path.join(root_dir, f"deg_{deg}", "deca_preds.npy")
            deca_sh = np.load(deca_load_pth, allow_pickle=True).item()
            self.deca_keys = sorted(list(deca_sh.keys()))
            self.deca_sh_params = np.array([deca_sh[k] for k in self.deca_keys])
            self.deca_rate = 0  # 1.  # 0.9
        else:
            self.deca_rate = 0
        # Sample only the DECA envmaps
        # deca_load_pth = os.path.join(root_dir, f"deg_{deg}", "deca_preds.npy")
        # deca_sh = np.load(deca_load_pth, allow_pickle=True).item()
        # self.deca_keys = sorted(list(deca_sh.keys()))
        # self.deca_sh_params = np.array([deca_sh[k] for k in self.deca_keys])
        # self.deca_rate = 1.  # 0.9
        #

    def __getitem__(self, idx):
        if np.random.uniform(0., 1.) < self.deca_rate:
            if idx >= self.deca_sh_params.shape[0]:
                idx = idx % self.deca_sh_params.shape[0]
            # rad_round_z = np.random.normal(0., np.pi * 0.1)
            # return self.rot_sh_around_z(self.deca_sh_params[idx], rad_round_z).astype(np.float32)
            # No random envmap rotation in training
            return self.deca_sh_params[idx].astype(np.float32)
            #
        else:
            if idx >= self.n_envmaps:
                idx = idx % self.n_envmaps
            # rad_round_z = np.random.uniform(0., np.pi * 2)
            # rad_round_z = np.random.normal(0., np.pi * 0.25)
            # rad_round_z = np.clip(np.random.normal(0., np.pi * 0.2), -np.pi * 0.3, np.pi * 0.3)
            # return self.rot_sh_around_z(self.sh_params[idx], rad_round_z).astype(np.float32)
            return self.sh_params[idx].astype(np.float32)

    def __len__(self):
        return self.n_envmaps

    def _random_rot_sh(self):
        pass

    def _random_rot_sh_around_z(self, coeffs):
        rad_round_z = np.random.uniform(0., np.pi * 2)
        return self.rot_sh_around_z(coeffs, rad_round_z)

    def rot_sh_around_z(self, coeffs, rad_round_z):
        rot_mat = np.eye(1, dtype=np.float32)
        rotated_sh = [coeffs[:1]]
        for i in range(1, self.deg + 1):
            coeff_seg = coeffs[i**2:(i + 1)**2]
            rot_mat = np.pad(rot_mat, (1, 1), mode='constant', constant_values=0.)
            rot_mat[0, 0] = np.cos(rad_round_z * i)
            rot_mat[0, -1] = np.sin(rad_round_z * i)
            rot_mat[-1, 0] = -np.sin(rad_round_z * i)
            rot_mat[-1, -1] = np.cos(rad_round_z * i)
            coeff_seg = rot_mat @ coeff_seg
            rotated_sh.append(coeff_seg)
        rotated_sh = np.concatenate(rotated_sh, axis=0)
        return rotated_sh

    def get_no_rot(self, idx):
        if np.random.uniform(0., 1.) < self.deca_rate:
            if idx >= self.deca_sh_params.shape[0]:
                idx = idx % self.deca_sh_params.shape[0]
            return self.deca_sh_params[idx].astype(np.float32)
        else:
            if idx >= self.n_envmaps:
                idx = idx % self.n_envmaps
            return self.sh_params[idx].astype(np.float32)

#----------------------------------------------------------------------------
