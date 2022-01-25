import numpy as np
import os, imageio
import json
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class BlenderDataset(Dataset):
    def __init__(self, data_path, split):
        super(BlenderDataset, self).__init__()
        self.img_paths = []
        self.poses = [] # extrinsic
        with open(os.path.join(data_path, 'transforms_{}.json'.format(split)), 'r') as fp:
            meta = json.load(fp)
        frames = meta['frames']
        for frame in frames:
            self.img_paths.append(os.path.join(data_path, frame['file_path'] + '.png'))
            self.poses.append(np.array(frame['transform_matrix']))

        self.poses = np.array(self.poses).astype(np.float32)
        # read the shape from the first image
        self.h, self.w = imageio.imread(self.img_paths[0]).shape[:2]
        self.camera_angle_x = float(meta['camera_angle_x'])
        self.focal = .5 * self.w/ np.tan(.5 * self.camera_angle_x)


        if split == 'train':
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        else: # split == test
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        self.dataset_flags = {
            'white_bkgd': True,
            'use_viewdirs': True, # use full 5D input instead of 3D
            'no_ndc': True,
        }

        self.bds = {'near': 2., 'far':6.}
        self.intrinsic =   np.array([
            [self.focal, 0, 0.5*self.w],
            [0, self.focal, 0.5*self.h],
            [0, 0, 1]
        ])
        self.hwf = [self.h, self.w, self.focal]


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        img = img.convert('RGB')
        pose = self.poses[idx]  # extrinsic and intrinsic
        # TODO Need to account for resizing if doing
        if self.transform:
            torch_img = self.transform(img)
        sample = {'torch_img':torch_img, 'pose': pose, 'hwf':self.hwf, "K":self.intrinsic}
        return sample