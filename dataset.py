import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils import data
import random


class my_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.images_list = []
        self.labels_list = []
        self.transform = transform
        count = 0
        for fold in os.listdir(self.root_dir):

            for image in os.listdir(self.root_dir + '/' + fold):
                image_address = self.root_dir + '/' + fold + '/' + image
                self.images_list.append(image_address)
                self.labels_list.append(count)
            count += 1
        self.labels_list = np.array(self.labels_list)
        print(f"num class:{count} num image:{len(self.images_list)}")

    def __getitem__(self, idx):
        image_path = self.images_list[idx]
        label = self.labels_list[idx]
        image = Image.open(image_path).convert('RGB')
        img = np.array(image)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        x0 = 0
        x1 = img.shape[1]
        y0 = 0
        y1 = img.shape[0]
        if img.shape[0] < img.shape[1]:
            img_cmp = np.concatenate(
                [
                    cv2.resize(img[y0:y0 + 1, x0:x1], (x1, max(int(0.5 * (x1 - y1)), 1))),
                    np.float32(img),
                    cv2.resize(img[y1 - 1:y1, x0:x1], (x1, max(int(0.5 * (x1 - y1)), 1)))
                ],
                axis=0)
        elif img.shape[0] > img.shape[1]:
            img_cmp = np.concatenate(
                [
                    cv2.resize(img[y0:y1, x0:x0 + 1], (max(int(0.5 * (y1 - x1)), 1), y1)),
                    np.float32(img),
                    cv2.resize(img[y0:y1, x1 - 1:x1], (max(int(0.5 * (y1 - x1)), 1), y1))
                ],
                axis=1)
        else:
            img_cmp = img
        img_cmp = Image.fromarray(np.uint8(img_cmp))

        if self.transform is not None:
            img_cmp = self.transform(img_cmp)

        return img_cmp, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.images_list)


if __name__ == '__main__':
    root = 'D:/paper_code/dataset/cubbirds'
    dataset = my_dataset('D:/datasets/cubbirds/train')
    train_loader = torch.utils.data.DataLoader(dataset,
                                       batch_size=10,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=0)

    for i, l in train_loader:
        print()



