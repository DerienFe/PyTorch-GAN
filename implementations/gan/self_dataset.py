# class of cat dataset

import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import cv2




class Self_dataset(Dataset):
    """
    Cat dataset
    """
    
    def __init__(self, PATH, transform = None):
        """
        args:
            PATH: directory where all images store
            transform: callable, optional, optional transform to be applied on sample.
        """
        self.PATH = PATH
        self.transform = transform
    
    def __len__(self):
        return len(os.listdir(self.PATH))
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        file_names = os.listdir(self.PATH)
            
        img_name = os.path.join(self.PATH, file_names[idx])
        
        image = cv2.imread(img_name, 0) #use this for grayscale
        #image = cv2.imread(img_name)
        
        image = transforms.functional.to_pil_image(image)
        
        if self.transform:
            image = self.transform(image)
        
        #print("dataset class type :")
        #print(type(image))
        
        return image

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image= sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (self.output_size, self.output_size))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return img
