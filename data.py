import os
import cv2
from paddle.io import Dataset

class ImageNetDataset(Dataset):
    def __init__(self, data_dir, info_txt, mode='train', transforms=None):
        self.data_dir = data_dir
        self.image_paths, self.labels = self.get_info(info_txt)
        self.mode = mode
        self.transforms = transforms

    def get_info(self, file_path):
        paths = []
        labels = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                image_name, label = line.strip().split(' ')
                #print ("name:",image_name)
                paths.append(os.path.join(self.data_dir, image_name))
                labels.append(int(label))
        return paths, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        if image is None:
            print(f"image_path={image_path} is None")
        if self.transforms:
            image = self.transforms(image)
        if self.mode == 'train':
            return image, label
        else:
            return image
