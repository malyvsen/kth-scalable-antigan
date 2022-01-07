import torch.utils.data as data
import numpy as np

class AntiganDataset(data.Dataset):
    def __init__(self, train_images, train_target, transform):
        self.image_set = train_images 
        self.target_set = train_target  

        self.transform = transform
 

    def __getitem__(self, index):
        target = self.target_set[index]
        img = self.image_set[index]

        img = self.transform(img)
        
        target = np.array(target)

        #target = self.transform(target)

        sample = {'img': img, 'annot': target}
        return sample

    def __len__(self):
        return len(self.image_set)

    def load_annotations(self, index):
        return np.array(self.target_set[index])
