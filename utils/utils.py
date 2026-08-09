# bascally here we gonna create custom dataset class build on top of Dataset class of torch
#The purpose of this class is to tell PyTorch how to load your images so that they can be used for training a neural network.


from torch.utils.data import Dataset
import os
from PIL import Image  #Python image library it allows us to open images we have seen this in AI-Attandance project


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform = None):   #Root is the folder that contains images
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = list(os.listdir(root)) #root foelder mai jitni bhi files hai read'em and images ko in form of lisst store
        self.files = [p for p in self.files if p.endswith('.jpg', '.png', '.jpeg')]#This is a filter that tells ki img ka allawa koi or file na ho

    def __len__(self):
        return len(self.files)  # This len returns how many samples are inside  
     
    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.files[idx]) #this if for to get the exect location of that image from the root folder like :- my_images/img.png and this img.png is we get from files list that we have created aboove
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image) 

        return image           #This func simply returns images from the folder