# bascally here we gonna create custom dataset class build on top of Dataset class of torch
#The purpose of this class is to tell PyTorch how to load your images so that they can be used for training a neural network.


from torch.utils.data import Dataset
import os
from PIL import Image  #Python image library it allows us to open images we have seen this in AI-Attandance project
from torchvision import transforms



class ImageFolderDataset(Dataset):
    def __init__(self, root, transform = None):   #Root is the folder that contains images
        super(ImageFolderDataset, self).__init__() #calling constructor of parent class 
        self.root = root
        self.transform = transform
        self.files = list(os.listdir(root)) #root folder mai jitni bhi files hai read'em and images ko in form of lisst store
        self.files = [p for p in self.files if p.lower().endswith(('.jpg', '.png', '.jpeg'))]#This is a filter that tells ki img ka allawa koi or file na ho

    def __len__(self):
        return len(self.files)  # This len returns how many samples are inside
     
    def __getitem__(self, idx): #Get item index value leta hai and we can get the item on that index 
        image_path = os.path.join(self.root, self.files[idx]) #this if for to get the exect location of that image from the root folder like :- my_images/img.png and this img.png is we get from files list that we have created above IMPPPPPPPPPPPPPPPPPPP  so yha root mai uss file ke naam se la rhe hai by passing index
        image = Image.open(image_path).convert('RGB')  #reading the image
        
        if self.transform:
            image = self.transform(image)      #just applying transforms on image before returning it   

        return image           #This func simply returns images from the folder



def get_transform(size, crop, final_size):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))  #jo given size by user hai usme transform the image 1st 
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))  #RandCrop is predefined to crop the image acc to final _size 
    else:
        transform_list.append(transforms.Resize(final_size))  #and then final size ke brabar resize the image and return 

    transform_list.append(transforms.ToTensor())  #image ko transform mai convert 
    return transforms.Compose(transform_list)    #And finally saari images pr transform apply krke return 



def adaptive_instance_normalization(content_feat, style_feat):
    # [batch_size, channels, h, w]  this is the general input to this AdaIN layer
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)



def calc_mean_std(feat, eps=1e-5):
    # [ batch size, channels, h, w]
    size = feat.size()
    assert (len(size) == 4)
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std