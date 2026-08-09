import argparse
import torch
from pathlib import Path
from utils.utils import ImageFolderDataset
from torch.utils.data import DataLoader


def parse_arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('--content_dir',type=str,default=r'C:\Final-Projects\Image-Transform\Content_temp',help='Location of content dataset')
    parser.add_argument('--style_dir',type=str,default=r'C:\Final-Projects\Image-Transform\style_temp',help='Location of style dataset')
    parser.add_argument('--vgg',type=str,default='',help='Location for pre-trained VGG')  #pre-trained CNN encoder we gonna use (VGG)
    parser.add_argument('--experiment',type=str,default='experiment1',help='Name of experiment')  #har exp ke baad yha pr save hoga
    parser.add_argument('--final_size',type,default='',help='')
    return parser.parse_args()



def main():
    args=parse_arguments()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir=Path('experiment')/args.experiment #This is for saving all the expreiments with output of that particular exp ---- to experiments naam ka folder bnaga in which args.expreiments ki values store hoti rhengi
    save_dir.mkdir(exist_ok=True,parents=True) #this is for just creating that directory the experiment dir and saving in that 

    #here we goonna save args for any particular experiment like what args we have used to get this output so we save that in a particular txt file
    with open(save_dir/'args.txt','w') as args_file:  #the save_dir we created above
        for key,val in vars(args).items():             #vars is to convert in dictionary we cannot use dictionary directly  key=name and val=actual value of that 
            args_file.write(f"{key}:{val}\n")

    content_transform=get_transform()
    style_transform=None

    content_dataset=ImageFolderDataset(args.content_dir,content_transform)  #creating datasets of content and style images and like getting data from the folder of content and style 
    style_dataset=ImageFolderDataset(args.style_dir,style_transform)

    content_dataloader = DataLoader(content_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True)
    style_dataloader = DataLoader(style_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True)

            
if __name__=='__main__':
    main()
