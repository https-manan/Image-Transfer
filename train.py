import argparse
import torch
from pathlib import Path
from utils.utils import *
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.models import *


def parse_arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('--content_dir',type=str,default=r'C:\Final-Projects\Image-Transform\Content_temp',help='Location of content dataset')
    parser.add_argument('--style_dir',type=str,default=r'C:\Final-Projects\Image-Transform\style_temp',help='Location of style dataset')
    parser.add_argument('--vgg',type=str,default=r'C:\Final-Projects\Image-Transform\utils\models.py',help='Location for pre-trained VGG')  #pre-trained CNN encoder we gonna use (VGG)
    parser.add_argument('--experiment',type=str,default='experiment1',help='Name of experiment')  #har exp ke baad yha pr save hoga

    #This we gonna take from use the operations we wanna apply on image
    parser.add_argument('--final_size',type=int,default=512,help='size of final image')
    parser.add_argument('--content_size',type=int,default=256,help='size of content image')
    parser.add_argument('--sytle_size',type=int,default=256,help='size of style image')
    parser.add_argument('--crop',type=int,action='store_true',default=True,help='Crop image')#basically jb bhi call --crop to image crop bhi hogi 
    parser.add_argument('--lr',type=float,default=1e-4,help='Learning rate')#1e-4 means that 10 to the power -4
    parser.add_argument('--lr_decay',type=float,default=5e-5,help='Learning rate decay')
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


    content_transform=get_transform(args.content_size,args.crop,args.final_size) #Here we gonna define the resizing and cropping transformation on image  and this we gonna ask from and take from use in parse args
    style_transform=get_transform(args.style_size,args.crop,args.final_size) #This is basically for the style size and upper one is for content size  

                                        #Root,transform in contentDataset
    content_dataset=ImageFolderDataset(args.content_dir,content_transform)  #creating datasets of content and style images and like getting data from the folder of content and style 
    style_dataset=ImageFolderDataset(args.style_dir,style_transform)

    content_dataloader = DataLoader(content_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True)  #shuffel  true means har epoch ke baad dataset gonna shuffel
    style_dataloader = DataLoader(style_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True)  #and this pin_memory is imp for the transformation from CPU to CGP

    encoder=VGGEncoder(args.vgg).to(device)
    decoder=Decoder().to(device)

    optimizer=optim.Adam(decoder.parameters(),lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(    #Scadular is for ki learning rate increase kb krna hai and dec kb krna hai
        optimizer,
        lr_lambda=lambda epoch:1.0/(1.0+args.lr_decay*epoch)
    )


if __name__=='__main__':
    main()
