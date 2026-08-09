import argparse
import torch
from pathlib import Path
from utils.utils import *
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.models import *
from tqdm import tqdm


def parse_arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('--content_dir',type=str,default=r'C:\Final-Projects\Image-Transform\Content_temp',help='Location of content dataset')
    parser.add_argument('--style_dir',type=str,default=r'C:\Final-Projects\Image-Transform\style_temp',help='Location of style dataset')
    parser.add_argument('--vgg',type=str,default=r'C:\Final-Projects\Image-Transform\utils\models.py',help='Location for pre-trained VGG')  #pre-trained CNN encoder we gonna use (VGG)
    parser.add_argument('--experiment',type=str,default='experiment1',help='Name of experiment')  #har exp ke baad yha pr save hoga

    #This we gonna take from use the operations we wanna apply on image
    parser.add_argument('--final_size',type=int,default=512,help='size of final image')
    parser.add_argument('--content_size',type=int,default=256,help='size of content image')
    parser.add_argument('--style_size',type=int,default=256,help='size of style image')
    parser.add_argument('--crop', action='store_true', default=True, help='Crop image')#basically jb bhi call --crop to image crop bhi hogi 
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr',type=float,default=1e-4,help='Learning rate')#1e-4 means that 10 to the power -4
    parser.add_argument('--lr_decay',type=float,default=5e-5,help='Learning rate decay') 
    parser.add_argument('--epochs',type=int,default=2,help='Number of epochs')
    parser.add_argument('--content_weight',type=float,default=1.0,help='content weights')
    parser.add_argument('--style_weight',type=float,default=10,help='style weights')
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


    content_transform = get_transform(args.final_size,args.crop,args.content_size)#Here we gonna define the resizing and cropping transformation on image  and this we gonna ask from and take from use in parse args
    style_transform = get_transform(args.final_size, args.crop, args.style_size) #This is basically for the style size and upper one is for content size  

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

    mse_loss=torch.nn.MSELoss()#So we gonna use mean sq error loss for the loss calculation

    running_loss=None     #So this is the total loss
    running_content_loss=None #These are the content and style losses
    running_style_loss=None 
    encoder.eval()  #This is to make the model in evaluation mode not in the training mode

    for epoch in range(args.epochs):
        progress_bar=tqdm(zip(content_dataloader,style_dataloader),total=min(len(content_dataloader),len(style_dataloader)))      #This tqdm library is for tracking loops and here for tracking training loops  ka progress like 10%.........25% and so on ....

        running_loss=0     #Setting loss to 0 for every iteration 
        running_content_loss=0 
        running_style_loss=0

        for content_batch,style_batch in progress_bar:  #Now we gonna iterate over this progress_bar and this 1 content and style batches are for 1 epoch 
            content_batch=content_batch.to(device)   #Moving images in content and style batch to GPU->device 
            style_batch=style_batch.to(device)

            c_feats=encoder(content_batch)  #So 1st we are sending every image from style and content batch through encoder
            s_feats=encoder(style_batch)

            t=adaptive_instance_normalization(c_feats[-1],s_feats[-1])  #Then the o/p of the encoder layer is passed through AdaIN layer this is defined in utils and this c_feats[-1] means we get 4 layers from the encoder and so we just only pass 1 to Adain and rest 3 we kept for loss and all

            g=decoder(t) #Then simply we pass the o/p of the AdaIN layer to the decoder and get the o/p image according to the architecture

            g_feats=encoder(g) #so content loss is between g_feats and t(from AdaIN)

            loss_c=mse_loss(g_feats[-1],t)*args.content_weights #So the content loss is calculated by g_feats and t by using MSE loss * content_wts
            loss_s=0
            for g_f,s_f in zip(g_feats,s_feats): #so style loss is calculated by in terms of mean and standard deviation 
                g_mean,g_std=calc_mean_std(g_f)  #mean and standard deviation of the output image
                s_mean,s_std=calc_mean_std(s_f)   #mean and SD of the style image
                loss_s +=mse_loss(g_mean,s_mean)+mse_loss(g_std,s_std) #calculating the loss by both of em

            loss_s=loss_s*args.style_weight
            loss=loss_c+loss_s#This is the final loss the content+style loss

            optimizer.zero_grad() #Optimizer to update wts but ut accumulate teh gradients so we are setting em 0
            loss.backward()
            optimizer.step()   #this is to performs a parameter update based on the current gradient 

            progress_bar.set_description(f'Loss:{loss.item():4f},Content Loss:{loss_c.item():4f},Style Loss:{loss_s.item():4f}')

            running_loss+=loss.item()   #We add these values whenn ever we go to the new batch 
            running_content_loss+=loss_c.item()
            running_style_loss+=loss_s.item()

        scheduler.step()  #After each epoch we gonna update the learning rate using scadular

        running_loss/=len(content_dataloader)
        running_content_loss/=len(content_dataloader)
        running_style_loss/=len(content_dataloader)

        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f"Iter {epoch+1}: Loss:{running_loss:4f}, Content Loss: {running_content_loss:4f}, Style Loss: {running_style_loss:4f}")


if __name__=='__main__':
    main()
