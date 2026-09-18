import argparse
import torch
from pathlib import Path
from utils.utils import *
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.models import *
from tqdm import tqdm
from torchvision.utils import save_image



def parse_arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('--content_dir',type=str,default=r'C:\Final-Projects\Image-Transform\Content_data',help='Location of content dataset')
    parser.add_argument('--style_dir',type=str,default=r'C:\Final-Projects\Image-Transform\Style_data',help='Location of style dataset')
    parser.add_argument('--vgg',type=str,default=r'C:\Final-Projects\Image-Transform\vgg_normalised.pth',help='Location for pre-trained VGG')  #pre-trained CNN encoder we gonna use (VGG)
    parser.add_argument('--experiment',type=str,default='experiment1',help='Name of experiment')  #har exp ke baad yha pr save hoga

    #This we gonna take from use the operations we wanna apply on image
    parser.add_argument('--final_size',type=int,default=256,help='size of final image')
    parser.add_argument('--content_size',type=int,default=256,help='size of content image')
    parser.add_argument('--style_size',type=int,default=256,help='size of style image')
    parser.add_argument('--crop', action='store_true', default=True, help='Crop image')#basically jb bhi call --crop to image crop bhi hogi 
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr',type=float,default=1e-4,help='Learning rate')#1e-4 means that 10 to the power -4
    parser.add_argument('--lr_decay',type=float,default=5e-5,help='Learning rate decay') 
    parser.add_argument('--epochs',type=int,default=8,help='Number of epochs')
    parser.add_argument('--content_weight',type=float,default=1.0,help='content weights')
    parser.add_argument('--style_weight',type=float,default=10,help='style weights')    
    parser.add_argument('--log_interval',type=int,default=1,help='Log interval')
    parser.add_argument('--save_interval', type=int, default=1, help='Epoch interval to save checkpoints')
    parser.add_argument('--resume',action='store_true',default=False,help='Resume training')
    parser.add_argument('--decoder_path',type=str,default=None,help='Path to decoder checkpoint')
    parser.add_argument('--optimizer_path',type=str,default=None,help='Path to optimizer checkpoint')
            
    return parser.parse_args()



# Per-iteration LR decay, replaces the old epoch-based LambdaLR scheduler
# Formula matches the original AdaIN paper: lr / (1 + decay * step), applied every batch instead of every epoch
def adjust_learning_rate(optimizer, step, base_lr, decay):
    lr = base_lr / (1.0 + decay * step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def main():
    args=parse_arguments()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir=Path('experiment')/args.experiment #This is for saving all the expreiments with output of that particular exp ---- to experiments naam ka folder bnaga in which args.expreiments ki values store hoti rhengi
    save_dir.mkdir(exist_ok=True,parents=True) #this is for just creating that directory the experiment dir and saving in that 

    #here we goonna save args for any particular experiment like what args we have used to get this output so we save that in a particular txt file
    with open(save_dir/'args.txt','w') as args_file:  #the save_dir we created above
        for key,val in vars(args).items():             #vars is to convert in dictionary we cannot use dictionary directly  key=name and val=actual value of that 
            args_file.write(f"{key}:{val}\n")

                      
    content_transform = get_transform(args.content_size,args.crop,args.final_size)#Here we gonna define the resizing and cropping transformation on image  and this we gonna ask from and take from use in parse args
    style_transform = get_transform(args.style_size, args.crop, args.final_size) #This is basically for the style size and upper one is for content size  


    #abb we gonna load content and style dataset       
                   
                                        #Path,transform for content and style Dataset
    content_dataset=ImageFolderDataset(args.content_dir,content_transform)  #creating datasets of content and style images and like getting data from the folder of content and style 
    style_dataset=ImageFolderDataset(args.style_dir,style_transform) 


    # So after dataset we gonna crete dataloaders simply import from torch
    # num_workers=4 added so CPU-side loading/augmentation happens in parallel instead of blocking the GPU between batches
    content_dataloader = DataLoader(content_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True,num_workers=4)  #shuffel  true means har epoch ke baad dataset gonna shuffel
    style_dataloader = DataLoader(style_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True,num_workers=4)  #and this pin_memory is imp for the transformation from CPU to CGP



    #---------------------------------------------------------------------------------------#
    #Encoder decoder part starts 

                    # vgg ka path pass that we have defined in args
    encoder=VGGEncoder(args.vgg).to(device)   #setting encoder and decoder to the GPU by .to(device)
    decoder=Decoder().to(device)


    #Optimizer for which we gonna use adam optimizer
    optimizer=optim.Adam(decoder.parameters(),lr=args.lr) #optimizer gonna take parameters and learning rate

    # Old epoch-based scheduler removed - lr is now updated per iteration via adjust_learning_rate() below
    # Mixed precision setup - helps fit training on a small (4GB) GPU and speeds things up
    scaler = torch.cuda.amp.GradScaler()
    global_step = 0


    #This is for resuming the training form here 
    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path))
        optimizer.load_state_dict(torch.load(args.optimizer_path))


    #Loss function for this we gonna simply use MSE
    mse_loss=torch.nn.MSELoss()#So we gonna use mean sq error loss for the loss calculation


    #Study later ki eval and training mode mai kya kya diff hota hai
    encoder.eval()  #This is to make the model in evaluation mode not in the training mode


    #Toal 3 losses we have to calculate 1.)Total 2.)Content 3.)Style
    running_loss=None     #So this is the total loss
    running_content_loss=None #These are the content and style losses
    running_style_loss=None 



    for epoch in range(args.epochs):
                             #Progress bar is over dataloaders
        progress_bar=tqdm(zip(content_dataloader,style_dataloader),total=min(len(content_dataloader),len(style_dataloader)))      #This tqdm library is for tracking loops and here for tracking training loops  ka progress like 10%.........25% and so on .... basically progress bar  zip is ki hum content and style dataloader ko combine kr rhe hai and this min(len...,len(.)) coz jo km hoga utne he batches banange


        running_loss=0     #Setting loss to 0 for every iteration 
        running_content_loss=0 
        running_style_loss=0


        #Now we gonna iterate over this progress_bar and this 1 content and style batches are for 1 epoch
        for content_batch,style_batch in progress_bar: 
            content_batch=content_batch.to(device)   #Moving images in content and style batch to GPU->device 
            style_batch=style_batch.to(device)

            # Update lr for this iteration before the forward/backward pass
            adjust_learning_rate(optimizer, global_step, args.lr, args.lr_decay)

            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
            #Yha se hum vo model create kr rhe hai that the one in diagram 1st encoder and 2nd AdAIN layer 3rd Decoder and so on....

            optimizer.zero_grad() #Optimizer to update wts but it accumulate the gradients so we are seting em 0

            # Forward pass wrapped in autocast for mixed precision (fp16 where safe, fp32 where needed)
            with torch.cuda.amp.autocast():
                #So 1st we are sending every image from style and content batch through encoder 
                c_feats=encoder(content_batch)  
                s_feats=encoder(style_batch)


                #Then the o/p of the encoder layer is passed through AdaIN layer this is defined in utils and this c_feats[-1] means we get 4 layers from the encoder and so we just only pass 1 to Adain and rest 3 we kept for loss and all and we pass the depest layer the last one layer
                t=adaptive_instance_normalization(c_feats[-1],s_feats[-1])


                #Then simply we pass the o/p of the AdaIN layer to the decoder and get the o/p image according to the architecture
                g=decoder(t)


                #TO the generated image of the decoder is passed through the VGG encoder again coz we have to calculate the losses
                g_feats=encoder(g) #so content loss is between g_feats and t(from AdaIN)


                #So the content loss is between the g_feat(from last layer g_feats[-1]) and t and we find it by MSE
                loss_c=mse_loss(g_feats[-1],t)*args.content_weight #So the content loss is calculated by g_feats and t by using MSE loss * content_wts


                #And the style loss is mean and StandardDeviation on these feature sets so we gonna loop over g_feats,s_feats and find style loss
                loss_s=0

                for g_f,s_f in zip(g_feats,s_feats): #so style loss is calculated by in terms of mean and standard deviation for all 4 layers so thats y for loop here
                    g_mean,g_std=calc_mean_std(g_f)  #mean and standard deviation of the output image
                    s_mean,s_std=calc_mean_std(s_f)   #mean and SD of the style image
                    loss_s +=mse_loss(g_mean,s_mean)+mse_loss(g_std,s_std) #calculating the loss by both of em

                #Now as we have the total style loss in loss_s so now we gonna mul it with wts
                loss_s=loss_s*args.style_weight

                #This is the final loss the content+style loss
                loss=loss_c+loss_s

            # backward/step now go through the GradScaler instead of calling loss.backward()/optimizer.step() directly
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            #Basically isse we track how owr loss is gonna change with each batch 
            progress_bar.set_description(f'Loss:{loss.item():4f},Content Loss:{loss_c.item():4f},Style Loss:{loss_s.item():4f}')

            running_loss+=loss.item()   #We add these values whenn ever we go to the new batch 
            running_content_loss+=loss_c.item()
            running_style_loss+=loss_s.item()


        # Old scheduler.step() removed - lr is now decayed every iteration inside the batch loop instead


        #This is to get the avg loss value 
        running_loss/=len(content_dataloader)
        running_content_loss/=len(content_dataloader)
        running_style_loss/=len(content_dataloader)


        #This gonna give loss after each epoch
        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f"Iter {epoch+1}: Loss:{running_loss:4f}, Content Loss: {running_content_loss:4f}, Style Loss: {running_style_loss:4f}")


        if (epoch+1)% args.save_interval==0:
            torch.save(decoder.state_dict(),save_dir/f'decoder_{epoch+1}.pth')  #so for save checkpoints we have inbuild function called save_interval and in that torch.save and this gonna save the decoder values coz encoder is freezed 
            torch.save(optimizer.state_dict(),save_dir/f'optimizer_{epoch+1}.pth') #This is to resume the fuhnctionality from the saved point so we save the optimizer also  

            with torch.no_grad():
                output=torch.cat([content_batch,style_batch,g],dim=0)  #So this is to concatinate all the otputs
                save_image(output,save_dir/f'output_{epoch+1}.png',nrow=args.batch_size)



if __name__=='__main__':
    main()