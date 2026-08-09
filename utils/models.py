import torch
import torch.nn as nn


class VGGEncoder(nn.Module):
    def __init__(self, vgg_path):
        super(VGGEncoder, self).__init__()

        self.vgg = nn.Sequential(
            #Block 1     
            nn.Conv2d(3, 3, (1, 1)),  #These 3 arguments are input_channels,output_channels and kernal
            nn.ReflectionPad2d((1, 1, 1, 1)),  #padding k jhaga we yse reflection padding coz its useful in image processing / computer vision because zero padding can create artificial black borders. and ye hoti ye hai k mid yale elem ko as padding use krte hai like normal padding mai we pad with 0 here mid elem ko like 019 to its gonna be 10191
            nn.Conv2d(3, 64, (3, 3)),                                                                                                                                                                                      #                                                                                                       234                 32343
            nn.ReLU(),                      # relu1-1  This means 1st block ka 1st ReLu 
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),                      # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

            # Block 2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),                      # relu2-1 means 2nd block ka 1st relu function
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),                      # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

            # Block 3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),                      # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),                      # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),                      # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),                      # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

            # Block 4
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),                      # relu4-1 (last layer used)
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),                      # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),                      # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),                      # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

            # Block 5
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),                      # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),                      # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),                      # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()                       # relu5-4
        )

        # Load pretrained VGG weights
        self.vgg.load_state_dict(torch.load(vgg_path))  #Isse model mai wts load krenge

        # Keep layers up to relu4-1 (31 layers)
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31]) #This is for breaking sequential and getting the output till relu 4-1 and ys vgg.child krta ye hai ki saare steps saare individual steps ko layers ki taraha treat krta hai and so relu4-1 becomes 30th layer by this logic so we have written [:31]
                                                                  #To aab humera paas 30th layer tk ka calculated data hai so what we gonna do is make a new sequential for this 
    
        enc_layers = list(self.vgg.children())  
        self.enc_1 = nn.Sequential(*enc_layers[:4])  #This is for ki 4th layer which is relu 1-1 tk ka res
        self.enc_2 = nn.Sequential(*enc_layers[4:11])#Then this is relu 2-1 kaa result which is from 4th till 11th layer 
        self.enc_3 = nn.Sequential(*enc_layers[11:18]) #and so on this is form 11th till 18 and all 
        self.enc_4 = nn.Sequential(*enc_layers[18:31])#aand here layer is simply means line of code coz ye vgg.children each line ko layer mai convert krdeta hai

        for name in ['enc_1','enc_2','enc_3','enc_4']:
            for params in getattr(self,name).parameters():
                params.requires_grad=False          #This means pyTorch will not calculate gradients for that tensor/parameter during backpropagation. and coz ye encoder hai pretrained so hume wts update krne bhi nahi h

    def forward(self, input):
        h1=self.enc_1(input)
        h2=self.enc_2(h1)
        h3=self.enc_3(h2)
        h4=self.enc_4(h3)
        return h1,h2,h3,h4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.decoder=nn.Sequential(
            # Block 1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            # Block 2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),

            # Block 3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),

            # Block 4
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),

            # Block 5
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            # Block 6
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),

            # Block 7
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            # Block 8
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),

            # Block 9
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

    def forward(self,input):      #Yha humne requires_grad ko false nahi kiya coz hume wts vagera update krne hai in decoder and so we just simply pass the input to the decoder
        return self.decoder(input)