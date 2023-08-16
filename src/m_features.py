import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from lstm_att import Seq2seqAttn
from metats.features.deep import Encoder_Decoder_TCN
from mlp import MLP

class Meta_F_Runner():    
    def __init__(self, model,epochs,lr,weightDecay,batchsize,device):
        self.epochs = epochs
        self.model = model
        self.batchsize = batchsize
        self.rec_loss = torch.nn.MSELoss()
        self.lr = lr
        self.weightDecay = weightDecay 
        self.device=device
        
    def train(self,xx):
        _ = self.get_dataloader(xx)
        self.optim = torch.optim.Adam(list(self.model.parameters()),
                                        lr=self.lr, weight_decay= self.weightDecay)
        self.optim.zero_grad()
        for epch in range(self.epochs):
            losses = []
            for idx , (x,) in enumerate(self.dataLoader):
                x = x.to(self.device) 
                latent = self.model.encoder(x)
                re_x = self.model.decoder(latent)
                # get loss and gradients
                loss = self.rec_loss(x, re_x)
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
                losses.append(loss.detach().cpu())
            print(np.mean(losses))            
        
    def evaluate(self, x):
        _ = self.get_dataloader(x)
        losses=[]
        latents=[]
        for idx , (x,) in enumerate(self.dataLoader):
            with torch.no_grad():
                x = x.to(self.device)
                latent = self.model.encoder(x)
                re_x = self.model.decoder(latent)
                # get loss
                loss = self.rec_loss(re_x, x)
                latents.append(latent.detach())
                losses.append(loss.detach().unsqueeze(0))
        latent = torch.cat(latents,dim=0)
        lossf = torch.cat(losses,dim=0)
        self.latent  = latent.cpu()
        self.loss = torch.mean(lossf) 

    def run(self, x, gradient):
        # if gradient=True :optimize NN parameters
        # if gradient=False: get output from NN
        self.model.to(self.device)
        if gradient:
            self.train(x)
            self.evaluate(x)
        else:
            self.evaluate(x)    
    
    def get_dataloader(self, x):
        t = TensorDataset(x)
        self.dataLoader = DataLoader(t, batch_size=self.batchsize, shuffle=False)


class Meta_Features():
    def __init__(self, device, lstm_att_features, tcn_feaures, input_length,epochs,lr,weightDecay,batchsize):
        self.input_length = input_length
        self.tcn = Encoder_Decoder_TCN(tcn_feaures, input_length,             
                            hidden_layers=(128,64))
        self.lstm_att = Seq2seqAttn(tlen=input_length, in_di=lstm_att_features, first_hs=1024, second_hs=256)
        f = self._get_mlp_input_shape()
        self.mlp = MLP(features1=272, features2=128, features3=8)

        arg = (epochs,lr,weightDecay,batchsize,device)
        # TCN:
        self.runner_tcn = Meta_F_Runner(self.tcn,*arg)
        # LSTM_att
        self.runner_lstm_att = Meta_F_Runner(self.lstm_att,*arg)
        # MLP
        self.runner_mlp = Meta_F_Runner(self.mlp,*arg)

    def feed_inputs(self, tcn_in, lstm_att_in, gradient):
        print("\n start tcn:")
        self.runner_tcn.run(tcn_in, gradient)
        self.runner_tcn.latent = self.runner_tcn.latent.unsqueeze(1).view(tcn_in.size(0),self.input_length,-1)
        
        print("\n start lstm_att:")
        self.runner_lstm_att.run(lstm_att_in, gradient)
        
        print("\n start mlp:")
        mlp_in = torch.concat((self.runner_lstm_att.latent,self.runner_tcn.latent),dim=2).detach()
        self.runner_mlp.run(mlp_in, gradient)

    def _get_mlp_input_shape(self):
         _, channel_shape = self.tcn.encoder._encoder_dim()
         return channel_shape

if  __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tcn = torch.randn(10,48,4)
    X_lstm_att = torch.randn(10,48,1)

    mfeatures = Meta_Features(device,X_lstm_att.size(2), X_tcn.size(2), X_tcn.size(1),3,0.06,0.03,2)
    # mfeatures.feed_inputs(tcn_in=X_tcn, lstm_att_in=X_lstm_att, gradient=True)
    mfeatures.feed_inputs(tcn_in=X_tcn, lstm_att_in=X_lstm_att, gradient=False)

    print(mfeatures.runner_tcn.latent.shape)
    print(mfeatures.runner_lstm_att.latent.shape)
    print(mfeatures.runner_mlp.latent.shape)