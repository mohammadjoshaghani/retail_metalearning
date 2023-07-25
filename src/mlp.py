import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, features1=268, features2=128, features3=8):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(features1, features2)
        self.linear2 = torch.nn.Linear(features2, features3)
        
        self.linear1de = torch.nn.Linear(features3, features2)
        self.linear2de = torch.nn.Linear(features2, features1)
    
    def encoder(self,x0): 
        x1 = self.linear1(x0)
        x2 = self.linear2(x1)
        return x2

    def decoder(self,x2): 
        x3 = self.linear1de(x2)
        self.rec_x = self.linear2de(x3)
        return self.rec_x

        
