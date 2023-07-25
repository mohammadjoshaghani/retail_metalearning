import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMS(nn.Module):
    def __init__(self, in_di, first_hs, second_hs):
      super(LSTMS, self).__init__()
      self.first_hidden_size = first_hs
      self.second_hidden_size = second_hs
      self.input_dimensions = in_di
      self.first_layer = nn.LSTM(input_size = self.input_dimensions, hidden_size = self.first_hidden_size, 
                                 num_layers = 1, batch_first = True)
      self.second_layer = nn.LSTM(input_size = self.first_hidden_size, hidden_size = self.second_hidden_size, 
                                  num_layers = 1, batch_first = True)
    
    def forward(self, x):
      batch_size, seq_len, _ = x.size()
      h_1 = torch.zeros(1, batch_size, self.first_hidden_size)
      c_1 = torch.zeros(1, batch_size, self.first_hidden_size)
      hidden_1 = (h_1, c_1)
      lstm_out, hidden_1 = self.first_layer(x, hidden_1)
      h_2 = torch.zeros(1, batch_size, self.second_hidden_size)
      c_2 = torch.zeros(1, batch_size, self.second_hidden_size)
      hidden_2 = (h_2, c_2)
      lstm_out, hidden_2 = self.second_layer(lstm_out, hidden_2)
     
      return lstm_out, hidden_2


class LSTMS_Decoder(nn.Module):
    def __init__(self, in_di, first_hs, second_hs):
      super(LSTMS_Decoder, self).__init__()
      self.first_hidden_size = first_hs
      self.second_hidden_size = second_hs
      self.input_dimensions = in_di
      self.first_layer = nn.LSTM(input_size = self.input_dimensions, hidden_size = self.first_hidden_size, 
                                 num_layers = 1, batch_first = True)
      self.second_layer = nn.LSTM(input_size = self.first_hidden_size, hidden_size = self.second_hidden_size, 
                                  num_layers = 1, batch_first = True)
    
    def forward(self, x, hidden_1):
        batch_size, seq_len, _ = x.size()
        lstm_out, hidden_1 = self.first_layer(x, hidden_1)
        h_2 = torch.zeros(1, batch_size, self.second_hidden_size)
        c_2 = torch.zeros(1, batch_size, self.second_hidden_size)
        hidden_2 = (h_2, c_2)
        lstm_out, hidden_2 = self.second_layer(lstm_out, hidden_2)
        
        return lstm_out, hidden_2


class EncRNN(nn.Module):
    def __init__(self, in_di, first_hs, second_hs, dout=0.3):
        super(EncRNN, self).__init__()
        self.first_hs=256
        self.rnn = LSTMS(in_di, self.first_hs, second_hs)
        self.dropout = nn.Dropout(dout)

    def forward(self, inputs):
        enc_outs, hidden = self.rnn(inputs)
        return self.dropout(enc_outs), hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim, method):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_dim = hidden_dim

        if method == 'general':
            self.w = nn.Linear(hidden_dim, hidden_dim)
        elif method == 'concat':
            self.w = nn.Linear(hidden_dim*2, hidden_dim)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_dim))

    def forward(self, dec_out, enc_outs):
        if self.method == 'dot':
            attn_energies = self.dot(dec_out, enc_outs) #13,32
        elif self.method == 'general':
            attn_energies = self.general(dec_out, enc_outs)
        elif self.method == 'concat':
            attn_energies = self.concat(dec_out, enc_outs)
        return F.softmax(attn_energies, dim=0)

    def dot(self, dec_out, enc_outs):
        return torch.sum(dec_out*enc_outs, dim=2)

    def general(self, dec_out, enc_outs):
        energy = self.w(enc_outs)
        return torch.sum(dec_out*energy, dim=2)

    def concat(self, dec_out, enc_outs):
        dec_out = dec_out.expand(enc_outs.shape[0], -1, -1)
        energy = torch.cat((dec_out, enc_outs), 2)
        return torch.sum(self.v * self.w(energy).tanh(), dim=2)


class DecRNN(nn.Module):
    def __init__(self, in_di, first_hs, second_hs, dout=0.3, attn='dot'):
        super(DecRNN, self).__init__()

        self.rnn = LSTMS_Decoder(in_di=second_hs, first_hs=256, second_hs=256)
        self.w = nn.Linear(512, in_di)
        self.attn = Attention(second_hs, attn)
        self.dec_input = nn.Linear(in_di, second_hs)
        self.out_projection = nn.Linear(in_di, in_di)
        self.dropout = nn.Dropout(dout)

    def forward(self, inputs, hidden, enc_outs):
        inputs = self.dec_input(inputs)
        dec_out, hidden = self.rnn(inputs, hidden) 

        attn_weights = self.attn(dec_out, enc_outs) 
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outs) 
        cats = self.w(torch.cat((dec_out, context), dim=2))
        pred = self.out_projection(cats.tanh().squeeze(0)) 
        return pred, hidden, context


class Seq2seqAttn(nn.Module):
    def __init__(self, tlen , in_di=5, first_hs=1024, second_hs=256):
        super().__init__()
        self.encoder_in = EncRNN(in_di, first_hs, second_hs)
        self.decoder_in = DecRNN(in_di, first_hs, second_hs)
        self.tlen = tlen

    def forward_lstm_att(self, srcs):

        enc_outs, hidden = self.encoder_in(srcs) 

        dec_inputs = torch.ones_like(srcs[:,0:1,:]) 
        outs = []
        contexts=[]

        for i in range(self.tlen):
            preds, hidden, context = self.decoder_in(dec_inputs, hidden, enc_outs)
            contexts.append(context)
            outs.append(preds)
            dec_inputs = preds
        self.out = torch.stack(outs,dim=1).squeeze(2)
        self.context_enc = torch.stack(contexts,dim=1).squeeze(2)
        del dec_inputs, enc_outs, preds, context, outs, contexts
        # return self.out, self.context_enc
        
    def encoder(self,x):
        _ = self.forward_lstm_att(x)
        return self.context_enc

    def decoder(self,x):
        return self.out      
            

     

if __name__=="__main__":
    input_x = torch.randn(3,48,5)
    model = Seq2seqAttn(tlen=48, in_di=5, first_hs=1024, second_hs=256)
    context = model.encoder(input_x)
    out = model.decoder(context)
    print(out.shape, context.shape)
