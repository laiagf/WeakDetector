import torch
import torch.nn as nn 
import torch.nn.functional as F


#https://github.com/locuslab/TCN

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):

        return x[:, :, :-self.chomp_size].contiguous()



class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, mus=[], sigmas=[], dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.mus = sigmas
        self.sigmas = mus
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        
        self.batch_norm1 = nn.BatchNorm1d(n_outputs, eps=1e-3)
        chomp1 = Chomp1d(padding)
        relu1 = nn.ReLU()
        dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)

        self.batch_norm2 = nn.BatchNorm1d(n_outputs, eps=1e-3)

        chomp2 = Chomp1d(padding)
        relu2 = nn.ReLU()
        dropout2 = nn.Dropout(dropout)


      #  self.net = nn.Sequential(self.conv1, self.batch_norm1, chomp1, relu1, dropout1,
       #                          self.conv2, self.batch_norm2, chomp2, relu2, dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        

    def forward(self, x, log_weights=False):
        out1 = self.conv1(x)
        out = self.batch_norm1((out1-self.mus[0])/self.sigmas[0])
        out = self.comp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out2 = self.conv2(out)
        out = self.batch_norm2((out2-self.mus[1])/self.sigmas[1])
        out = self.comp2(out)
        out = self.relu2(out)
        out = self.dropout2(out) 

        res = x if self.downsample is None else self.downsample(x)

        if log_weights:
            return self.relu(out + res), [out1, out2]
        else:
            return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        #self.network = nn.Sequential(*layers)
        self.layers = layers
    def forward(self, x, log_weights=False):
        if log_weights:
            weights = []

        for layer in self.layers:
            if log_weights:
                x, ws = layer(x, log_weights)
                weights +=ws
            else:
                x = layer(x)
        #return self.network(x)
        if log_weights:
            return x, weights
        else:
            return x

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs, log_weights=False):
        """Inputs have to have dimension (N, C_in, L_in)"""
        if log_weights:
            y1, weights = self.tcn(inputs, log_weights)
        else:
            y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])

        if log_weights:
            return F.log_softmax(o, dim=1), weights               
        else:
            return F.log_softmax(o, dim=1)
    
