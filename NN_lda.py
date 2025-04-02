import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils_data
import scipy.io as sio

class Net(nn.Module):

    def __init__(self, input_size, topic_num):
        super(Net, self).__init__()

        # 3 input image channel, 6 output channels, 5x5 square convolution kernel
        self.linear1 = nn.Linear(input_size, 20)
        
        # 6 input channels, 16 output channels, 5x5 square convolutional kernel
        self.linear2 = nn.Linear(20, topic_num)
        
        

    def forward(self, x):

        x = F.relu(self.linear1(x))
        x =  torch.abs(self.linear2(x))
        return x
       




