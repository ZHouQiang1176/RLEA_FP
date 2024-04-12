import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.nn import init


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNDModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        
        feature_output = 12 * 12 * 64
        self.predictor = nn.Sequential(
                    nn.Conv2d(1, 16, 8, stride=4, padding=2), nn.LeakyReLU(),
                    nn.Conv2d(16, 32, 4, stride=2,padding=1), nn.LeakyReLU(),
                    nn.Conv2d(32, 16, 4, stride=2,padding=1), nn.LeakyReLU(),
                    nn.Conv2d(16, 8, 3, stride=1), nn.LeakyReLU(), Flatten(),
                    nn.Linear(8 * 14 * 14, 512), nn.ReLU(),
                    #nn.Linear(7200, 512), nn.ReLU(),
                    nn.Linear(512, 512))
        self.target = nn.Sequential(
                    nn.Conv2d(1, 16, 8, stride=4, padding=2), nn.LeakyReLU(),
                    nn.Conv2d(16, 32, 4, stride=2,padding=1), nn.LeakyReLU(),
                    nn.Conv2d(32, 16, 4, stride=2,padding=1), nn.LeakyReLU(),
                    nn.Conv2d(16, 8, 3, stride=1), nn.LeakyReLU(), Flatten(),
                    nn.Linear(8 * 14 * 14, 512))
                    #nn.Linear(7200, 512))
        # Initialize weights    
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        # Set target parameters as untrainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)
        return predict_feature, target_feature