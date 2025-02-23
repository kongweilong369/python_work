import torch
from torch import nn
from torch.nn import Sequential


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),# input(3,224,224) output(96,54,54)
            nn.MaxPool2d(kernel_size=3, stride=2),# output(96,26,26)
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),# output(256, 26, 26)
            nn.MaxPool2d(kernel_size=3, stride=2),# output(256, 12, 12)
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),# output(384, 12, 12)
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),# output(384, 12, 12)
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),# output(256, 12, 12)
            nn.MaxPool2d(kernel_size=3, stride=2),# output(256, 5, 5)
            nn.Flatten(),# output(256*5*5)
            nn.Linear(6400, 4096), nn.ReLU(),# output(4096)
            nn.Dropout(p=0.5),#
            nn.Linear(4096, 4096), nn.ReLU(),# output(4096)
            nn.Dropout(p=0.5),#
            nn.Linear(4096, 10)# output(10)
        )
    def forward(self,input):
        output=self.model(input)
        return output

net=AlexNet().model
if __name__=='__main__':
    X = torch.randn(1, 3, 224, 224)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
# print(net)