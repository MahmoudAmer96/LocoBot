import torch as trch
import torchvision.datasets as dts
import torchvision.transforms as trnsfrms
import torch.nn as nn
import matplotlib.pyplot as plot

trnsform = trnsfrms.Compose([trnsfrms.ToTensor(), trnsfrms.Normalize((0.7,), (0.7,)),])

mnisttrainset = dts.MNIST(root='./data', train=True, download=True, transform=trnsform)
trainldr = trch.utils.data.DataLoader(mnisttrainset, batch_size=10, shuffle=True)

mnist_testset = dts.MNIST(root='./data', train=False, download=True, transform=trnsform)
testldr = trch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)
trnsform = trnsfrms.Compose([trnsfrms.ToTensor(), trnsfrms.Normalize((0.7,), (0.7,)),])
class classicationmodel(nn.Module):
    def __init__(self):
        super( classicationmodel,self).__init__()
        self.linear1 = nn.Linear(28*28, 100)
        self.linear2 = nn.Linear(100, 50)
        self.final = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, image):
        a = image.view(-1, 28*28)
        a = self.relu(self.linear1(a))
        a = self.relu(self.linear2(a))
        a = self.final(a)
        return a

cmodel = classicationmodel()
print(cmodel) 