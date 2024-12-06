import torch

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader

# this is one of Hyper parameter, but let's select given below value
batch_size = 512

from torchvision.utils import make_grid
# this will help us to create Grid of images

import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
from deap import base, creator, tools, algorithms
import timeit
import struct

class LeNet5(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(6, 16, kernel_size = 5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )



    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logit = self.classifier(x)
        return logit

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def accuracy(output, labels):
    _, preds = torch.max(output, dim = 1)

    return torch.sum(preds == labels).item() / len(preds)


device = get_default_device()
device

def eevaluate(model, loss_fn, val_dl, metric = None, device='cuda'):

    with torch.no_grad():

        results = [loss_batch(model, loss_fn, x, y, metric = metric) for x, y in val_dl]

        losses, nums, metrics = zip(*results)

        total = np.sum(nums)

        avg_loss = np.sum(np.multiply(losses, nums)) / total

        avg_metric = None

        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total

    return avg_loss, total, avg_metric

def loss_batch(model, loss_func, x, y, opt = None, metric = None):

    pred = model(x)

    loss = loss_func(pred, y)

    if opt is not None:

        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None

    if metric is not None:

        metric_result = metric(pred, y)

    return loss.item(), len(x), metric_result

model = torch.load('lenet.pth', map_location=device)

import torchvision
# transform is used to convert data into Tensor form with transformations
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

trans = transforms.Compose([
    # To resize image
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    # To normalize image
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.MNIST(
root = './data',
train = True,
download = True,
transform = trans
)

test_set = torchvision.datasets.MNIST(
root = './data',
train = False,
download = True,
transform = trans
)

test_loader = DeviceDataLoader(DataLoader(test_set, batch_size=256), device)
result = eevaluate(model, F.cross_entropy, test_loader, metric = accuracy)
result
Accuracy = result[2] * 100
Accuracy
loss = result[0]
print("Total Losses: {}, Accuracy: {}".format(loss, Accuracy))


best_ind0001 = [('features.0.weight', 116), ('features.0.weight', 40), ('features.3.weight', 1809), ('classifier.0.weight', 14352), ('classifier.4.weight', 1), ('features.0.weight', 144)]
best_ind001 = [('classifier.2.weight', 4989), ('classifier.0.weight', 30234), ('features.3.weight', 1730), ('classifier.0.weight', 40182), ('features.0.weight', 58), ('features.3.weight', 1898), ('classifier.2.weight', 1481), ('classifier.4.weight', 541), ('features.3.weight', 2397), ('classifier.4.weight', 138), ('classifier.0.weight', 26271), ('classifier.0.weight', 12746), ('features.0.weight', 45), ('classifier.2.weight', 1114), ('features.3.weight', 2098), ('features.0.weight', 64), ('classifier.0.weight', 45332), ('features.3.weight', 249), ('features.0.weight', 29), ('classifier.2.weight', 5946), ('features.0.weight', 134), ('classifier.2.weight', 8383), ('classifier.4.weight', 163), ('classifier.2.weight', 9016), ('classifier.2.weight', 719), ('features.3.weight', 1372), ('classifier.4.weight', 696), ('features.3.weight', 664), ('features.3.weight', 732), ('features.0.weight', 107), ('classifier.0.weight', 20284), ('classifier.2.weight', 4731), ('features.3.weight', 481), ('features.3.weight', 906), ('features.0.weight', 108), ('features.0.weight', 117), ('classifier.2.weight', 6458), ('classifier.4.weight', 511), ('features.0.weight', 126), ('features.3.weight', 888), ('classifier.4.weight', 165), ('classifier.2.weight', 7434), ('classifier.2.weight', 5238), ('features.0.weight', 1), ('features.3.weight', 2241), ('classifier.0.weight', 22175), ('classifier.4.weight', 839), ('features.3.weight', 1093), ('features.3.weight', 2177), ('features.0.weight', 95), ('features.0.weight', 36), ('features.0.weight', 46), ('classifier.2.weight', 5615), ('classifier.0.weight', 13872), ('features.0.weight', 68), ('classifier.4.weight', 213), ('classifier.2.weight', 545), ('classifier.4.weight', 676), ('classifier.4.weight', 586), ('classifier.4.weight', 181), ('classifier.2.weight', 9485)]
best_ind01 = [('features.3.weight', 1127), ('classifier.2.weight', 8585), ('features.0.weight', 28), ('features.0.weight', 30), ('classifier.2.weight', 2476), ('features.0.weight', 133), ('classifier.4.weight', 808), ('classifier.2.weight', 4185), ('classifier.0.weight', 27617), ('classifier.4.weight', 786), ('features.3.weight', 368), ('classifier.2.weight', 9171), ('features.0.weight', 88), ('features.0.weight', 99), ('classifier.0.weight', 40362), ('features.0.weight', 4), ('classifier.2.weight', 772), ('classifier.0.weight', 18228), ('classifier.0.weight', 7601), ('classifier.4.weight', 707), ('classifier.2.weight', 4728), ('classifier.4.weight', 185), ('features.3.weight', 102), ('classifier.2.weight', 6975), ('features.0.weight', 3), ('features.0.weight', 122), ('classifier.0.weight', 33921), ('classifier.0.weight', 10744), ('classifier.0.weight', 10578), ('classifier.0.weight', 26811), ('classifier.2.weight', 1955), ('classifier.4.weight', 223), ('features.0.weight', 97), ('classifier.0.weight', 31955), ('features.3.weight', 2375), ('features.0.weight', 135), ('classifier.2.weight', 5778), ('features.3.weight', 1393), ('classifier.4.weight', 669), ('classifier.4.weight', 573), ('features.3.weight', 663), ('classifier.0.weight', 13663), ('classifier.0.weight', 43704), ('classifier.4.weight', 740), ('classifier.0.weight', 7435), ('features.3.weight', 781), ('features.3.weight', 283), ('classifier.2.weight', 9791), ('classifier.0.weight', 17590), ('features.3.weight', 546), ('features.3.weight', 751), ('classifier.0.weight', 27216), ('classifier.4.weight', 224), ('classifier.0.weight', 23328), ('features.0.weight', 107), ('classifier.4.weight', 508), ('classifier.4.weight', 528), ('features.3.weight', 2034), ('classifier.2.weight', 7895), ('features.3.weight', 219), ('classifier.0.weight', 15072), ('classifier.4.weight', 509), ('features.0.weight', 128), ('classifier.4.weight', 561), ('classifier.0.weight', 9287), ('classifier.2.weight', 1820), ('classifier.2.weight', 6970), ('classifier.0.weight', 27279), ('classifier.0.weight', 23281), ('classifier.0.weight', 14284), ('classifier.4.weight', 166), ('classifier.4.weight', 267), ('features.0.weight', 52), ('classifier.0.weight', 31912), ('classifier.2.weight', 9794), ('classifier.2.weight', 4911), ('classifier.0.weight', 45464), ('classifier.0.weight', 20068), ('features.0.weight', 106), ('classifier.2.weight', 8482), ('classifier.4.weight', 605), ('classifier.0.weight', 9797), ('classifier.0.weight', 14206), ('features.3.weight', 2110), ('features.3.weight', 1625), ('classifier.4.weight', 237), ('features.0.weight', 141), ('features.0.weight', 18), ('features.0.weight', 138), ('classifier.2.weight', 2864), ('features.0.weight', 48), ('classifier.4.weight', 556), ('features.3.weight', 69), ('features.3.weight', 146), ('features.0.weight', 90), ('classifier.2.weight', 5594), ('classifier.2.weight', 2693), ('features.3.weight', 2199), ('classifier.2.weight', 2016), ('classifier.2.weight', 781), ('features.0.weight', 139), ('classifier.2.weight', 1144), ('classifier.4.weight', 110), ('features.3.weight', 1766), ('classifier.0.weight', 3880), ('classifier.2.weight', 9709), ('classifier.0.weight', 35453), ('features.0.weight', 92), ('features.0.weight', 131), ('classifier.0.weight', 3041), ('classifier.0.weight', 42210), ('classifier.2.weight', 7992), ('features.3.weight', 1450), ('features.0.weight', 60), ('features.0.weight', 131), ('classifier.4.weight', 309), ('classifier.4.weight', 471), ('classifier.0.weight', 4191), ('classifier.2.weight', 3665), ('features.3.weight', 675), ('classifier.4.weight', 320), ('classifier.0.weight', 26633), ('features.3.weight', 942), ('classifier.2.weight', 6678), ('features.0.weight', 47), ('features.3.weight', 1374), ('features.3.weight', 379), ('classifier.2.weight', 9400), ('features.3.weight', 432), ('features.0.weight', 142), ('features.3.weight', 2189), ('classifier.4.weight', 71), ('features.3.weight', 112), ('classifier.0.weight', 4862), ('features.0.weight', 39), ('classifier.2.weight', 1407), ('classifier.4.weight', 18), ('classifier.0.weight', 42012), ('classifier.4.weight', 89), ('classifier.0.weight', 30130), ('classifier.4.weight', 429), ('features.0.weight', 85), ('classifier.2.weight', 2303), ('features.3.weight', 888), ('features.0.weight', 65), ('classifier.2.weight', 9448), ('classifier.2.weight', 6618), ('classifier.4.weight', 539), ('classifier.0.weight', 47587), ('classifier.0.weight', 12941), ('classifier.0.weight', 42151), ('classifier.2.weight', 4486), ('classifier.0.weight', 33161), ('features.3.weight', 1327), ('classifier.0.weight', 26404), ('features.3.weight', 1953), ('features.3.weight', 1017), ('features.3.weight', 40), ('features.0.weight', 57), ('features.0.weight', 69), ('classifier.2.weight', 1623), ('classifier.2.weight', 6635), ('features.3.weight', 964), ('classifier.0.weight', 42741), ('features.3.weight', 1499), ('features.3.weight', 191), ('features.0.weight', 32), ('classifier.4.weight', 826), ('features.0.weight', 73), ('features.0.weight', 0), ('features.0.weight', 22), ('features.3.weight', 1408), ('features.3.weight', 211), ('classifier.2.weight', 8235), ('features.0.weight', 137), ('classifier.2.weight', 4784), ('features.0.weight', 77), ('classifier.2.weight', 5700), ('classifier.2.weight', 6842), ('features.3.weight', 1845), ('features.0.weight', 126), ('features.0.weight', 66), ('features.3.weight', 1944), ('features.3.weight', 1867), ('classifier.0.weight', 5313), ('classifier.0.weight', 15466), ('classifier.0.weight', 15689), ('classifier.0.weight', 23037), ('classifier.0.weight', 28050), ('features.3.weight', 516), ('features.3.weight', 2193), ('classifier.0.weight', 30028), ('classifier.2.weight', 6340), ('features.0.weight', 14), ('classifier.4.weight', 498), ('classifier.0.weight', 37824), ('features.0.weight', 147), ('features.0.weight', 39), ('classifier.2.weight', 1767), ('classifier.4.weight', 231), ('classifier.0.weight', 36364), ('features.0.weight', 94), ('features.0.weight', 76), ('features.0.weight', 139), ('classifier.0.weight', 40774), ('classifier.0.weight', 3717), ('features.0.weight', 79), ('classifier.4.weight', 619), ('classifier.2.weight', 5491), ('classifier.2.weight', 127), ('classifier.4.weight', 256), ('classifier.2.weight', 9227), ('classifier.0.weight', 17749), ('classifier.0.weight', 39159), ('classifier.0.weight', 24761), ('classifier.4.weight', 478), ('classifier.2.weight', 5907), ('classifier.0.weight', 28076), ('features.0.weight', 41), ('features.3.weight', 1194), ('features.0.weight', 11), ('features.0.weight', 83), ('features.3.weight', 732), ('classifier.4.weight', 26), ('classifier.0.weight', 1661), ('features.0.weight', 91), ('features.0.weight', 123), ('classifier.2.weight', 2703), ('classifier.4.weight', 735), ('classifier.0.weight', 11184), ('features.0.weight', 25), ('features.3.weight', 447), ('classifier.2.weight', 2239), ('features.3.weight', 172), ('classifier.4.weight', 307), ('features.3.weight', 1047), ('features.3.weight', 767), ('features.0.weight', 132), ('features.3.weight', 1216), ('features.0.weight', 140), ('classifier.0.weight', 30357), ('classifier.2.weight', 8389), ('classifier.2.weight', 3457), ('classifier.0.weight', 7234), ('features.3.weight', 1826), ('features.3.weight', 851), ('features.0.weight', 116), ('features.3.weight', 1291), ('classifier.2.weight', 1631), ('features.3.weight', 1598), ('features.0.weight', 128), ('classifier.2.weight', 7640), ('classifier.0.weight', 37871), ('classifier.2.weight', 8421), ('classifier.4.weight', 368), ('features.3.weight', 1488), ('features.3.weight', 1314), ('classifier.4.weight', 326), ('features.3.weight', 1304), ('classifier.0.weight', 41641), ('classifier.4.weight', 584), ('classifier.0.weight', 7859), ('classifier.0.weight', 17711), ('features.0.weight', 74), ('features.3.weight', 1724), ('features.0.weight', 73), ('features.3.weight', 2242), ('classifier.2.weight', 2083), ('classifier.0.weight', 15419), ('features.0.weight', 138), ('classifier.4.weight', 301), ('classifier.4.weight', 774), ('classifier.2.weight', 533), ('classifier.2.weight', 4311), ('features.0.weight', 76), ('classifier.0.weight', 10106), ('classifier.0.weight', 45242), ('features.0.weight', 145), ('classifier.2.weight', 9486), ('features.3.weight', 1016), ('classifier.2.weight', 5438), ('classifier.2.weight', 9403), ('features.0.weight', 25), ('classifier.0.weight', 23296), ('features.0.weight', 135), ('classifier.4.weight', 1), ('classifier.2.weight', 3310), ('classifier.4.weight', 347), ('classifier.4.weight', 295), ('classifier.0.weight', 17796), ('classifier.0.weight', 442), ('classifier.0.weight', 46301), ('features.3.weight', 1593), ('classifier.0.weight', 15658), ('classifier.2.weight', 752), ('classifier.0.weight', 31071), ('classifier.2.weight', 8230), ('features.0.weight', 141), ('classifier.0.weight', 25977), ('classifier.4.weight', 241), ('classifier.0.weight', 3552), ('classifier.0.weight', 6407), ('classifier.4.weight', 242), ('classifier.0.weight', 20183), ('features.0.weight', 46), ('classifier.2.weight', 5937), ('classifier.0.weight', 17159), ('classifier.2.weight', 866), ('features.0.weight', 70), ('features.3.weight', 1390), ('features.0.weight', 61), ('features.3.weight', 1752), ('classifier.4.weight', 250), ('classifier.2.weight', 8911), ('features.3.weight', 1225), ('features.0.weight', 39), ('features.0.weight', 31), ('features.0.weight', 68), ('classifier.0.weight', 19535), ('classifier.4.weight', 532), ('features.0.weight', 73), ('classifier.2.weight', 431), ('classifier.2.weight', 5889), ('features.3.weight', 1254), ('classifier.4.weight', 218), ('features.0.weight', 108), ('classifier.2.weight', 7160), ('features.3.weight', 1269), ('classifier.4.weight', 127), ('classifier.2.weight', 6159), ('features.0.weight', 95), ('classifier.0.weight', 32307), ('classifier.2.weight', 3115), ('classifier.4.weight', 40), ('classifier.0.weight', 23513), ('classifier.0.weight', 41787), ('classifier.0.weight', 12586), ('classifier.2.weight', 3050), ('features.3.weight', 790), ('features.0.weight', 54), ('features.0.weight', 39), ('features.0.weight', 11), ('classifier.4.weight', 127), ('features.0.weight', 10), ('features.3.weight', 2222), ('classifier.2.weight', 7721), ('features.3.weight', 2245), ('features.0.weight', 8), ('classifier.2.weight', 8183), ('classifier.4.weight', 615), ('classifier.4.weight', 96), ('classifier.4.weight', 707), ('classifier.2.weight', 9043), ('classifier.2.weight', 177), ('classifier.2.weight', 6317), ('classifier.2.weight', 8345), ('classifier.4.weight', 305), ('features.3.weight', 803), ('features.0.weight', 110), ('classifier.0.weight', 2128), ('classifier.4.weight', 555), ('features.3.weight', 922), ('classifier.4.weight', 67), ('features.3.weight', 2008), ('features.3.weight', 1201), ('classifier.4.weight', 764), ('features.0.weight', 27), ('classifier.0.weight', 3569), ('classifier.0.weight', 19980), ('features.3.weight', 942), ('classifier.0.weight', 24681), ('classifier.4.weight', 529), ('classifier.4.weight', 533), ('classifier.0.weight', 31355), ('features.3.weight', 2110), ('classifier.0.weight', 16262), ('classifier.4.weight', 324), ('features.3.weight', 1995), ('features.3.weight', 1505), ('classifier.4.weight', 12), ('features.0.weight', 86), ('features.3.weight', 1510), ('classifier.4.weight', 288), ('classifier.4.weight', 423), ('features.3.weight', 72), ('features.0.weight', 52), ('features.3.weight', 920), ('features.0.weight', 41), ('features.0.weight', 28), ('classifier.4.weight', 138), ('classifier.2.weight', 8780), ('classifier.2.weight', 991), ('features.0.weight', 125), ('features.3.weight', 2078), ('classifier.0.weight', 21186), ('classifier.0.weight', 23820), ('classifier.0.weight', 3232), ('classifier.0.weight', 6236), ('classifier.4.weight', 94), ('classifier.2.weight', 6175), ('features.0.weight', 77), ('features.3.weight', 1581), ('classifier.2.weight', 2270), ('classifier.0.weight', 20032), ('features.3.weight', 1687), ('classifier.2.weight', 7822), ('features.0.weight', 65), ('features.0.weight', 22), ('features.0.weight', 128), ('features.0.weight', 13), ('features.3.weight', 1609), ('classifier.0.weight', 40496), ('features.0.weight', 51), ('classifier.0.weight', 43016), ('classifier.4.weight', 488), ('classifier.0.weight', 34476), ('classifier.4.weight', 367), ('classifier.2.weight', 4356), ('features.3.weight', 1523), ('features.3.weight', 2013), ('classifier.0.weight', 14076), ('features.0.weight', 113), ('classifier.2.weight', 9571), ('features.0.weight', 1), ('classifier.0.weight', 15131), ('classifier.2.weight', 2865), ('classifier.0.weight', 26875), ('classifier.0.weight', 25224), ('features.3.weight', 853), ('classifier.2.weight', 3242), ('classifier.0.weight', 12549), ('classifier.0.weight', 35826), ('classifier.4.weight', 622), ('features.0.weight', 15), ('features.0.weight', 7), ('features.0.weight', 43), ('classifier.0.weight', 11597), ('features.0.weight', 18), ('features.3.weight', 1840), ('classifier.2.weight', 6152), ('features.0.weight', 99), ('classifier.2.weight', 2176), ('classifier.0.weight', 43896), ('classifier.4.weight', 379), ('classifier.2.weight', 8913), ('classifier.2.weight', 4847), ('classifier.0.weight', 7286), ('features.0.weight', 33), ('classifier.4.weight', 318), ('classifier.4.weight', 293), ('features.3.weight', 382), ('classifier.2.weight', 2874), ('classifier.2.weight', 4153), ('features.3.weight', 318), ('classifier.4.weight', 262), ('features.0.weight', 73), ('classifier.2.weight', 9126), ('classifier.4.weight', 687), ('features.0.weight', 51), ('classifier.2.weight', 7862), ('features.0.weight', 111), ('features.0.weight', 106), ('classifier.4.weight', 301), ('classifier.4.weight', 684), ('classifier.4.weight', 643), ('classifier.4.weight', 613), ('features.0.weight', 39), ('classifier.2.weight', 3858), ('classifier.4.weight', 704), ('features.3.weight', 1849), ('classifier.4.weight', 315), ('features.0.weight', 78), ('classifier.0.weight', 42292), ('classifier.0.weight', 208), ('classifier.4.weight', 476), ('features.3.weight', 864), ('features.3.weight', 188), ('classifier.4.weight', 533), ('classifier.2.weight', 760), ('features.3.weight', 2105), ('features.3.weight', 1122), ('features.3.weight', 2239), ('classifier.0.weight', 10214), ('classifier.2.weight', 974), ('classifier.0.weight', 40306), ('classifier.2.weight', 3790), ('classifier.0.weight', 20060), ('classifier.0.weight', 25551), ('features.3.weight', 1206), ('classifier.2.weight', 4274), ('classifier.2.weight', 4218), ('classifier.4.weight', 443), ('classifier.0.weight', 37201), ('features.0.weight', 63), ('features.0.weight', 21), ('features.0.weight', 18), ('classifier.0.weight', 8330), ('features.3.weight', 303), ('classifier.4.weight', 95), ('classifier.0.weight', 38331), ('classifier.0.weight', 35424), ('classifier.4.weight', 73), ('features.3.weight', 800), ('classifier.2.weight', 9189), ('classifier.4.weight', 421), ('classifier.2.weight', 542), ('classifier.4.weight', 672), ('classifier.4.weight', 177), ('features.3.weight', 606), ('features.3.weight', 1521), ('features.0.weight', 18), ('features.3.weight', 2398), ('classifier.0.weight', 46551), ('features.3.weight', 707), ('classifier.0.weight', 47191), ('classifier.0.weight', 37707), ('classifier.0.weight', 23697), ('classifier.2.weight', 315), ('features.3.weight', 327), ('classifier.2.weight', 6588), ('classifier.4.weight', 674), ('features.0.weight', 14), ('features.0.weight', 54), ('features.3.weight', 1139), ('classifier.2.weight', 882), ('classifier.2.weight', 3080), ('features.0.weight', 130), ('features.0.weight', 74), ('features.0.weight', 71), ('features.3.weight', 912), ('classifier.2.weight', 5867), ('classifier.2.weight', 4517), ('classifier.0.weight', 44605), ('classifier.0.weight', 24516), ('features.0.weight', 113), ('features.3.weight', 793), ('features.0.weight', 17), ('classifier.2.weight', 5549), ('classifier.0.weight', 33277), ('classifier.2.weight', 6424), ('classifier.0.weight', 29406), ('classifier.4.weight', 92), ('classifier.0.weight', 36925), ('features.0.weight', 107), ('classifier.2.weight', 5786), ('classifier.0.weight', 41680), ('features.0.weight', 112), ('features.3.weight', 2284), ('classifier.2.weight', 4105), ('features.3.weight', 313), ('features.3.weight', 1778), ('features.0.weight', 29), ('classifier.2.weight', 1524), ('classifier.0.weight', 31700), ('features.3.weight', 710), ('features.3.weight', 1163), ('classifier.0.weight', 13135), ('features.0.weight', 144), ('classifier.2.weight', 5199), ('features.0.weight', 98), ('classifier.0.weight', 19044), ('features.0.weight', 140), ('features.0.weight', 85), ('classifier.0.weight', 686), ('features.0.weight', 132), ('classifier.0.weight', 47601), ('features.3.weight', 2120), ('features.0.weight', 69), ('features.3.weight', 77), ('classifier.2.weight', 6518), ('features.0.weight', 9), ('classifier.4.weight', 474), ('classifier.4.weight', 554), ('classifier.4.weight', 436), ('classifier.0.weight', 37758), ('features.0.weight', 57), ('features.0.weight', 26), ('classifier.4.weight', 511), ('features.0.weight', 133), ('classifier.2.weight', 8508), ('features.0.weight', 85), ('classifier.2.weight', 4274), ('classifier.4.weight', 697), ('features.0.weight', 108), ('classifier.2.weight', 1103), ('classifier.2.weight', 5251), ('classifier.2.weight', 7735), ('features.0.weight', 147), ('classifier.4.weight', 440), ('classifier.2.weight', 8065), ('classifier.4.weight', 547), ('classifier.0.weight', 7749), ('classifier.2.weight', 9018), ('features.3.weight', 94), ('features.3.weight', 2280), ('classifier.0.weight', 9565), ('classifier.4.weight', 374), ('classifier.2.weight', 10026), ('classifier.0.weight', 3955), ('features.0.weight', 86), ('classifier.0.weight', 45844), ('features.0.weight', 94), ('features.0.weight', 102), ('classifier.4.weight', 803), ('classifier.4.weight', 570), ('classifier.0.weight', 5861), ('features.0.weight', 139), ('classifier.4.weight', 656), ('features.0.weight', 20), ('classifier.2.weight', 8031), ('classifier.4.weight', 501), ('features.3.weight', 1113), ('classifier.4.weight', 56), ('classifier.4.weight', 793)]


######....................parameter number............................######
layer_names = [name for name in model.state_dict().keys() if 'weight' in name]
print(layer_names)
num_weights_per_layer = {name: model.state_dict()[name].numel() for name in layer_names}
weight_count = {
    'features.0.weight': 0,
    'features.3.weight': 0,
    'classifier.0.weight': 0,
    'classifier.2.weight': 0,
    'classifier.4.weight': 0,
}
for layer in layer_names:
  weight_count[layer] = num_weights_per_layer[layer]
  print(layer, weight_count[layer])


##################LVF################


vul_weight_count = {
    'features.0.weight': 0,
    'features.3.weight': 0,
    'classifier.0.weight': 0,
    'classifier.2.weight': 0,
    'classifier.4.weight': 0,
}


# Iterate over the model layers
for (layer, index) in best_ind01:
    vul_weight_count[layer] += 1

print("..............Vulnerable Weights................")

# Output the result
for layer in layer_names:
    print(f'{vul_weight_count[layer]} weights for {layer}')

LVF = {
    'features.0.weight': 0,
    'features.3.weight': 0,
    'classifier.0.weight': 0,
    'classifier.2.weight': 0,
    'classifier.4.weight': 0,
}

print("..............LVF................")

for layer in layer_names:
  LVF[layer] = vul_weight_count[layer] / weight_count[layer]
  print(layer, LVF[layer])











