from Nets.theory_UNET import theory_UNET_progressive 

import torch

import random

device = torch.device("cuda:0")

model = theory_UNET_progressive(
    in_channels = 6
)

model = model.to(device)
model.eval()

# dummy inputs
x = torch.ones(2, 6, 128, 128, 128)
x = x.to(device)

features = list()
features.append(torch.zeros(2, 6, 128, 128, 128))
features.append(torch.ones(2, 16, 64, 64, 64)*1)
features.append(torch.ones(2, 32, 32, 32, 32)*2)
features.append(torch.ones(2, 64, 16, 16, 16)*3)
features.append(torch.ones(2, 128, 8, 8, 8)*4)
features.append(torch.ones(2, 256, 8, 8, 8)*5)
features.append(torch.ones(2, 64, 16, 16, 16)*6)
features.append(torch.ones(2, 32, 32, 32, 32)*7)
features.append(torch.ones(2, 16, 64, 64, 64)*8)
features.append(torch.ones(2, 1, 128, 128, 128)*9)
# features = features.to(device)
# print([feature[0,0,0,0,0] for feature in features ])
features = [feature.to(device) for feature in features]

with torch.no_grad():
    pred = model(x, features)
    # pred = model(x)

pred = pred.detach().cpu().numpy()

print(pred[0, 0, 0, 0, 0])