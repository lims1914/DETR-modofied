import torch
from models.transformer import PrunedLayer
pthfile = "./checkpoint/checkpoint.pth"
net = torch.load(pthfile)
print(net)