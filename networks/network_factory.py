import torch
import torch.nn as nn
from torch_geometric.nn import DataParallel

class NetworkFactory():
    def __init__(self, network, device_ids):
        self.model = DataParallel(network, device_ids=device_ids)
        self.model.to(device=torch.device('cuda', device_ids[0]))
        print()
        print("Model info:\n --- Model: {}\n --- Model parameter number: {}".format(type(network), sum(x.numel() for x in self.model.parameters())))
    
    def initialize(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_model(self):
        return self.model
