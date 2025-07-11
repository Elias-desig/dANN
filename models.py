import torch 
from torch import nn
import torch.nn.functional as F
from masks import rf_mask, somatic_mask


class dANN(nn.Module):
    def __init__(self, image_size, num_dendrites, num_somas, num_out, type):
        super().__init__()
        self.dendrites = nn.Linear(image_size[0] * image_size[1], num_dendrites)
        self.register_buffer('rf_mask', rf_mask(image_size, num_dendrites, type=type).to(self.dendrites.weight.dtype))
        self.dendrites.weight.data = self.dendrites.weight.data * self.rf_mask

        self.somas = nn.Linear(num_dendrites, num_somas)
        self.register_buffer('somatic_mask', somatic_mask(num_dendrites, num_somas).to(self.dendrites.weight.dtype))
        self.somas.weight.data = self.somas.weight.data * self.somatic_mask

        self.classifier = nn.Linear(num_somas, num_out)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.dendrites(x))
        x = F.relu(self.somas(x))
        logits = self.classifier(x)
        out = F.softmax(logits, dim=1)
        return out

    def apply_masks(self):
        with torch.no_grad():
            self.dendrites.weight.data = self.dendrites.weight.data * self.rf_mask
            self.somas.weight.data = self.somas.weight.data * self.somatic_mask 


class vANN(nn.Module):
    def __init__(self, image_size, num_dendrites, num_somas, num_out, type):
        super().__init__()
        self.dendrites = nn.Linear(image_size[0] * image_size[1], num_dendrites)
        self.somas = nn.Linear(num_dendrites, num_somas)
        self.classifier = nn.Linear(num_somas, num_out)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.dendrites(x))
        x = F.relu(self.somas(x))
        logits = self.classifier(x)
        out = F.softmax(logits, dim=1)
        return out                           

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)