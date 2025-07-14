import torch 
from torch import nn
import torch.nn.functional as F
from masks import rf_mask, somatic_mask
import os
from datetime import datetime


class dANN(nn.Module):
    def __init__(self, image_size, num_dendrites, num_somas, num_out, type):
        super().__init__()
        self.dendrites = nn.Linear(image_size[0] * image_size[1], num_dendrites)
        self.register_buffer('rf_mask', rf_mask(image_size, num_dendrites, num_somas, type=type).to(self.dendrites.weight.dtype))
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
        return logits

    def apply_masks(self):
        with torch.no_grad():
            self.dendrites.weight.data = self.dendrites.weight.data * self.rf_mask
            self.somas.weight.data = self.somas.weight.data * self.somatic_mask 


class vANN(nn.Module):
    def __init__(self, image_size, num_dendrites, num_somas, num_out):
        super().__init__()
        self.dendrites = nn.Linear(image_size[0] * image_size[1], num_dendrites)
        self.somas = nn.Linear(num_dendrites, num_somas)
        self.classifier = nn.Linear(num_somas, num_out)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.dendrites(x))
        x = F.relu(self.somas(x))
        logits = self.classifier(x)
        return logits                           

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, loss, num_dendrites, num_somas, dendritic, regularized, rf_type=None, accuracy=None, image_size=(28, 28), num_out=10):
    base_dir = './models'
    sub_dir = 'dendritic' if dendritic else 'vanilla'
    checkpoint_dir = os.path.join(base_dir, sub_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)

    config = {
        'image_size': image_size,
        'num_dendrites': num_dendrites,
        'num_somas': num_somas,
        'num_out': num_out,
    }
    if dendritic:
        config['type'] = rf_type

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'config': config,
        'timestamp': datetime.now().strftime("%Y%m%d-%H%M%S"),
    }

    # Save checkpoint
    filename = f"{'dANN' if dendritic else 'vANN'}_checkpoint_{checkpoint['timestamp']}_{checkpoint['epoch']}.pt"
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))