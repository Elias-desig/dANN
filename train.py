import torch
import torch.nn.functional as F
from tqdm import tqdm

def train(model, dataloader, optimizer, device):
    model.train()
    loss_total = 0.0

    for (data, target) in (tqdm(dataloader)):

        data = data.to(device)
        target = target.to(device)
    
        optimizer.zero_grad()

        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    
        
        optimizer.step()  # Update the model parameters
        loss_total += loss.item()
        model.apply_masks()
    return loss_total / len(dataloader)

def test(model, dataloader, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for (data, target) in tqdm(dataloader, desc='Testing'):
            data = data.to(device)
            target = target.to(device)


            output = model(data)
            loss = F.cross_entropy(output, target)
            test_loss += loss.item()
            pred_target = torch.argmax(output, dim=1)
            correct += (pred_target == target).sum().item()
            total += target.size(0)
    accuracy = correct / total
    return test_loss / len(dataloader), accuracy