from models import dANN, vANN, count_parameters, save_checkpoint
from train import train, test
import multiprocessing
import torchvision.datasets as datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import os
import torch



def train_model(args):
    # Unpack arguments
    dendrite_count, soma_ratio, device, train_loader, test_loader, model_type, rf_type, regularize = args

    soma_count = int(dendrite_count * soma_ratio)
    if model_type == 'dendritic':
        model = dANN([28,28], dendrite_count, soma_count, 10, rf_type)
    if model_type == 'vanilla':
        model = vANN([28,28], dendrite_count, soma_count, 10)
    else:
        raise NameError('Wrong model type!')
    if regularize:
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, decoupled_weight_decay=.001)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    epochs = 20
    dendritic = model_type == 'dendritic'
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, device)
        test_loss, accuracy = test(model, test_loader, device)
        try:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=test_loss,
                num_dendrites=dendrite_count,
                num_somas=soma_count,
                dendritic=dendritic,
                rf_type=rf_type,
                accuracy=accuracy
            )
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
        print(f'train loss: {train_loss}, test loss: {test_loss}, test accuracy: {accuracy}')        



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.1307], std=[0.3081])
    ])
    os.makedirs('./data', exist_ok=True)

    fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transforms)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=train_transforms)
    train_loader = DataLoader(fmnist_train, batch_size=128, shuffle=True)
    test_loader = DataLoader(fmnist_test, batch_size=128, shuffle=False)

    paramter_counts = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    soma_ratios = [1 / 2, 1 / 4, 1 / 8]
    rf_types = ['global', 'local', 'random']
    regularize_options = [True, False]


    tasks = []
    for parameter_count in paramter_counts:
        for soma_ratio in soma_ratios:
            for rf_type in rf_types:
                tasks.append((parameter_count, soma_ratio, device, train_loader, test_loader, 'dendritic', rf_type, False))
            tasks.append((parameter_count, soma_ratio, device, train_loader, test_loader, 'vanilla', None, False))
            tasks.append((parameter_count, soma_ratio, device, train_loader, test_loader, 'vanilla', None, True))

    # Use multiprocessing Pool to limit simultaneous processes
    with multiprocessing.Pool(processes=3) as pool: 
        pool.map(train_model, tasks)

