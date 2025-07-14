import csv
import multiprocessing
from models import dANN, vANN, count_parameters
from train import train, test
import torchvision.datasets as datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import os
import torch

train_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True), 
    v2.Normalize(mean=[0.1307], std=[0.3081])  
])


os.makedirs('./data', exist_ok=True)

def read_csv(filename):
    try:
        with open(filename, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            print(f"Loaded {len(rows)} rows from {filename}.")
            return rows
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return []

csv_filename = './results.csv'
logged_results = read_csv(csv_filename)

paramter_counts = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
soma_ratios = [1 / 2, 1 / 4, 1 / 8]
rf_types = ['global', 'local', 'random']
regularize_options = [True, False]

expected_combinations = []
for parameter_count in paramter_counts:
    for soma_ratio in soma_ratios:
        for rf_type in rf_types:
            expected_combinations.append((parameter_count, soma_ratio, 'dendritic', rf_type, False))
        expected_combinations.append((parameter_count, soma_ratio, 'vanilla', None, False))
        expected_combinations.append((parameter_count, soma_ratio, 'vanilla', None, True))
print(len(expected_combinations))

from train_multiple_models import train_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transforms)
fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=train_transforms)

train_loader = DataLoader(fmnist_train, batch_size=128, shuffle=True)
test_loader = DataLoader(fmnist_test, batch_size=128, shuffle=False)


missing_combinations = []
for param_count, soma_ratio, model_type, rf_type, regularize in expected_combinations:
    soma_count = int(param_count * soma_ratio)
    # scan epochs 0–19
    for epoch in range(20):
        found = any(
            int(row['Num Dendrites']) == param_count
            and int(row['Num Somas']) == soma_count
            and row['Model Type'] == model_type
            and row['RF Type'] == (rf_type if model_type == 'dendritic' else "None")
            and row['Regularized'] == str(regularize)
            and int(row['Epoch']) == epoch
            for row in logged_results
        )
        if not found:
            missing_combinations.append((param_count, soma_ratio, device,
                                         train_loader, test_loader,
                                         model_type, rf_type, regularize))
            break   # stop scanning epochs for this config

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    print(f"Need to retrain {len(missing_combinations)} model configs")
    for args in missing_combinations:
        print("Retraining:", args[:4], "rf_type=", args[6], "reg=", args[7])

    with multiprocessing.Pool(processes=2) as pool:
        pool.map(train_model, missing_combinations)