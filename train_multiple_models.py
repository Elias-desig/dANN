from models import dANN, vANN, count_parameters, save_checkpoint
from train import train, test
import multiprocessing
import torchvision.datasets as datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import os
import torch
import csv
import gc


def log_results_to_csv(filename, epoch, test_loss, accuracy, num_dendrites, num_somas, num_params, regularized, model_type, rf_type):
    """Logs training results to a CSV file."""
    # Ensure the file exists and has a header
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            # Write header if file is new
            writer.writerow([
                "Epoch", "Test Loss", "Accuracy", "Num Dendrites", "Num Somas",
                "Total Parameters", "Regularized", "Model Type", "RF Type"
            ])
        # Write the data row
        writer.writerow([
            epoch, test_loss, accuracy, num_dendrites, num_somas,
            num_params, regularized, model_type, rf_type
        ])


def purge_results_for_config(filename, num_dendrites, num_somas, regularized, model_type, rf_type):
    """Remove any existing rows for this config so we can re‐log cleanly."""
    try:
        with open(filename, newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = [
                row for row in reader
                if not (
                    int(row['Num Dendrites']) == num_dendrites and
                    int(row['Num Somas']) == num_somas and
                    row['Regularized'] == str(regularized) and
                    row['Model Type'] == model_type and
                    row['RF Type'] == rf_type
                )
            ]
    except FileNotFoundError:
        # nothing to purge
        return

    # write back only the rows we want to keep
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def train_model(args):
    model = optimizer = None
    try:
        dendrite_count, soma_ratio, device, train_loader, test_loader, model_type, rf_type, regularize = args

        soma_count = int(dendrite_count * soma_ratio)
        if model_type == 'dendritic':
            model = dANN([28,28], dendrite_count, soma_count, 10, rf_type).to(device)
        elif model_type == 'vanilla':
            model = vANN([28,28], dendrite_count, soma_count, 10).to(device)
        else:
            raise NameError('Wrong model type!')
        if regularize:
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.001)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        epochs = 20
        dendritic = model_type == 'dendritic'
        num_params = count_parameters(model)

        csv_filename = './results.csv'
        purge_results_for_config(
            csv_filename,
            num_dendrites=dendrite_count,
            num_somas=soma_count,
            regularized=regularize,
            model_type=model_type,
            rf_type=(rf_type if dendritic else "None")
        )


        for epoch in range(epochs):
            
            train_loss = train(model, train_loader, optimizer, device, masked = dendritic)
            test_loss, accuracy = test(model, test_loader, device)
            if epoch == 9 or epoch == 19:
                try:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        loss=test_loss,
                        num_dendrites=dendrite_count,
                        num_somas=soma_count,
                        dendritic=dendritic,
                        regularized=regularize,
                        rf_type=rf_type,
                        accuracy=accuracy
                    )
                except Exception as e:
                    print(f"Failed to save checkpoint: {e}")
            
            log_results_to_csv(
                filename=csv_filename,
                epoch=epoch,
                test_loss=test_loss,
                accuracy=accuracy,
                num_dendrites=dendrite_count,
                num_somas=soma_count,
                num_params=num_params,
                regularized=regularize,
                model_type=model_type,
                rf_type=rf_type if model_type == 'dendritic' else "None"
            )            

            print(f'train loss: {train_loss}, test loss: {test_loss}, test accuracy: {accuracy}')
    except Exception as e:
        print(f"Error in train_model: {e}")
    finally:
        # free GPU + Python objects
        if optimizer is not None:
            del optimizer
        if model is not None:
            del model
        torch.cuda.empty_cache()
        gc.collect()




if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    try:
        print("Script started...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        train_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.1307], std=[0.3081])
        ])
        os.makedirs('./data', exist_ok=True)
        print("Loading datasets...")
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
        print("Launching Tasks...")
        with multiprocessing.Pool(processes=3) as pool:
            print("Pool created.") 
            pool.map(train_model, tasks)
    except Exception as e:
        print(f"Error occurred: {e}")
