import numpy as np
import scipy.ndimage as scn
import torch

def rf_mask(image_size, num_dendrites, num_somas, type='local', rf_size=16):
    seed = 1
    rng = np.random.default_rng(seed)
    layer_size = image_size[0]*image_size[1]
    mask = np.zeros(shape=(num_dendrites, layer_size))
    connectivity = num_dendrites // num_somas
    if type == 'local':
        image =  np.arange(layer_size).reshape((image_size[0], image_size[1]))
        centers = rng.choice(layer_size, num_dendrites, replace=True)
        for i, center in enumerate(centers):
            nb = nb_vals(image, center, rf_size // 8)
            if len(nb) < rf_size:
                counter = 1
                while len(nb) < rf_size:
                    nb = nb_vals(image, center,  (rf_size // 8) + counter)
                    counter += 1
            synapses = rng.choice(nb, rf_size)
            flat_indices = np.ravel_multi_index((synapses[:, 0], synapses[:, 1]), (28, 28))
            mask[i, flat_indices] = 1
        
    elif type == 'global':
        image =  np.arange(layer_size).reshape((image_size[0], image_size[1]))
        center_centers = rng.choice(layer_size, num_somas)
        for i, center in enumerate(center_centers):
            nb = nb_vals(image, center, rf_size // 4)
            if len(nb) < rf_size:
                counter = 1
                while len(nb) < rf_size:
                    nb = nb_vals(image, center,  (rf_size // 8) + counter)
                    counter += 1
            synapses = rng.choice(nb, rf_size)
            flat_indices = np.ravel_multi_index((synapses[:, 0], synapses[:, 1]), (28, 28))
            mask[i, flat_indices] = 1             

        centers = rng.choice(layer_size, num_dendrites, replace=True)
        for i, center in enumerate(centers):
            nb = nb_vals(image, center, rf_size // 8)
            if len(nb) < rf_size:
                counter = 1
                while len(nb) < rf_size:
                    nb = nb_vals(image, center,  (rf_size // 8) + counter)
                    counter += 1
            synapses = rng.choice(nb, rf_size)
            flat_indices = np.ravel_multi_index((synapses[:, 0], synapses[:, 1]), (28, 28))
            mask[i, flat_indices] = 1        
    elif type == 'random':
        for i in range(num_dendrites):
            ids = rng.choice(layer_size, rf_size, replace=False)
            mask[i,ids] = 1

    else:
        raise NameError('type must be either \'local\' or \'global\'')
    return torch.from_numpy(mask)

def somatic_mask(num_dendrites, num_somas):
    connectivity = num_dendrites // num_somas
    mask = np.zeros(shape=(num_somas, num_dendrites))
    for i in range(num_somas):
        mask[i, i * connectivity: i * connectivity + connectivity] = 1
    return torch.from_numpy(mask)




def nb_vals(matrix, index, size):
    coords = np.unravel_index(index, matrix.shape)
    dist = np.ones(matrix.shape)
    dist[coords] = 0
    dist = scn.distance_transform_cdt(dist, metric='chessboard')
    
    nb_indices = np.transpose(np.nonzero(dist <= size))
    return nb_indices
