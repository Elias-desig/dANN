import numpy as np
import math
import scipy.ndimage as scn
import torch

def rf_mask(image_size, num_dendrites, num_somas, type='local', rf_size=16):
    seed = 1
    rng = np.random.default_rng()
    layer_size = image_size[0]*image_size[1]
    mask = np.zeros(shape=(num_dendrites, layer_size))
    connectivity = math.ceil(num_dendrites / num_somas)
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
            flat_indices = np.ravel_multi_index((synapses[:, 0], synapses[:, 1]), (image_size[0], image_size[1]))
            mask[i, flat_indices] = 1
        
    elif type == 'global':
        image = np.arange(layer_size).reshape(image_size)
        # one distinct region‐center per soma
        centers = rng.choice(layer_size, num_somas, replace=True)
        somatic_nbs = []
        for i in range(num_somas):
            somatic_nbs.append(nb_vals(image, centers[i], 4))
            counter = 1
            while len(somatic_nbs[i]) < 81:
                somatic_nbs[i] = nb_vals(image, centers[i],  4 + counter)
                counter += 1            
        for i in range(num_dendrites):
            soma_id = min(i // connectivity, num_somas - 1)
            glob_nb = somatic_nbs[soma_id]
            idx = rng.integers(len(glob_nb))
            centre_coord = glob_nb[idx]
            centre_flat  = np.ravel_multi_index(tuple(centre_coord), image_size) 
            nb = nb_vals(image, centre_flat, rf_size // 8)
    
            counter = 1
            while len(nb) < rf_size:
                nb = nb_vals(image, centre_flat,  (rf_size // 8) + counter)
                counter += 1            
            synapses = rng.choice(nb, rf_size)
            flat_indices = np.ravel_multi_index((synapses[:, 0], synapses[:, 1]), (image_size[0], image_size[1]))
            mask[i, flat_indices] = 1
        
                                      
    elif type == 'random':
        for i in range(num_dendrites):
            ids = rng.choice(layer_size, rf_size, replace=False)
            mask[i,ids] = 1

    else:
        raise NameError('type must be either \'local\' or \'global\'')
    return torch.from_numpy(mask)

def somatic_mask(num_dendrites, num_somas):
    connectivity = math.ceil(num_dendrites / num_somas)
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
