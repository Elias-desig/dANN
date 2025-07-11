import numpy as np
import scipy.ndimage as scn
import torch

def rf_mask(image_size, num_dendrites, type='local', rf_size=16):
    seed = 0
    rng = np.random.default_rng(seed)
    layer_size = image_size[0]*image_size[1]
    mask = np.zeros(shape=(num_dendrites, layer_size))
    if type == 'local':
        indices = np.arange(layer_size * num_dendrites).reshape((num_dendrites, layer_size))
        centers = rng.choice(layer_size * num_dendrites, num_dendrites)
        for i, center in enumerate(centers):
            nb = nb_vals(indices, center, rf_size // 4)
            if len(nb) < rf_size:
                counter = 1
                while len(nb) < rf_size:
                    nb = nb_vals(indices, center,  (rf_size // 4) + counter)
                    counter += 1
            synapses = rng.choice(nb, rf_size)
            mask[i, synapses] = 1
        
    elif type == 'global':
        pass
    elif type == 'random':
        for i in range(num_dendrites):
            ids = rng.choice(layer_size, rf_size, replace=False)
            mask[i,ids] = 1

    else:
        raise NameError('type must be either \'local\' or \'global\'')
    return torch.from_numpy(mask)

def somatic_mask(num_dendritres, num_somas):
    connectivity = num_dendritres // num_somas
    mask = np.zeros(shape=(num_somas, num_dendritres))
    for i in range(num_somas):
        mask[i, i * connectivity: i * connectivity + connectivity] = 1
    return torch.from_numpy(mask)




def nb_vals(matrix, indices, size):
    indices_ = tuple(np.transpose(np.atleast_2d(indices)))
    arr_shape = matrix.shape
    dist = np.ones(arr_shape)
    dist[indices_] = 0
    dist = scn.distance_transform_cdt(dist, metric='chessboard')
    
    nb_indices = np.transpose(np.nonzero(dist <= size))
    return nb_indices
