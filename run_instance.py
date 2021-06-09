import pickle, os
from parallel_gaec_py import parallel_gaec_eigen
# from lpmp_py.raw_solvers import amwc_solver # For comparison with LPMP GAEC.
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

HEIGHT = 256
WIDTH = 512

def get_mask_image(cluster_labels_1d):
    assert len(cluster_labels_1d) == HEIGHT * WIDTH
    cluster_labels_2d = np.array(cluster_labels_1d).reshape(HEIGHT, WIDTH)

    mask_image = np.zeros(cluster_labels_2d.shape, dtype=np.float32)
    values = np.unique(cluster_labels_2d)
    for (i, v) in enumerate(values):
        mask_image[cluster_labels_2d == v] = i
    print(f"Found: {i + 1} clusters.")
    mask_image = mask_image / i
    mask_image = plt.cm.tab20(mask_image)
    mask_image = (mask_image * 255).astype(np.uint8)
    return Image.fromarray(mask_image)

instance_dir = '/home/ahabbas/data/multicut/cityscapes_small_val_instances/' # Contains input image, MC instance, and GAEC result from LPMP
results_folder = './'
start_id = 1
end_id = 2 # Can be upto 52

for id in range(start_id, end_id):
    file_prefix_path = os.path.join(instance_dir, str(id))
    instance = pickle.load(open(file_prefix_path + '.pkl', 'rb'))
    edge_indices = instance['edge_indices']
    edge_costs = instance['edge_costs']
    cluster_labels_1d = parallel_gaec_eigen(edge_indices, edge_costs[:, np.newaxis])

    ''' For running AMWC from LPMP: (https://github.com/aabbas90/LPMP/tree/268132a4b95308ae1e8a1afeadeac1da2e63e8f9) 
    num_nodes = instance['edge_indices'].max() + 1
    node_costs = np.zeros((num_nodes, 1)) # Create fake node costs.
    _, cluster_labels_1d, edge_labels_1d, solver_cost = amwc_solver(node_costs, edge_indices, edge_costs[:, np.newaxis], partitionable = np.array([True] * 1, dtype='bool'))
    print(f"edge_labels mean: {edge_labels_1d.mean()}, solver_cost: {solver_cost}") '''
    
    cluster_labels_2d_image = get_mask_image(cluster_labels_1d)
    cluster_labels_2d_image.save(os.path.join(results_folder, str(id) + '_parallel_gaec_result.png'))
    print(f"LPMP GAEC cost: {instance['GAEC_cost']:.3f}")