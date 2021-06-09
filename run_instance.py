import pickle, os
from parallel_gaec_py import parallel_gaec_eigen
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
    print(f"Found: {v} clusters.")
    mask_image = mask_image / v
    mask_image = plt.cm.tab20(mask_image)
    mask_image = (mask_image * 255).astype(np.uint8)
    return Image.fromarray(mask_image)

instance_dir = '/home/ahabbas/data/multicut/cityscapes_small_val_instances/'
results_folder = 'out/'
start_id = 1
end_id = 2 # Can be upto 52

for id in range(start_id, end_id):
    file_prefix_path = os.path.join(instance_dir, str(id))
    instance = pickle.load(open(file_prefix_path + '.pkl', 'rb'))

    cluster_labels_1d = parallel_gaec_eigen(instance['edge_indices'], instance['edge_costs'])
    cluster_labels_2d_image = get_mask_image(cluster_labels_1d)
    cluster_labels_2d_image.save(os.path.join(results_folder, str(id) + '.png'))
