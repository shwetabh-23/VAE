import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from vae import VAE
import yaml
import torch
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

with open('config.yaml', 'r') as f:
        file = yaml.safe_load(f)

os.makedirs('data', exist_ok= True)
model = VAE(file= file)

bottleneck_layer_size = '64'
model_path = f'/home/harsh/AI-Projects/VAE/models/cifar/size_variation/{bottleneck_layer_size}/model.pth'
#breakpoint()
logger = TensorBoardLogger(model_path[:-9] + 'logs/', name= 'modelv1')
pca = PCA(n_components= 2)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict=state_dict)

def get_weights_visualization():
    mean_layer = model.mean_layer.weight
    stddev_layer = model.stddev_layer.weight
    epsilon = torch.rand_like(stddev_layer)
    bottleneck_layer = mean_layer + epsilon * (1.0/2.0) * stddev_layer

    bottleneck_layer = bottleneck_layer.detach().numpy()

    pca = PCA(n_components= 2)
    pca_results = pca.fit_transform(bottleneck_layer)

    plt.scatter(pca_results[:, 0], pca_results[:, 1])
    plt.title('PCA visualization of bottleneck layer')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.show()

def get_class_visualization():
    all_means = []
    for i in range(0, 10):

        latent_vector = model.get_latent_vectors(25, i)
        latent_vector = latent_vector.detach().numpy()
        if latent_vector.shape[1] > 2:
            pca = PCA(n_components= 2)

            pca_results = pca.fit_transform(latent_vector)

            x_mean = pca_results[:, 0].mean()
            y_mean = pca_results[:, 1].mean()
            all_means.append((x_mean, y_mean))
        else:
            x_mean = latent_vector[:, 0].mean()
            y_mean = latent_vector[:, 1].mean()
            all_means.append((x_mean, y_mean))

    #x_values, y_values = zip(*all_means)
    #breakpoint()

# Plot the data
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    mean_vectors = {
        0: np.array(all_means[0]),
        1: np.array(all_means[1]),
        2: np.array(all_means[2]),
        3: np.array(all_means[3]),
        4: np.array(all_means[4]),
        5: np.array(all_means[5]),
        6: np.array(all_means[6]),
        7: np.array(all_means[7]),
        8: np.array(all_means[8]),
        9: np.array(all_means[9]),
    }

    # Mapping markers and colors to each digit
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', '+', 'x']
    colors = plt.cm.viridis(np.linspace(0, 1, len(digits)))

    # Create a scatter plot
    fig, ax = plt.subplots()
    for i, digit in enumerate(digits):
        x, y = mean_vectors[digit]
        marker = markers[i % len(markers)]  # Cycle through markers
        color = colors[i]
        label = f'Class {digit}'
        ax.scatter(x, y, marker=marker, color=color, label=label, s=100)

    # Display information on the graph
    for digit, (x, y) in mean_vectors.items():
        plt.text(x, y, f'Class {digit}', fontsize=8, ha='right', va='bottom')

    # Set labels and title
    ax.set_xlabel('Mean X')
    ax.set_ylabel('Mean Y')
    ax.set_title('Mean Latent Space Representations for Each Class')

    # Display legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

    # Show the plot
    plt.savefig('your_figure2.jpg', format='jpg', bbox_inches='tight')

    plt.show()
if __name__ == '__main__':
     get_class_visualization()