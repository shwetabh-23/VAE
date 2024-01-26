import yaml
import torchvision
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from src import create_decoder, create_encoder, create_output_layer


class VAE(pl.LightningModule):
    def __init__(self, file):
        super(VAE, self).__init__()

        self.conv_layers = file['architecture']['conv_layers']
        self.conv_kernel_size = file['architecture']['conv_kernel_size']
        self.conv_stride = file['architecture']['conv_stride']
        self.conv_padding = file['architecture']['conv_paddings']
        self.conv_channels = file['architecture']['conv_channels']
        self.conv_z_dim = file['architecture']['z_dimension']

        self.epochs =  int(file['hyperparameters']['epochs'])
        self.batch_size =  int(file['hyperparameters']['batch_size'])
        self.learning_rate =  float(file['hyperparameters']['learning_rate'])
        self.scheduler_step_size = self.epochs // 2

        self.dataset_method = file['configuration']['ds_method']
        if self.dataset_method == 'MNIST':
            self.dataset_method = torchvision.datasets.MNIST
        elif self.dataset_method == 'CIFAR10':
            self.dataset_method = torchvision.datasets.CIFAR10
        self.dataset_shape = file['configuration']['ds_shape']
        self.dataset_path = file['configuration']['ds_path']

        self.encoder, self.encoder_shapes = create_encoder(architecture= file['architecture'], input_shape= self.dataset_shape)

        self.inp_features = self.conv_channels[-1] * np.prod(self.encoder_shapes[-1][:])
        
        self.mean_layer = nn.Linear(in_features= self.inp_features, out_features= self.conv_z_dim)
        self.stddev_layer = nn.Linear(in_features= self.inp_features, out_features= self.conv_z_dim)

        self.decoder = create_decoder(architecture= file['architecture'], encoder_shape= self.encoder_shapes)

        self.decoder_input = nn.Linear(in_features=self.conv_z_dim, out_features= self.inp_features)
        self.output_layer = create_output_layer(architecture= file['architecture'], input_shape= self.dataset_shape)

    def encode(self, X):

        #breakpoint()

        encoded_input = self.encoder(X)
        encoded_input = torch.flatten(encoded_input, start_dim = 1)

        mean = self.mean_layer(encoded_input)
        stddev = self.stddev_layer(encoded_input)

        return mean, stddev
    
    def compute_latent_vector(self, mean, stddev):

        epsilon = torch.rand_like(stddev)
        return mean + epsilon * (1.0/2.0) * stddev

    def decode(self, z):

        decoded_input = self.decoder_input(z)
        height = self.encoder_shapes[-1][0] 
        width = self.encoder_shapes[-1][1]
        decoder_input = decoded_input.view(-1, self.conv_channels[-1], height, width)

        decoded_output = self.decoder(decoder_input)

        network_output = self.output_layer(decoded_output)

        return network_output
    
    def forward(self, X):

        mean, stddev = self.encode(X)

        latent_vector = self.compute_latent_vector(mean=mean, stddev= stddev)

        decoded_output = self.decode(latent_vector)

        return decoded_output, mean, stddev
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(params= self.parameters(), lr= self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer= optimizer, step_size= self.scheduler_step_size, gamma= 0.1)

        return [optimizer], [scheduler]
    
    @staticmethod
    def data_fidility_loss_term(X, X_hat, eps = 1e-10):

        loss = torch.sum((
            X * torch.log(X_hat + eps) + (1 - X) * torch.log(eps + 1 - X_hat)
        ), axis = [1,2,3])

        return loss
    
    @staticmethod
    def kl_divergence_loss(mean, stddev):

        loss = (1/2) * torch.sum(
            torch.exp(stddev) + torch.square(mean) - 1 - stddev, axis = 1 
        )

        return loss
    
    @staticmethod
    def criterion(X, X_hat, mean, stddev):

        data_fidility_loss = VAE.data_fidility_loss_term(X, X_hat)
        kl_divergence_loss = VAE.kl_divergence_loss(mean, stddev)

        loss = -data_fidility_loss + kl_divergence_loss

        losses = {
            'data_fidility' : data_fidility_loss,
            'kl_divergence' : kl_divergence_loss,
            'loss' : torch.mean(loss)
        }

        return losses

    def train_dataloader(self):

        train_set = self.dataset_method(root = self.dataset_path, train = True, download = True, transform = torchvision.transforms.ToTensor())
        train_data_loader = DataLoader(dataset= train_set, batch_size= self.batch_size, shuffle= True)

        return train_data_loader 


    def test_dataloader(self):

        test_set = self.dataset_method(root = self.dataset_path, train = False, download = True, transform = torchvision.transforms.ToTensor())
        self.testdataloader = DataLoader(dataset= test_set, batch_size= self.batch_size, shuffle= True)

        return (self.testdataloader) 
    
    def training_step(self, batch, batch_idx):

        X, y = batch
        X_hat, mean, stddev = self(X)

        losses = VAE.criterion(X=X, X_hat=X_hat, mean=mean, stddev=stddev)
        return losses

    def test_step(self, batch, batch_idx):

        X, y = batch
        X_hat, mean, stddev = self(X)

        losses = VAE.criterion(X=X, X_hat=X_hat, mean=mean, stddev=stddev)

        mse_loss = nn.MSELoss()
        mse_loss_val = mse_loss(X, X_hat)
        return losses, mse_loss_val

    def sample(self, n):

        z = torch.randn(n*n, self.conv_z_dim)
        samples = self.decode(z)
        cmap = None
        if self.dataset_shape == [1, 28, 28]:
            cmap = 'gray'

        elif self.dataset_shape == [3, 28, 28]:
            cmap = 'viridis'

        plot_multiple(samples.detach().numpy(), n, self.dataset_shape, cmap, name = '1')

    def reconstruct(self, n):

        images = 0
        tensors = []
        while n*n > images:
            breakpoint()
            batch, y = next(iter(self.testdataloader))
            images += len(batch)
            tensors.append(batch)

        X = torch.cat(tensors=tensors, dim= 0)

        X_hat, mean, stddev = self(X)
        min_images = min(n, len(batch))

        cmap = None

        if self.dataset_shape == [1, 28, 28]:
            cmap == 'gray'

        elif self.dataset_shape == [3, 28, 28]:
            cmap = 'viridis'

        plot_multiple(images= X_hat.detach().numpy(), n= min_images, dim= self.dataset_shape, cmap= cmap, name= '2')

def plot_multiple(images, n, dim, cmap, name):

    # unpack the image dimensions
    z_dim, x_dim, y_dim = dim

    # if image is grayscale
    if (z_dim == 1):
        # initialize some limits on x&y
        x_limit = np.linspace(-2, 2, n)
        y_limit = np.linspace(-2, 2, n)

        # initialize the final combined image
        empty = np.empty((x_dim*n, y_dim*n))

        current = 0
        for i, zi in enumerate(x_limit):
            for j, pi in enumerate(y_limit):
                # each image insert it into a subsection of the final image
                empty[(n-i-1)*x_dim:(n-i)*x_dim, j*y_dim:(j+1)*y_dim] = images[current][0]
                current+=1

        plt.figure(figsize=(8, 10))

        x,y = np.meshgrid(x_limit, y_limit)
        plt.imshow(empty, origin="upper", cmap=cmap)
        plt.savefig(f'{name}.png')
        plt.grid(False)
        plt.show()

    # if the image is rgb
    elif (z_dim == 3):
        # initialize some limits on x&y
        x_limit = np.linspace(-2, 2, n)
        y_limit = np.linspace(-2, 2, n)

        # initialize the final combined image (now with one more dim)
        empty = np.empty((x_dim*n, y_dim*n, 3))

        current = 0
        for i, zi in enumerate(x_limit):
            for j, pi in enumerate(y_limit):
                # flatten the image
                curr_img = images[current].ravel()
                # reshape it into the correct shape for pyplot
                curr_img = np.reshape(curr_img, (x_dim, y_dim, z_dim), order='F')
                # rotate it by 270 degrees
                curr_img = np.rot90(curr_img, 3)

                # insert it into a subsection of the final image
                empty[(n-i-1)*x_dim:(n-i)*x_dim, j*y_dim:(j+1)*y_dim] = curr_img
                current+=1

        plt.figure(figsize=(8, 10))

        x,y = np.meshgrid(x_limit, y_limit)
        plt.imshow(empty, origin="upper", cmap=cmap)
        plt.savefig(f'{name}.png')
        plt.grid(False)
        plt.show()



if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        file = yaml.safe_load(f)
    vae = VAE(file= file)
    breakpoint()



