import torch 
import torch.nn as nn
import yaml
from .utils import compute_output_shape

def create_encoder(architecture, input_shape):
    conv_sets = []
    in_channels = input_shape[0]
    curr_shape = (input_shape[1], input_shape[2])
    shape_per_layer = [curr_shape]

    for layer in range(architecture['conv_layers']):

        conv_sets.append(nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                out_channels= architecture['conv_channels'][layer],
                                                kernel_size= architecture['conv_kernel_size'][layer],
                                                stride= architecture['conv_stride'][layer],
                                                padding= architecture['conv_paddings'][layer] ), nn.LeakyReLU(negative_slope=0.15), 
                                                nn.BatchNorm2d(num_features= architecture['conv_channels'][layer])))
        

        curr_shape = (compute_output_shape(current_shape= curr_shape, stride= architecture['conv_stride'][layer],
                            padding= architecture['conv_paddings'][layer],
                            kernel_size= architecture['conv_kernel_size'][layer]))
        shape_per_layer.append(curr_shape)
        in_channels = architecture['conv_channels'][layer]

    return nn.Sequential(*conv_sets), shape_per_layer

if __name__ == '__main__':
    with open(r'/home/harsh/AI-Projects/VAE/config.yaml', 'r') as f:
        architecture = yaml.safe_load(f)
    architecture = architecture['architecture']
    in_shape = (3, 28, 28)
    conv_net, shape_per_layer = create_encoder(architecture=architecture, input_shape=in_shape)
