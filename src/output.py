import torch.nn as nn
import yaml
from .encoder import create_encoder
from .decoder import create_decoder


def create_output_layer(architecture, input_shape):

    in_channels = int(architecture['conv_channels'][0])

    kernel_size = architecture['conv_kernel_size'][0]
    stride = architecture['conv_stride'][0]
    padding = architecture['conv_paddings'][0]


    return nn.Sequential(nn.ConvTranspose2d(in_channels= in_channels, 
                            out_channels= in_channels, 
                            kernel_size= kernel_size, 
                            stride= stride, 
                            padding= padding), 
                            nn.SELU(), 
                            nn.BatchNorm2d(num_features= in_channels), 
                            nn.Conv2d(in_channels= in_channels, 
                                      out_channels= input_shape[0], 
                                      kernel_size= kernel_size, 
                                      stride= stride, 
                                      padding= padding), nn.Sigmoid())

if __name__ == '__main__':
    with open(r'/home/harsh/AI-Projects/VAE/config.yaml', 'r') as f:
        architecture = yaml.safe_load(f)
    architecture = architecture['architecture']
    in_shape = (3, 28, 28)
    encoder, encoder_shape = create_encoder(architecture=architecture, input_shape=in_shape)
    
    decoder = create_decoder(architecture=architecture, encoder_shape= encoder_shape)
    final_layer = output_layer(architecture=architecture, input_shape= in_shape)
