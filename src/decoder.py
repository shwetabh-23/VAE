import torch.nn as nn
from .utils import create_output_padding, create_transpose_output_shape
import yaml
from .encoder import create_encoder

def create_decoder(architecture, encoder_shape):

    conv_sets = []
    in_channels = architecture['conv_channels'][-1]
    for layer in range(architecture['conv_layers'] - 1, -1, -1):

        out_channels = architecture['conv_channels'][layer]
        padding = architecture['conv_paddings'][layer]
        kernel = architecture['conv_kernel_size'][layer]
        stride = architecture['conv_stride'][layer]

        curr_shape = encoder_shape[layer + 1]
        target_shape = encoder_shape[layer]

        out_shape = create_transpose_output_shape(curr_shape=curr_shape, kernel_size= kernel, stride= stride, padding= padding)
        output_padding = create_output_padding(out_shape, target_shape)
        conv_sets.append(nn.Sequential(nn.ConvTranspose2d(in_channels= in_channels, 
                                                          out_channels= out_channels, 
                                                          kernel_size= kernel, 
                                                          stride= stride, 
                                                          padding= padding, 
                                                          output_padding= output_padding)))
        
        in_channels = out_channels
        
    return nn.Sequential(*conv_sets)

if __name__ == '__main__':
    with open(r'/home/harsh/AI-Projects/VAE/config.yaml', 'r') as f:
        architecture = yaml.safe_load(f)
    architecture = architecture['architecture']
    in_shape = (3, 28, 28)
    encoder, encoder_shape = create_encoder(architecture=architecture, input_shape=in_shape)
    
    decoder = create_decoder(architecture=architecture, encoder_shape= encoder_shape)
