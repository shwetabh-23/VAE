

def create_transpose_output_shape(curr_shape, kernel_size, stride, padding):

    dimensions = len(curr_shape)
    return tuple(((curr_shape[i] - 1) * stride[i] - 2 * padding[i] + (kernel_size[i] - 1) + 1) for i in range(dimensions)) 

def create_output_padding(curr_shape, target_shape):
    dimensions = len(curr_shape)
    return tuple((curr_shape[i] - target_shape[i]) for i in range(dimensions))

def compute_output_shape(current_shape,  stride, padding, kernel_size):
    
    dimensions = len(current_shape)
    # for i in range(dimensions):
         
    #     print(current_shape[i])
    #     print('\n')
    #     print(kernel_size[i])
    #     print('\n')
    #     print(padding[i])
    #     print('\n')
    #     print(stride[i])
    #     print('\n')
    #     shape = ((current_shape[i] - kernel_size[i] + 2 * padding[i]) // stride[i] + 1)
    #     print(shape)

    #dimensions = len(current_shape)
    return tuple((current_shape[i] - kernel_size[i] + 2 * padding[i]) // stride[i] + 1
                 for i in range(dimensions))

         
