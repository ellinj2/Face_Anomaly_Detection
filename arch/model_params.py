import torch
import torch.nn as nn

def calc_out_dim(in_dim, padding, dilation, kernel_size, stride) -> int:
    return (in_dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

in_dim = (3,400,400)
out_dim = 200

fc_layer2_neurons = 800

conv1_filters = 96
conv2_filters = 256
conv3_filters = 384

conv1_kernel_size = (7,7)
conv2_kernel_size = (5,5)
conv3_kernel_size = (3,3)

maxpool1_kernel_size = (3,3)
maxpool2_kernel_size = (3,3)
maxpool3_kernel_size = (3,3)

conv1_stride = 4
conv2_stride = 1
conv3_stride = 1

maxpool1_stride = 2
maxpool2_stride = 2
maxpool3_stride = 2

conv1_padding = 2
conv2_padding = 0
conv3_padding = 0

conv1_dim_h = calc_out_dim(in_dim[1], conv1_padding, 1, conv1_kernel_size[0], conv1_stride)
conv1_dim_w = calc_out_dim(in_dim[2], conv1_padding, 1, conv1_kernel_size[1], conv1_stride)

maxpool1_dim_h = calc_out_dim(conv1_dim_h, 0, 1, maxpool1_kernel_size[0], maxpool1_stride)
maxpool1_dim_w = calc_out_dim(conv1_dim_w, 0, 1, maxpool1_kernel_size[1], maxpool1_stride)

conv2_dim_h = calc_out_dim(maxpool1_dim_h, conv2_padding, 1, conv2_kernel_size[0], conv2_stride)
conv2_dim_w = calc_out_dim(maxpool1_dim_w, conv2_padding, 1, conv2_kernel_size[1], conv2_stride)

maxpool2_dim_h = calc_out_dim(conv2_dim_h, 0, 1, maxpool2_kernel_size[0], maxpool2_stride)
maxpool2_dim_w = calc_out_dim(conv2_dim_w, 0, 1, maxpool2_kernel_size[1], maxpool2_stride)

conv3_dim_h = calc_out_dim(maxpool2_dim_h, conv3_padding, 1, conv3_kernel_size[0], conv3_stride)
conv3_dim_w = calc_out_dim(maxpool2_dim_w, conv3_padding, 1, conv3_kernel_size[1], conv3_stride)


maxpool3_dim_h = calc_out_dim(conv3_dim_h, 0, 1, maxpool3_kernel_size[0], maxpool3_stride)
maxpool3_dim_w = calc_out_dim(conv3_dim_w, 0, 1, maxpool3_kernel_size[1], maxpool3_stride)

# CONV LAYERS
conv1 = nn.Conv2d(3, conv1_filters, conv1_kernel_size, stride=conv1_stride, padding=conv1_padding)
conv2 = nn.Conv2d(conv1_filters, conv2_filters, conv2_kernel_size, stride=conv2_stride, padding=conv2_padding)
conv3 = nn.Conv2d(conv2_filters, conv3_filters, conv3_kernel_size, stride=conv3_stride, padding=conv3_padding)

# MAX POOLS
maxpool1 = nn.MaxPool2d(maxpool1_kernel_size, maxpool1_stride)
maxpool2 = nn.MaxPool2d(maxpool2_kernel_size, maxpool2_stride)
maxpool3 = nn.MaxPool2d(maxpool3_kernel_size, maxpool3_stride)

# DROPOUTS
dropout1 = nn.Dropout2d(0.5) # AFTER CONVS
dropout2 = nn.Dropout(0.5) # BETWEEN DENSE

# FIRST DENSE LAYER SIZE
fc_inputs = int(conv3_filters * maxpool3_dim_w * maxpool3_dim_h)

# DENSE LAYERS
lin1 = nn.Linear(fc_inputs, fc_layer2_neurons)
lin2 = nn.Linear(fc_layer2_neurons, out_dim)

# DECODER ARCHITECTURE
decode_lin1 = nn.Linear(out_dim, fc_layer2_neurons)
decode_lin2 = nn.Linear(fc_layer2_neurons, fc_inputs)

decode_unflatten = nn.Unflatten(dim=-1, unflattened_size=(conv3_filters, maxpool3_dim_h, maxpool3_dim_w))

decode_conv1 = nn.ConvTranspose2d(conv3_filters, conv2_filters, conv3_kernel_size[0]-conv3_padding, stride=2)
decode_conv2 = nn.ConvTranspose2d(conv2_filters, conv1_filters, conv2_kernel_size[0]-conv2_padding, stride=2)
decode_conv3 = nn.ConvTranspose2d(conv1_filters, 3, conv1_kernel_size[0]-conv1_padding, stride=2)

if __name__ == "__main__":
    print("CONV_LAYERS: 1    2   3")
    print("CONV_Hs   : {}".format([conv1_dim_h, conv2_dim_h, conv3_dim_h]))
    print("CONV_Ws   : {}".format([conv1_dim_w, conv2_dim_w, conv3_dim_w]))
    print("MAXPOOL_Hs: {}".format([maxpool1_dim_h, maxpool2_dim_h, maxpool3_dim_h]))
    print("MAXPOOL_Ws: {}".format([maxpool1_dim_w, maxpool2_dim_w, maxpool3_dim_w]))
    print("DENSE_LAYERS:  1      2")
    print("DENSE       : {}".format([fc_inputs,fc_layer2_neurons]))
    print("OUTPUT: {}".format(out_dim))
