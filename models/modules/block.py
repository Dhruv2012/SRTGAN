from collections import OrderedDict
import torch
import torch.nn as nn
from scipy.signal import gaussian
import numpy as np
import functools
import kornia

####################
# Basic blocks


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'sigm':
        layer = nn.Sigmoid()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


####################
# Useful blocks
####################

class VGG_Block(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3,norm_type='batch', act_type='leakyrelu'):
        super(VGG_Block, self).__init__()

        self.conv0 = conv_block(in_nc, out_nc, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.conv1 = conv_block(out_nc, out_nc, kernel_size=kernel_size, stride=2, norm_type=None,act_type=act_type)

    def forward(self, x):
        x1 = self.conv0(x)
        out = self.conv1(x1)
        
        return out


class VGGGAPQualifier(nn.Module):
    def __init__(self, in_nc=3, base_nf=32, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(VGGGAPQualifier, self).__init__()
        # 1024,768,3

        B11 = VGG_Block(in_nc,base_nf,norm_type=norm_type,act_type=act_type)
        # 512,384,32
        B12 = VGG_Block(base_nf,base_nf,norm_type=norm_type,act_type=act_type)
        # 256,192,32
        B13 = VGG_Block(base_nf,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 128,96,64
        B14 = VGG_Block(base_nf*2,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 64,48,64

        # 1024,768,3
        B21 = VGG_Block(in_nc,base_nf,norm_type=norm_type,act_type=act_type)
        # 512,384,32
        B22 = VGG_Block(base_nf,base_nf,norm_type=norm_type,act_type=act_type)
        # 256,192,32
        B23 = VGG_Block(base_nf,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 128,96,64
        B24 = VGG_Block(base_nf*2,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 64,48,64


        B3 = VGG_Block(base_nf*2,base_nf*4,norm_type=norm_type,act_type=act_type)
        # 32,24,128
        B4 = VGG_Block(base_nf*4,base_nf*8,norm_type=norm_type,act_type=act_type)
        # 16,12,256
        B5 = VGG_Block(base_nf*8,base_nf*16,norm_type=norm_type,act_type=act_type)
        
        self.feature1 = sequential(B11,B12,B13,B14)
        self.feature2 = sequential(B21,B22,B23,B24)

        self.combine = sequential(B3,B4,B5)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # classifie
        self.classifier = nn.Sequential(
            nn.Linear(base_nf*16, 512), nn.LeakyReLU(0.2, True), nn.Dropout(0.25), nn.Linear(512,256),nn.LeakyReLU(0.2, True), nn.Dropout(0.5), nn.Linear(256, 1), nn.LeakyReLU(0.2, True))

    def forward(self, x):

        f1 = self.feature1(x)
        f2 = self.feature2(x)
        x = self.gap(self.combine(f1-f2))

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGGGAPQualifierwave(nn.Module):
    def __init__(self, in_nc=9, base_nf=32, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(VGGGAPQualifierwave, self).__init__()
        # 512,384,9
        B11 = VGG_Block(in_nc,base_nf,norm_type=norm_type,act_type=act_type)
        # 256, 192, 32
        B12 = VGG_Block(base_nf,base_nf,norm_type=norm_type,act_type=act_type)
        # 128,96,32
        B13 = VGG_Block(base_nf,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 64,48,64
        B14 = VGG_Block(base_nf*2,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 32,24,64

        # 512,384,9
        B21 = VGG_Block(in_nc,base_nf,norm_type=norm_type,act_type=act_type)
        # 256, 192, 32
        B22 = VGG_Block(base_nf,base_nf,norm_type=norm_type,act_type=act_type)
        # 128,96,32
        B23 = VGG_Block(base_nf,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 64,48,64
        B24 = VGG_Block(base_nf*2,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 32,24,64

        B3 = VGG_Block(base_nf*2,base_nf*4,norm_type=norm_type,act_type=act_type)
        # 16,12,128
        B4 = VGG_Block(base_nf*4,base_nf*8,norm_type=norm_type,act_type=act_type)
        # 8,6,1024
        # B5 = VGG_Block(base_nf*8,base_nf*16,norm_type=norm_type,act_type=act_type)
        
        self.feature1 = sequential(B11,B12,B13,B14)
        self.feature2 = sequential(B21,B22,B23,B24)
        self.combine = sequential(B3,B4)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(base_nf*8, 512), nn.LeakyReLU(0.2, True), nn.Dropout(0.25), nn.Linear(512,256),nn.LeakyReLU(0.2, True), nn.Dropout(0.5), nn.Linear(256, 1), nn.LeakyReLU(0.2, True))
        
        # 256,192,9
        B111 = VGG_Block(in_nc,base_nf,norm_type=norm_type,act_type=act_type)
        # 128, 96, 32
        B112 = VGG_Block(base_nf,base_nf,norm_type=norm_type,act_type=act_type)
        # 64,48,32
        B113 = VGG_Block(base_nf,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 32,24,64
        # B14 = VGG_Block(base_nf*2,base_nf*2,norm_type=norm_type,act_type=act_type)
        # # 32,24,64

        # 256,192,9
        B121 = VGG_Block(in_nc,base_nf,norm_type=norm_type,act_type=act_type)
        # 128, 96, 32
        B122 = VGG_Block(base_nf,base_nf,norm_type=norm_type,act_type=act_type)
        # 64,48,32
        B123 = VGG_Block(base_nf,base_nf*2,norm_type=norm_type,act_type=act_type)
        # 32,24,64
        # B14 = VGG_Block(base_nf*2,base_nf*2,norm_type=norm_type,act_type=act_type)
        # # 32,24,64


        B13 = VGG_Block(base_nf*2,base_nf*4,norm_type=norm_type,act_type=act_type)
        # 16,12,128
        B14 = VGG_Block(base_nf*4,base_nf*16,norm_type=norm_type,act_type=act_type)
        # 8,6,512
        # B5 = VGG_Block(base_nf*8,base_nf*16,norm_type=norm_type,act_type=act_type)
        
        self.feature11 = sequential(B111,B112,B113)
        self.feature12 = sequential(B121,B122,B123)
        self.combine1 = sequential(B13,B14)
        self.gap1 = nn.AdaptiveAvgPool2d((1,1))
        
        # classifie
        self.classifier1 = nn.Sequential(
            nn.Linear(base_nf*8, 512), nn.LeakyReLU(0.2, True), nn.Dropout(0.25), nn.Linear(512,256),nn.LeakyReLU(0.2, True), nn.Dropout(0.5), nn.Linear(256, 1), nn.LeakyReLU(0.2, True))

    def forward(self,x,x1):
        self.hfwt  = 1
        f1 = self.feature1(x)
        f2 = self.feature2(x)
        x0 = self.gap(self.combine(f1-f2))
        x0 = x0.view(x0.size(0), -1)
        x0 = self.classifier(x0)

        f11 = self.feature11(x1)
        f12 = self.feature12(x1)
        x1 = self.gap1(self.combine(f11-f12))

        x1 = x1.view(x1.size(0), -1)
        x1 = self.classifier1(x1)
        return self.hfwt*x0+x1


class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, out_nc, mid_nc= 6, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        conv2 = conv_block(in_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res_path1 = sequential(conv0, conv1)
        self.res_path2 = sequential(conv2)
        self.res_scale = res_scale

    def forward(self, x):
        path1 = self.res_path1(x).mul(self.res_scale)
        path2 = self.res_path2(x)
        return path1 + path2

class ResNetBlockcanny(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, out_nc, mid_nc=6, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlockcanny, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        
        conv2 = conv_block(1, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x

        canny_block = CannyFilter()
        self.res_path1 = sequential(conv0, conv1)
        self.res_path2 = sequential(canny_block, conv2)
        self.res_scale = res_scale

    def forward(self, x):
        path1 = self.res_path1(x).mul(self.res_scale)
        path2 = self.res_path2(x)
        return path1 + path2 

class CannyFilter(nn.Module):
    def __init__(self, threshold=10.0, use_cuda=True):
        super(CannyFilter, self).__init__()

        self.threshold = threshold
        self.use_cuda = use_cuda

        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

    def forward(self, imgn):
        batch_size = imgn.shape[0]
        #print(batch_size)
        img = imgn.clone()
        img_r = img[:,0:1]
        img_g = img[:,1:2]
        img_b = img[:,2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES

        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)+ torch.sqrt(grad_x_g**2 + grad_y_g**2)+ torch.sqrt(grad_x_b**2 + grad_y_b**2)
        #grad_mag = grad_mag + torch.sqrt(grad_x_g**2 + grad_y_g**2)
        #grad_mag = grad_mag + torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
        grad_orientation += 180.0
        grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)

        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width
        pixel_range = torch.FloatTensor([range(pixel_count)])
        #print(pixel_range.shape)
        
        inidices_positive = torch.squeeze(inidices_positive)
        #print(inidices_positive.data.shape)
        #print(inidices_negative.view(1,-1).shape)
        #print(pixel_range.repeat(1,batch_size).shape)

        if self.use_cuda:
            pixel_range = torch.cuda.FloatTensor([range(pixel_count)])


        indices = (inidices_positive.view(1,-1).data * pixel_count + pixel_range.repeat(1, batch_size)).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(batch_size,1,height,width)

        indices = (inidices_negative.view(1,-1).data * pixel_count + pixel_range.repeat(1, batch_size)).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(batch_size, 1, height, width)
        channel_select_filtered = torch.cat([channel_select_filtered_positive,channel_select_filtered_negative], 1)
        is_max = channel_select_filtered.min(dim=1)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=1)

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # THRESHOLD

        thresholded = thin_edges.clone()
        thresholded[thin_edges<self.threshold] = 0.0

        early_threshold = grad_mag.clone()
        early_threshold[grad_mag<self.threshold] = 0.0

        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()
        #print(thresholded.shape)
        return thresholded

    
class ResidualDenseBlock_5C(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(gc, nc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=None, mode=mode)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.conv6 = conv_block(nc, 16, kernel_size=1, norm_type=None, act_type='prelu')
        self.conv7 = conv_block(16, nc, kernel_size=1, norm_type=None, act_type='sigm')

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x6 = self.conv7(self.conv6(self.gap(x4)))
        return x4.mul(x6) + x


class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.conv1 = conv_block(nc, nc, 1, stride, norm_type=None, act_type=None)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        out = self.conv1(x) + out
        return out


####################
# Upsampler
####################


def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=None, act_type='relu'):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)

def downconv_blcok(in_nc, out_nc, downscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    f = 0.5
    upsample = nn.Upsample(scale_factor=f)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)

class EdgeScoreNet(nn.Module):
    """
    """
    def __init__(self,input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a EdgeScoreNet
            Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(EdgeScoreNet, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
      
        self.grad_RGB = kornia.filters.Sobel(normalized = True)
        
        # 70*70 PatchGAN Discriminator
        kw = 70
        padw = 23
        self.edge_score_conv = nn.Conv2d(1, 1, kernel_size=kw, stride=8, padding=padw)
        self.edge_score_conv.weight.data.fill_(1)
        self.edge_score_conv.bias.data.fill_(0)
        
        self.edge_score_act = nn.LeakyReLU(0.2, True)
        self.edge_score = sequential(self.edge_score_conv, self.edge_score_act)

    
    def forward(self, input):
        """Standard forward."""
        grad_RGB = self.grad_RGB(input.cuda())
        
        #Avg Gradient across all channels
        grad = torch.rand((input.shape[0], 1, input.shape[2], input.shape[3]))
        grad[:,0,:,:] = (grad_RGB[:,0,:,:] + grad_RGB[:,1,:,:] + grad_RGB[:,2,:,:])/3
        
        #Normalize and convert it to 0 and 1 range
        min = torch.min(grad)
        grad = torch.add(grad, abs(min))
        grad = torch.div(grad, torch.max(grad))

        grad = self.edge_score(grad.cuda())
        min = torch.min(grad)
        grad = torch.add(grad, abs(min))
        grad = torch.div(grad, torch.max(grad))
        #Edge Score
        return grad