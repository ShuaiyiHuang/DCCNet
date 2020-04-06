'''
Shuaiyi Huang
Implement Spatial Context Encoder Efficient Version. Not used.
'''

import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch.autograd import Variable
from torch.nn.modules.utils import _quadruple
import numpy as np

def global_spatial_representation_efficient(data,kernel_size):
    '''
    2019-04-27 Applies self local similarity with fixed sliding window. Efficient version.
    Args:
        data: featuren map, variable of shape (b,c,h,w)
        kernel_size: width/heigh of local window, int
    Returns:
        output: global spatial map, variable of shape (b,k^2,h,w)
    '''

    padding = int(kernel_size//2) #5.7//2 = 2.0, 5.2//2 = 2.0
    b, c, h, w = data.shape
    p2d = _quadruple(padding) #(pad_l,pad_r,pad_t,pad_b)
    data_padded = Func.pad(data,p2d,'constant',0) #output variable
    assert data_padded.shape==(b,c,(h+2*padding),(w+2*padding)),'Error: data_padded shape{} wrong!'.format(data_padded.shape)

    output = Variable(torch.zeros(b,kernel_size*kernel_size,h,w),requires_grad = data.requires_grad)
    if data.is_cuda:
        output = output.cuda(data.get_device())

    xs,xe = padding,w+padding
    ys,ye = padding,h+padding
    patch_center = data_padded[:,:,ys:ye,xs:xe]

    i = 0
    for dy in np.arange(-padding,padding+1):
        for dx in np.arange(-padding,padding+1):
            hs = ys+dy
            he = ye+dy
            ws = xs+dx
            we = xe+dx

            patch_neighbor = data_padded[:,:,hs:he,ws:we] #(b,c,h,w)
            correlation_tensor = torch.sum(patch_neighbor*patch_center,dim=1)
            output[:, i, :, :] = correlation_tensor
            i+=1

    return output

def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature,norm)


class SpatialContextEncoderEfficient(nn.Module):
    def __init__(self,kernel_size,input_dim,hidden_dim):
        super(SpatialContextEncoderEfficient, self).__init__()
        self.embeddingFea = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

        self.kernel_size = kernel_size
        print('verbose...SpatialContextEncoderEfficientBlock with input_dim {},hidden_dim {}'.format(input_dim,hidden_dim))

    def forward(self,x):
        kernel_size = self.kernel_size
        feature_gs = global_spatial_representation_efficient(x,kernel_size=kernel_size)

        #Add L2norm
        feature_gs = featureL2Norm(feature_gs)

        #concatenate
        feature_cat = torch.cat([x, feature_gs], 1)

        # embed
        feature_embd = self.embeddingFea(feature_cat)

        return feature_embd

if __name__ == '__main__':
    print()
    import time

    b,c,h,w = 1,1024,25,25
    data_a = Variable(torch.rand(b,c,h,w))
    data_b = Variable(torch.rand(b,c,h,w))
    obj = SpatialContextEncoderEfficient(kernel_size=25, input_dim=1024+25*25, hidden_dim=1024)

    st = time.time()
    out = obj.forward(data_a)
    et= time.time()
    print('verbose log..', data_a.mean(), data_b.mean(), out.mean(), out.shape, 'time', et - st)