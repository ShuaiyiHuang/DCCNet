import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_image(im,batch_idx=0,return_im=False):
    if im.dim()==4:
        im=im[batch_idx,:,:,:]
    mean=Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3,1,1))
    std=Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3,1,1))
    if im.is_cuda:
        mean=mean.cuda()
        std=std.cuda()
    im=im.mul(std).add(mean)*255.0
    im=im.permute(1,2,0).data.cpu().numpy().astype(np.uint8)
    if return_im:
        return im
    plt.imshow(im)
    plt.show()

#2019-01-28 visualize
def plot_image_debug(im,batch_idx=0,return_im=False):
    if im.dim()==4:
        im=im[batch_idx,:,:,:]
    mean=Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3,1,1))
    std=Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3,1,1))
    if im.is_cuda:
        mean=mean.cuda()
        std=std.cuda()
    im=im.mul(std).add(mean)*255.0
    im=im.permute(1,2,0).data.cpu().numpy().astype(np.uint8)
    #duplicate pair
    im = np.concatenate(np.stack([im,im],0))

    #verbose
    verbose = False
    if verbose:
        print('debug imshape in plot.py',im.shape) #(800,800,3)
    if return_im:
        return im
    plt.imshow(im)
    plt.show()

def save_plot(filename):
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(filename, bbox_inches = 'tight',
        pad_inches = 0)

#2019-02-21 shuaiyi plot loss
def plot_loss(train_loss, val_loss, pth, figname):

    N = len(train_loss)
    assert (len(val_loss) == N)
    x = np.arange(1,N+1)

    plt.figure()

    plt.plot(x,train_loss,label='Train loss')
    plt.plot(x, val_loss, label='Val loss')
    plt.title('Loss')
    plt.legend()

    plt.savefig(os.path.join(pth,figname))

    return
