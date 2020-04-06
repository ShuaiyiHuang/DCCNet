from __future__ import print_function, division
import os
from os.path import exists, join, basename, dirname
from os import makedirs
import numpy as np
import datetime
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from lib.dataloader import DataLoader
from lib.im_pair_dataset import ImagePairDataset
from lib.normalization import NormalizeImageDict
from lib.torch_util import save_checkpoint
from lib.torch_util import BatchTensorToVars
from lib.eval_util_dynamic import pfdataset_pck, pfpascal_val_dataloader

# import DCCNet
from models.model_dynamic import DCCNet
from models.loss_dynamic import weak_loss


# Seed and CUDA
use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
np.random.seed(1)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('DCCNet training script')

# Argument parsing
parser = argparse.ArgumentParser(description='Compute PF Pascal matches')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--image_size', type=int, default=400)
parser.add_argument('--dataset_image_path', type=str, default='datasets/pf-pascal/', help='path to PF Pascal dataset')
parser.add_argument('--dataset_csv_path', type=str, default='datasets/pf-pascal/image_pairs/', help='path to PF Pascal training csv')
parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--result_model_fn', type=str, default='checkpoint_adam', help='trained model filename')
parser.add_argument('--result-model-dir', type=str, default='../model/checkpoints', help='path to trained models folder')
parser.add_argument('--fe_finetune_params',  type=int, default=0, help='number of layers to finetune')
parser.add_argument('--exp_name', type=str, default='exp_delete', help='experiment name')

# DCCNet args
parser.add_argument('--ncons_kernel_sizes', nargs='+', type=int, default=[5,5,5], help='kernels sizes in neigh. cons.')
parser.add_argument('--ncons_channels', nargs='+', type=int, default=[16,16,1], help='channels in neigh. cons')

parser.add_argument('--sce_kernel_size',type=int,default=25,help='kernel size in sce.')
parser.add_argument('--sce_hidden_dim',type=int,default=1024,help='hidden dim in sce')
parser.add_argument('--scaleloss_weight',type=float,default=1.0,help='whether use scale loss, if use the weight for scale loss')
parser.add_argument('--att_scale_ncons_kernel_sizes', nargs='+', type=int, default=[5,5,5], help='kernels sizes in dynamic fusion net.')
parser.add_argument('--att_scale_ncons_channels', nargs='+', type=int, default=[16,16,1], help='channels in dynamic fusion net')

args = parser.parse_args()
print(args)

# Create model
print('Creating CNN model...')
model = DCCNet(use_cuda=use_cuda,
               checkpoint=args.checkpoint,
               ncons_kernel_sizes=args.ncons_kernel_sizes,
               ncons_channels=args.ncons_channels,
               sce_kernel_size=args.sce_kernel_size,
               sce_hidden_dim=args.sce_hidden_dim,
               att_scale_ncons_kernel_sizes=args.att_scale_ncons_kernel_sizes,
               att_scale_ncons_channels=args.att_scale_ncons_channels,
               )

#Multi-GPU support
model = nn.DataParallel(model)

# Set which parts of the model to train
if args.fe_finetune_params>0:
    for i in range(args.fe_finetune_params):
        for p in model.module.FeatureExtraction.model[-1][-(i+1)].parameters():
            p.requires_grad=True

print('Trainable parameters:')
count = 0
for i,param in enumerate(model.named_parameters()):
    name,p = param
    if p.requires_grad:
        count+=1
        print(str(count)+": "+name+"\t"+str(p.shape)+"\t")

print(model)


# Optimizer
print('using Adam optimizer')
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
cnn_image_size=(args.image_size,args.image_size)

Dataset = ImagePairDataset
train_csv = 'train_pairs.csv'
#val_pairs_nocoords.csv: for compute loss, with flip column in csv, no coordinates
#val_pairs.csv: for compute pck, with coordinates
val_nocoordinates_csv = 'val_pairs_nocoords.csv'
val_csv = 'image_pairs/val_pairs.csv'


normalization_tnf = NormalizeImageDict(['source_image','target_image'])
batch_preprocessing_fn = BatchTensorToVars(use_cuda=use_cuda)   

# Dataset and dataloader
dataset = Dataset(transform=normalization_tnf,
	              dataset_image_path=args.dataset_image_path,
	              dataset_csv_path=args.dataset_csv_path,
                  dataset_csv_file = train_csv,
                  output_size=cnn_image_size,
                  )

dataloader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, 
                        num_workers=0)

dataset_val = Dataset(transform=normalization_tnf,
                      dataset_image_path=args.dataset_image_path,
                      dataset_csv_path=args.dataset_csv_path,
                      dataset_csv_file=val_nocoordinates_csv,
                      output_size=cnn_image_size)

# compute val loss
dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size,
                            shuffle=True, num_workers=4)

# compute val pck
dataloader_val_pck = pfpascal_val_dataloader(image_size=args.image_size, eval_dataset_path=args.dataset_image_path, csv_file=val_csv) #load pfpascal val dataset

# Define checkpoint name
checkpoint_dir = os.path.join(args.result_model_dir,args.exp_name)
checkpoint_name = os.path.join(args.result_model_dir,args.exp_name,
                               datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")+'_'+args.result_model_fn + '.pth.tar')
log_name = os.path.join(args.result_model_dir,args.exp_name, 'logmain_'+args.exp_name+'.txt')
if not exists(dirname(log_name)):
    makedirs(dirname(log_name))
print('Checkpoint name: '+checkpoint_name)
    
# Train
best_val_pck = float("-inf")

loss_fn = lambda model,batch: weak_loss(model, batch, normalization='softmax', scaleloss_weight=args.scaleloss_weight)

# define epoch function
def process_epoch(mode,epoch,model,loss_fn,optimizer,dataloader,batch_preprocessing_fn,use_cuda=True,log_interval=50):
    epoch_loss = 0
    for batch_idx, batch in enumerate(dataloader):

        st = time.time()

        if mode=='train':            
            optimizer.zero_grad()
        tnf_batch = batch_preprocessing_fn(batch)
        loss = loss_fn(model,tnf_batch)
        loss_np = loss.data.cpu().numpy()[0]
        #loss_np = loss.data.cpu().numpy()
        epoch_loss += loss_np
        if mode=='train':
            loss.backward()
            optimizer.step()
        else:
            loss=None
        if batch_idx % log_interval == 0:
            print(mode.capitalize()+' Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.12f}\t\tcost time: {:.1f}'.format(
                epoch, batch_idx , len(dataloader),
                100. * batch_idx / len(dataloader), loss_np,time.time()-st))
    epoch_loss /= len(dataloader)
    print(mode.capitalize()+' set: Average loss: {:.12f}'.format(epoch_loss))
    return epoch_loss

train_loss = np.zeros(args.num_epochs)
val_loss = np.zeros(args.num_epochs)
val_pcks = np.zeros(args.num_epochs)

model.module.FeatureExtraction.eval()


print('Starting training...')
for epoch in range(1, args.num_epochs+1):
    st = time.time()
    train_loss_curepoch = process_epoch('train',epoch,model,loss_fn,optimizer,dataloader,batch_preprocessing_fn,log_interval=1)
    time_train = time.time()-st

    st = time.time()

    val_loss_curepoch = process_epoch('val', epoch, model, loss_fn, optimizer, dataloader_val, batch_preprocessing_fn, log_interval=1)

    time_valloss = time.time()-st

    st = time.time()
    val_pck_curepoch = pfdataset_pck(dataloader=dataloader_val_pck,model=model,verbose=False)
    time_valpck = time.time()-st

    train_loss[epoch - 1] = train_loss_curepoch
    val_loss[epoch - 1] = val_loss_curepoch
    val_pcks[epoch-1] = val_pck_curepoch

    # remember best loss
    is_best = val_pcks[epoch - 1] > best_val_pck
    best_val_pck = max(val_loss[epoch - 1], best_val_pck)
    save_checkpoint({
        'epoch': epoch,
        'args': args,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_pck': val_pcks,
        'best_val_pck':best_val_pck,
    }, is_best,checkpoint_name,save_all_epochs=False)

    message = 'Epoch{}\tTrain_loss{:.6f}\tcost time{:.1f}\tVal_loss{:.6f}\tcost time{:.1f}\tVal_pck{:.6f}\tcost time{:.1f}\n'.format\
        (epoch, train_loss_curepoch, time_train, val_loss_curepoch, time_valloss,val_pck_curepoch,time_valpck,)
    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)


print('Done!')
