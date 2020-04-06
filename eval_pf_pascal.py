from __future__ import print_function, division
import torch

from models.model_dynamic import DCCNet
from lib.eval_util_dynamic import pfpascal_test_dataloader,pfdataset_pck

import argparse

print('DCCNet evaluation script - PF Pascal dataset')

use_cuda = torch.cuda.is_available()

# Argument parsing
parser = argparse.ArgumentParser(description='Compute PF Pascal matches')
parser.add_argument('--checkpoint', type=str, default='./trained_models/best_dccnet.pth.tar')
parser.add_argument('--image_size', type=int, default=400)
parser.add_argument('--eval_dataset_path', type=str, default='./datasets/pf-pascal/', help='path to PF Pascal dataset')
parser.add_argument('--pck_alpha', type=float, default=0.1, help='pck alpha for evaluation')

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

# Dataset and dataloader
dataloader = pfpascal_test_dataloader(image_size=args.image_size,eval_dataset_path=args.eval_dataset_path)

pck = pfdataset_pck(dataloader=dataloader, model=model,verbose=True,alpha=args.pck_alpha)

