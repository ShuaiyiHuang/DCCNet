# Thanks Qiuyue Wang's involvement in TSS evaluation code.

from __future__ import print_function, division
import os
import numpy as np
import torch

from torch.utils.data import DataLoader

from lib.eval_util_dynamic import corr_to_matches
from lib.eval_util_dynamic import flow_metrics

from lib.normalization import NormalizeImageDict
from lib.torch_util import BatchTensorToVars

from lib.dataloader import default_collate
from lib.tss_dataset import TSSDataset

from models.model_dynamic import DCCNet
import argparse

print('DCCNet evaluation script - TSS dataset')

use_cuda = torch.cuda.is_available()

# Argument parsing
parser = argparse.ArgumentParser(description='Compute TSS matches')
parser.add_argument('--checkpoint', type=str, default='./trained_models/best_dccnet.pth.tar')
parser.add_argument('--image_size', type=int, default=400)
parser.add_argument('--eval_dataset_path', type=str, default='./datasets/tss', help='path to TSS dataset')
parser.add_argument('--flow_output_dir', type=str, default='./datasets/dccnet_results')
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
Dataset = TSSDataset
collate_fn = default_collate
csv_file = 'test_pairs_tss.csv'

cnn_image_size = (args.image_size, args.image_size)

dataset = Dataset(csv_file=os.path.join(args.eval_dataset_path, csv_file),
                  dataset_path=args.eval_dataset_path,
                  transform=NormalizeImageDict(['source_image', 'target_image']),
                  output_size=cnn_image_size)
dataset.pck_procedure = 'scnet'

# Only batch_size=1 is supported for evaluation
batch_size = 1

dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0,
                        collate_fn=collate_fn)

batch_tnf = BatchTensorToVars(use_cuda=use_cuda)

model.eval()

# initialize vector for storing results
stats = {}
stats['point_tnf'] = {}
stats['point_tnf']['pck'] = np.zeros((len(dataset), 1))

# Compute
for i, batch in enumerate(dataloader):
    batch = batch_tnf(batch)
    batch_start_idx = batch_size * i

    out = model(batch)

    # get matches
    xA, yA, xB, yB, sB = corr_to_matches(out, do_softmax=True)

    matches = (xA, yA, xB, yB)
    # stats = pck_metric(batch, batch_start_idx, matches, stats, args, use_cuda)
    stats =  flow_metrics(batch, batch_start_idx, matches, stats, args, use_cuda)
    print('Batch: [{}/{} ({:.0f}%)]'.format(i, len(dataloader), 100. * i / len(dataloader)))

# Print results
results = stats['point_tnf']['pck']
good_idx = np.flatnonzero((results != -1) * ~np.isnan(results))
print('Total: ' + str(results.size))
print('Valid: ' + str(good_idx.size))
filtered_results = results[good_idx]
print('Flow files have been saved to '+args.flow_output_dir)
