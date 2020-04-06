import torch
import torch.nn
import numpy as np
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader

from lib.dataloader import default_collate
from lib.torch_util import BatchTensorToVars

from lib.point_tnf_dynamic import corr_to_matches
from lib.normalization import NormalizeImageDict

# dense flow
from geotnf.flow import th_sampling_grid_to_np_flow,write_flo_file
from lib.py_util import create_file_path

from lib.pf_dataset import PFPascalDataset
from lib.point_tnf_dynamic import PointsToUnitCoords, PointsToPixelCoords, bilinearInterpPointTnf

def pck(source_points,warped_points,L_pck,alpha=0.1):
    # compute precentage of correct keypoints
    batch_size=source_points.size(0)
    pck=torch.zeros((batch_size))
    for i in range(batch_size):
        p_src = source_points[i,:]
        p_wrp = warped_points[i,:]
        N_pts = torch.sum(torch.ne(p_src[0,:],-1)*torch.ne(p_src[1,:],-1))
        point_distance = torch.pow(torch.sum(torch.pow(p_src[:,:N_pts]-p_wrp[:,:N_pts],2),0),0.5)
        L_pck_mat = L_pck[i].expand_as(point_distance)
        correct_points = torch.le(point_distance,L_pck_mat*alpha)
        pck[i]=torch.mean(correct_points.float())
    return pck


def pck_metric(batch,batch_start_idx,matches,stats,args,use_cuda=True,alpha=0.1):
       
    source_im_size = batch['source_im_size']
    target_im_size = batch['target_im_size']

    source_points = batch['source_points']
    target_points = batch['target_points']
    
    # warp points with estimated transformations
    target_points_norm = PointsToUnitCoords(target_points,target_im_size)

    # compute points stage 1 only
    warped_points_norm = bilinearInterpPointTnf(matches,target_points_norm)
    warped_points = PointsToPixelCoords(warped_points_norm,source_im_size)
    
    L_pck = batch['L_pck'].data
    
    current_batch_size=batch['source_im_size'].size(0)
    indices = range(batch_start_idx,batch_start_idx+current_batch_size)

    # compute PCK
    pck_batch = pck(source_points.data, warped_points.data, L_pck,alpha=alpha)
    stats['point_tnf']['pck'][indices] = pck_batch.unsqueeze(1).cpu().numpy()
        
    return stats

#2019/02/22 Friday Shuaiyi
'''
Given dataset and model, turn it into dataloader with batchsize=1, evaluate and report pck
'''

def pfpascal_test_dataloader(image_size,eval_dataset_path,csv_file = 'image_pairs/test_pairs.csv'):
    # Dataset and dataloader
    Dataset = PFPascalDataset
    collate_fn = default_collate

    cnn_image_size = (image_size, image_size)

    dataset = Dataset(csv_file=os.path.join(eval_dataset_path, csv_file),
                      dataset_path=eval_dataset_path,
                      transform=NormalizeImageDict(['source_image', 'target_image']),
                      output_size=cnn_image_size)
    dataset.pck_procedure = 'scnet'

    # Only batch_size=1 is supported for evaluation
    batch_size = 1

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0,
                            collate_fn=collate_fn)

    return dataloader

def pfdataset_pck(dataloader, model, verbose = False,alpha=0.1):
    model.eval()
    use_cuda = torch.cuda.is_available()
    collate_fn = default_collate
    batch_size = 1

    batch_tnf = BatchTensorToVars(use_cuda=use_cuda)

    model.eval()

    # initialize vector for storing results
    stats = {}
    stats['point_tnf'] = {}
    stats['point_tnf']['pck'] = np.zeros((len(dataloader.dataset), 1))

    # Compute
    for i, batch in enumerate(dataloader):

        batch = batch_tnf(batch)
        batch_start_idx = batch_size * i

        # corr4d = model(batch)
        out = model(batch)

        # get matches
        xA, yA, xB, yB, sB = corr_to_matches(out, do_softmax=True)

        matches = (xA, yA, xB, yB)
        stats = pck_metric(batch, batch_start_idx, matches, stats, None, use_cuda,alpha=alpha)
        if verbose:
            print('Batch: [{}/{} ({:.0f}%)]'.format(i, len(dataloader), 100. * i / len(dataloader)))

    # Print results
    results = stats['point_tnf']['pck']
    good_idx = np.flatnonzero((results != -1) * ~np.isnan(results))
    if verbose:
        print('Total: ' + str(results.size))
        print('Valid: ' + str(good_idx.size))
    filtered_results = results[good_idx]

    if verbose:
        print('PCK:', '{:.2%}'.format(np.mean(filtered_results)))

    pck_value = np.mean(filtered_results)

    return pck_value

'''
Given val dataset and model, turn it into dataloader with batchsize=1, evaluate and report pck
'''
def pfpascal_val_dataloader(image_size,eval_dataset_path,csv_file = 'image_pairs/val_pairs.csv'):
    # Dataset and dataloader
    Dataset = PFPascalDataset
    collate_fn = default_collate

    cnn_image_size = (image_size, image_size)

    dataset = Dataset(csv_file=os.path.join(eval_dataset_path, csv_file),
                      dataset_path=eval_dataset_path,
                      transform=NormalizeImageDict(['source_image', 'target_image']),
                      output_size=cnn_image_size)
    dataset.pck_procedure = 'scnet'

    # Only batch_size=1 is supported for evaluation todo
    batch_size = 1

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0,
                            collate_fn=collate_fn)

    return dataloader

# for dense flow evaluation
def flow_metrics(batch, batch_start_idx, matches, stats, args, use_cuda=True):
    result_path = args.flow_output_dir

    # pt = PointTnf(use_cuda=use_cuda)

    batch_size = batch['source_im_size'].size(0)
    for b in range(batch_size):
        h_src = int(batch['source_im_size'][b, 0].data.cpu().numpy())
        w_src = int(batch['source_im_size'][b, 1].data.cpu().numpy())
        h_tgt = int(batch['target_im_size'][b, 0].data.cpu().numpy())
        w_tgt = int(batch['target_im_size'][b, 1].data.cpu().numpy())

        grid_X, grid_Y = np.meshgrid(np.linspace(-1, 1, w_tgt), np.linspace(-1, 1, h_tgt))
        grid_X = torch.FloatTensor(grid_X).unsqueeze(0).unsqueeze(3)
        grid_Y = torch.FloatTensor(grid_Y).unsqueeze(0).unsqueeze(3)
        grid_X = Variable(grid_X, requires_grad=False)
        grid_Y = Variable(grid_Y, requires_grad=False)
        if use_cuda:
            grid_X = grid_X.cuda()
            grid_Y = grid_Y.cuda()

        grid_X_vec = grid_X.view(1, 1, -1)
        grid_Y_vec = grid_Y.view(1, 1, -1)

        grid_XY_vec = torch.cat((grid_X_vec, grid_Y_vec), 1)

        def pointsToGrid(x, h_tgt=h_tgt, w_tgt=w_tgt):
            return x.contiguous().view(1, 2, h_tgt, w_tgt).transpose(1, 2).transpose(2, 3)

        idx = batch_start_idx + b
        source_im_size = batch['source_im_size']
        warped_points_norm = bilinearInterpPointTnf(matches, grid_XY_vec)

        # warped_points = PointsToPixelCoords(warped_points_norm,source_im_size)
        warped_points = pointsToGrid(warped_points_norm)

        # grid_aff = pointsToGrid(pt.affPointTnf(theta_aff[b, :].unsqueeze(0), grid_XY_vec))
        flow_aff = th_sampling_grid_to_np_flow(source_grid=warped_points, h_src=h_src, w_src=w_src)
        flow_aff_path = os.path.join(result_path, batch['flow_path'][b])

        create_file_path(flow_aff_path)

        write_flo_file(flow_aff, flow_aff_path)

        idx = batch_start_idx + b
    return stats