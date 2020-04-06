from __future__ import print_function, division
import torch
import torch.nn as nn
from collections import OrderedDict

from lib.conv4d import Conv4d
from lib.modules import FeatureExtraction,NeighConsensus,MutualMatching,FeatureCorrelation

from models.spatial_context_encoder import SpatialContextEncoder
from models.dynamic_fusion_att import DynamicFusionNet

class DCCNet(nn.Module):
    def __init__(self,
                 feature_extraction_cnn='resnet101',
                 feature_extraction_last_layer='',
                 feature_extraction_model_file=None,
                 ncons_kernel_sizes=[3,3,3],
                 ncons_channels=[10,10,1],
                 normalize_features=True,
                 train_fe=False,
                 use_cuda=True,
                 half_precision=False,
                 checkpoint=None,

                 sce_kernel_size = None,
                 sce_hidden_dim=None,

                 att_scale_ncons_kernel_sizes = None,  #hsy 0316
                 att_scale_ncons_channels = None,

                 ):
        
        super(DCCNet, self).__init__()


        self.use_cuda = use_cuda
        self.normalize_features = normalize_features

        self.half_precision = half_precision

        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe,
                                                   feature_extraction_cnn=feature_extraction_cnn,
                                                   feature_extraction_model_file=feature_extraction_model_file,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,
                                                   use_cuda=self.use_cuda)
                                                   
        self.FeatureCorrelation = FeatureCorrelation(shape='4D', normalization=False)
        self.SpatialContextEncoder = SpatialContextEncoder(kernel_size=sce_kernel_size, input_dim=sce_kernel_size * sce_kernel_size + 1024, hidden_dim=sce_hidden_dim)

        self.NeighConsensus = NeighConsensus(use_cuda=self.use_cuda,
                                             kernel_sizes=ncons_kernel_sizes,
                                             channels=ncons_channels)

        self.DynamicFusionNet = DynamicFusionNet(att_scale_ncons_kernel_sizes=att_scale_ncons_kernel_sizes, att_scale_ncons_channels=att_scale_ncons_channels)
        self.DynamicFusionNet.cuda()

        #################################################

        # Load weights
        if checkpoint is not None and checkpoint is not '':
            print('Loading checkpoint from{}...'.format(checkpoint))
            checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            checkpoint['state_dict'] = OrderedDict(
                [(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])

            # process dataparallel
            ckpt_statedict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if k[:7] == 'module.':
                    name = k[7:]  # remove `module.`
                else:
                    name = k

                ckpt_statedict[name] = v

            print('Copying weights...')
            self.load_state_dict(ckpt_statedict, strict=True)

            print('Done!')
        
        self.FeatureExtraction.eval()

        if self.half_precision:
            for p in self.NeighConsensus.parameters():
                p.data=p.data.half()
            for l in self.NeighConsensus.conv:
                if isinstance(l,Conv4d):
                    l.use_half=True


    # used only for foward pass at eval and for training with strong supervision
    def forward(self, tnf_batch):
        #Part1-a: extract features of different scales
        feat_set,corr_in_set = self.feat_compute_main(tnf_batch=tnf_batch)

        #Part2: Neighborhood concensus to produce score maps of different scales
        corr_out_set = self.scoremaps_compute_main(corr_in_set=corr_in_set)

        #Part3: Attention scale module
        A_scaleatts_set, B_scaleatts_set = self.scaleatt_compute_main(corr_out_set=corr_out_set)

        out = (corr_out_set,A_scaleatts_set,B_scaleatts_set)

        return out

    def feat_compute_main(self,tnf_batch):
        # scale1--local conv feature
        feature_A = self.FeatureExtraction(tnf_batch['source_image'])
        feature_B = self.FeatureExtraction(tnf_batch['target_image'])
        if self.half_precision:
            feature_A = feature_A.half()
            feature_B = feature_B.half()
        
        corr_lc = self.FeatureCorrelation(feature_A=feature_A,feature_B=feature_B)

        # scale2--context-aware semantic feature
        
        feature_A_embd = self.SpatialContextEncoder(feature_A)
        feature_B_embd = self.SpatialContextEncoder(feature_B)
        corr_embd = self.FeatureCorrelation(feature_A=feature_A_embd,feature_B=feature_B_embd)
        
        # output
        feat_scale_lc = torch.stack([feature_A,feature_B],dim=1) #(B,C1,H,W)->(B,2,C1,H,W)
        feat_scale_embd = torch.stack([feature_A_embd,feature_B_embd],dim=1) #(B,C2,H,W)->(B,2,C2,H,W)

        feat_set = [feat_scale_lc,feat_scale_embd]
        corr_set = [corr_lc,corr_embd]
       
        return feat_set,corr_set

    def scaleatt_compute_main(self,corr_out_set):
        A_scaleatts_set, B_scaleatts_set = self.DynamicFusionNet.forward(corr_out_set)

        return A_scaleatts_set, B_scaleatts_set

    def scoremaps_compute_main(self, corr_in_set):
        # Return list of 4D corrmap: [(B,1,Ha,Wa,Hb,Wb) for scale1, (B,1,Ha,Wa,Hb,Wb) for scale2,...]

        S = len(corr_in_set)
        corr_out_set = []
        B,_,Ha,Wa,Hb,Wb = corr_in_set[0].shape

        for si in range(S): #iterate over scales
            corr4d_in_si = corr_in_set[si]
            corr4d_out_si = self.run_match_model(corr4d=corr4d_in_si)
            corr_out_set.append(corr4d_out_si)

        corr_out_set = torch.stack(corr_out_set,dim=1) #(B,S,1,Ha,Wa,Hb,Wb)
        assert (corr_out_set.shape==(B,S,1,Ha,Wa,Hb,Wb)),'corr_out_set shape {} is not consistent with{},{},1,{},{},{},{}'.format(corr_out_set.shape,
                                                                                                                           B,S,Ha,Wa,Hb,Wb)

        return corr_out_set

    def run_match_model(self,corr4d):

        corr4d = MutualMatching(corr4d)

        corr4d = self.NeighConsensus(corr4d)

        corr4d = MutualMatching(corr4d)

        return corr4d