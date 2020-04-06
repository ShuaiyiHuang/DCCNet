# Shuaiyi Huang
# Dynamic Fusion Network based on scale attention

import torch
import torch.nn as nn
import torch.nn.functional as Func
from lib.modules import NeighConsensus


class DynamicFusionNet(nn.Module):
    '''
    Generate attention maps to dynamically fuse S kind of 4D correlation maps based on attention mechanism
    Input:
            corr_set: tensor of shape (B,S,1,Ha,Wa,Hb,Wb), S is num of scales, S=2 by default (local and context)
    Output:
            att_maps: attention maps for Image Ia and Ib, list of length 2. att_maps[0] is tensor of shape (B,S,Ha,Wa), S attention maps for Ia.
    '''

    def __init__(self,S = 2, att_scale_ncons_kernel_sizes = None, att_scale_ncons_channels = None):
        super(DynamicFusionNet, self).__init__()

        self.S = S # num of scales
        att_scale_input_dim = self.S * 25 * 25

        use_cuda = True

        self.extract_corrfeas = NeighConsensus(use_cuda=use_cuda,
                                               kernel_sizes=att_scale_ncons_kernel_sizes,
                                               channels=att_scale_ncons_channels)

        self.att = nn.Sequential(
                nn.Conv2d(att_scale_input_dim, S, kernel_size=1, padding=0),)

        print('verbose in DynamicFusionNet.....input_attscale_dim {}, scale S {},'
              'att_scale_ncons_kernel_sizes {}, att_scale_ncons_channels {}'.
              format(att_scale_input_dim, S,att_scale_ncons_kernel_sizes, att_scale_ncons_channels))

        return

    def forward(self, corr_set):

        att_maps= self._forward_corr_to_att(corr_set=corr_set)

        return att_maps

    def _forward_feaextracture_by_4D(self,corr_set):
        '''
        Feature extraction from 4D corr maps using 4D Conv
        Input:
            corr_set: tensor of shape (B,S,1,Ha,Wa,Hb,Wb)
        Return:
            corr_4dfea_set: tensor of shape (B,S,1,Ha,Wa,Hb,Wb)
        '''
        S = self.S
        B, _, c, Ha, Wa, Hb, Wb = list(corr_set.shape)
        assert (_==S)

        fea_in = corr_set.view(B*S,1,Ha,Wa,Hb,Wb)
        fea_out = self.extract_corrfeas(fea_in) #(B*S,1,Ha,Wa,Hb,Wb)
        corr_4dfea_set = fea_out.view(B,S,1,Ha,Wa,Hb,Wb)

        return corr_4dfea_set

    def _forward_corr_to_att(self, corr_set):
        '''
        Given S scales 4D correlation maps of Ia and Ib, generate attention maps for Ia and Ib.

        Inputs:
            corr_set: tensor of shape (B,S,1,Ha,Wa,Hb,Wb)
        Return:
            A_scaleatts_set: tensor of shape (B,S,H,W) with att from scale=1 to scale=S
            B_scaleatts_set: tensor of shape (B,S,H,W) with att from scale=1 to scale=S
        '''
        # Num of scale
        S = self.S

        if type(corr_set) ==list:
            assert (len(corr_set[0].shape)==6)
            corr_set = torch.stack(corr_set,dim=1) #(B,S,1,Ha,Wa,Hb,Wb)

        # Apply 4D conv for corrmap feature extraction
        corr_set = self._forward_feaextracture_by_4D(corr_set=corr_set)

        B,_,c,Ha,Wa,Hb,Wb = list(corr_set.shape)
        La = Ha*Wa
        Lb = Hb*Wb
        assert (len(corr_set.shape)==7 and (_==S) and (c==1))

        corr_set = corr_set.squeeze(2) #(B,S,Ha,Wa,Hb,Wb)
        assert (corr_set.shape==(B,S,Ha,Wa,Hb,Wb))

        # prepare tensor A_att_in of shape [B,SxLB,HA,WA]
        A_att_in = corr_set.view(B,S,Ha,Wa,Hb*Wb).permute(0,1,4,2,3).contiguous()
        assert (A_att_in.shape==(B,S,Lb,Ha,Wa))
        A_att_in = A_att_in.view(B,S*Lb,Ha,Wa)

        # prepare tensor B_att_in of shape [B,SxLA,HB,WB]
        B_att_in = corr_set.view(B,S,Ha*Wa,Hb,Wb)
        assert (B_att_in.shape==(B,S,La,Hb,Wb))
        B_att_in = B_att_in.view(B,S*La,Hb,Wb)

        # compute att maps
        A_scaleatts_set = self.att.forward(A_att_in) #(B,S*Lb,Ha,Wa)->(B,S,Ha,Wa)
        B_scaleatts_set = self.att.forward(B_att_in) #(B,S*La,Hb,Wb)->(B,S,Hb,Wb)

        assert (A_scaleatts_set.shape==(B,S,Ha,Wa))
        assert (B_scaleatts_set.shape==(B,S,Hb,Wb))

        A_scaleatts_set = Func.softmax(A_scaleatts_set.view(B,S,Ha*Wa),dim=1).view(B,S,Ha,Wa)
        B_scaleatts_set = Func.softmax(B_scaleatts_set.view(B, S, Hb * Wb), dim=1).view(B, S, Hb, Wb)

        return A_scaleatts_set,B_scaleatts_set


if __name__ == '__main__':
    print()