import torch
import numpy as np
# Shuaiyi Huang
# Dynamic loss

def weak_loss(model, batch, normalization='softmax', scaleloss_weight=None):
    b = batch['source_image'].size(0)

    #positive
    score_pos_merge, score_pos_overscales = weak_loss_singlebatch(model=model,batch=batch,normalization=normalization,)

    #negative
    batch['source_image'] = batch['source_image'][np.roll(np.arange(b), -1), :]  # roll
    score_neg_merge, score_neg_overscales = weak_loss_singlebatch(model=model, batch=batch,normalization=normalization, )

    # loss
    loss_merge = score_neg_merge - score_pos_merge

    if scaleloss_weight:
        loss_scales = torch.sum(torch.cat(score_neg_overscales))-torch.sum(torch.cat(score_pos_overscales))
        loss = loss_merge+scaleloss_weight*loss_scales
    else:
        loss = loss_merge

    return loss


def weak_loss_singlebatch(model,batch,normalization='softmax', alpha=30):
    # positive
    out = model(batch)

    # corr_out_set = out['corr'] #(B,S,1,Ha,Wa,Hb,Wb)
    # A_scaleatts_set = out['scaleatt']['A'] #(B,S,Ha,Wa)
    # B_scaleatts_set = out['scaleatt']['B'] #(B,S,Hb,Wb)

    corr_out_set, A_scaleatts_set, B_scaleatts_set = out

    B,S,_,Ha,Wa,Hb,Wb = corr_out_set.shape

    score_pos_overscales = []
    M_A_norm_overscales = []
    M_B_norm_overscales = []
    for si in range(S): #iterate over scales
        corr_out_si = corr_out_set[:,si,:,:,:,:,:].contiguous() #todo why
        score_pos_si,M_A_norm,M_B_norm = score_for_single_corr4d(corr4d=corr_out_si,normalization=normalization) #MA_norm:(B,LB,Ha,Wa)

        #add
        score_pos_overscales.append(score_pos_si)
        M_A_norm_overscales.append(M_A_norm)
        M_B_norm_overscales.append(M_B_norm)

    M_A_norm_overscales = torch.stack(M_A_norm_overscales,dim=1) #(B,LB,Ha,Wa)->(B,S,LB,Ha,Wa)
    M_B_norm_overscales = torch.stack(M_B_norm_overscales, dim=1) #(B,LA,Hb,Wb)->(B,S,LA,Hb,Wb)

    #merge scoremap using atts
    MergedA = torch.sum(M_A_norm_overscales*A_scaleatts_set.view(B,S,1,Ha,Wa),dim=1)         #(B,LB,Ha,Wa)
    MergedB = torch.sum(M_B_norm_overscales * B_scaleatts_set.view(B, S, 1, Hb, Wb), dim=1)  #(B,LA,Hb,Wb)

    # compute matching scores
    scores_B_merge, _ = torch.max(MergedB, dim=1)
    scores_A_merge, _ = torch.max(MergedA, dim=1)
    score_pos_merge = torch.mean(scores_A_merge + scores_B_merge) / 2

    return score_pos_merge,score_pos_overscales

def score_for_single_corr4d(corr4d,normalization='softmax'):
    if normalization is None:
        normalize = lambda x: x
    elif normalization == 'softmax':
        normalize = lambda x: torch.nn.functional.softmax(x, 1)
    elif normalization == 'l1':
        normalize = lambda x: x / (torch.sum(x, dim=1, keepdim=True) + 0.0001)

    batch_size = corr4d.size(0)
    feature_size = corr4d.size(2)
    nc_B_Avec = corr4d.view(batch_size, feature_size * feature_size, feature_size,
                            feature_size)  # [batch_idx,k_A,i_B,j_B] (B,LA,HB,WB)
    nc_A_Bvec = corr4d.view(batch_size, feature_size, feature_size, feature_size * feature_size).permute(0, 3, 1, 2)  #(B,LB,HA,WA)

    #normalize
    nc_B_Avec = normalize(nc_B_Avec)
    nc_A_Bvec = normalize(nc_A_Bvec)

    # compute matching scores
    scores_B, _ = torch.max(nc_B_Avec, dim=1)
    scores_A, _ = torch.max(nc_A_Bvec, dim=1)
    score_pos = torch.mean(scores_A + scores_B) / 2

    return score_pos,nc_A_Bvec,nc_B_Avec
