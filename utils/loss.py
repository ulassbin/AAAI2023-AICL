import torch
import torch.nn as nn
import torch.nn.functional as F

from NCELoss.NNIICLUV_Tests.loss import InfoNCELoss

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_criterion = nn.BCELoss()

    def forward(self, logits, label):
        label = label / torch.sum(label, dim=1, keepdim=True) + 1e-10
        loss = -torch.mean(torch.sum(label * F.log_softmax(logits, dim=1), dim=1), dim=0)
        return loss


class GeneralizedCE(nn.Module):
    def __init__(self, q):
        self.q = q
        super(GeneralizedCE, self).__init__()

    def forward(self, logits, label):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]
        pos_factor = torch.sum(label, dim=1) + 1e-7
        neg_factor = torch.sum(1 - label, dim=1) + 1e-7
        first_term = torch.mean(torch.sum(((1 - (logits + 1e-7)**self.q)/self.q) * label, dim=1)/pos_factor)
        second_term = torch.mean(torch.sum(((1 - (1 - logits + 1e-7)**self.q)/self.q) * (1-label), dim=1)/neg_factor)
        return first_term + second_term


class VidPseudoLoss(nn.Module):
    def __init__(self):
        super(VidPseudoLoss, self).__init__()
        self.bce_criterion = nn.BCELoss()
    
    def forward(self, video_scores, pseudo_label):
        loss = self.bce_criterion(video_scores, pseudo_label)
        return loss
        

class LatentLoss(nn.Module):
    def __init__(self):
        super(LatentLoss, self).__init__()
        self.mse_criterion = nn.MSELoss()
    
    def forward(self, base_feature, decoded_feature):
        loss = self.mse_criterion(base_feature, decoded_feature)
        return loss



class TotalLoss(nn.Module):
    def __init__(self, cfg):
        super(TotalLoss, self).__init__()
        self.action_criterion = ActionLoss() # Replace 1
        self.snico_criterion = SniCoLoss() # Replace 2

        self.nce_criterion = InfoNCELoss()
        self.vid_pseudo_loss = VidPseudoLoss()
        self.latent_loss = LatentLoss()
        self.nce_weight = cfg.nce_weight
        self.pseudo_weight = cfg.pseudo_weight
        self.latent_weight = cfg.latent_loss_weight


    def forward(self, video_scores, label, contrast_pairs, sampled_embeddings, positives, negatives, pseudo_label, enc_decoder_embeddings):
        input_feature, decoded_inter, decoded_intra = enc_decoder_embeddings
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_nce = self.nce_criterion(sampled_embeddings, positives, negatives)
        loss_pseudo = self.vid_pseudo_loss(video_scores, pseudo_label)
        loss_latent_inter = self.latent_loss(input_feature, decoded_inter)
        loss_latent_intra = self.latent_loss(input_feature, decoded_intra)
        loss_total = loss_cls + 0.01 * loss_snico + self.nce_weight * loss_nce + self.pseudo_weight * loss_pseudo
        
        loss_total += self.latent_weight *(loss_latent_inter + loss_latent_intra)

        loss_dict = {
            'Loss/Total': loss_total,
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/Intra': loss_nce,
            'Loss/Pseudo': loss_pseudo,
            'Loss/LatentInter': loss_latent_inter,
            'Loss/LatentIntra': loss_latent_intra,
            'Loss/LatentCombined': loss_latent_inter + loss_latent_intra
        }

        return loss_total, loss_dict
