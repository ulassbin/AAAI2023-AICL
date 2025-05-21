import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import json
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from inference_thumos import inference
from utils import misc_utils
from torch.utils.data import Dataset
from dataset.thumos_features import ThumosFeature
from utils.loss import CrossEntropyLoss, GeneralizedCE, LatentLoss, VidPseudoLoss
from NCELoss.NNIICLUV_Tests.loss import InfoNCELoss
from config.config_thumos import Config, parse_args, class_dict
from models.model import AICL

from NCELoss.NNIICLUV_Tests.custom_queue import Queue


np.set_printoptions(formatter={'float_kind': "{:.2f}".format})

np.set_printoptions(threshold=np.inf)

def load_weight(net, config):
    if config.load_weight:
        model_file = os.path.join(config.model_path, "CAS_Only.pkl")
        print("loading from file for training: ", model_file)
        pretrained_params = torch.load(model_file)

        selected_params = OrderedDict()
        for k, v in pretrained_params.items():
            if 'base_module' in k:
                selected_params[k] = v

        model_dict = net.state_dict()
        model_dict.update(selected_params)
        net.load_state_dict(model_dict)


def get_dataloaders(config):
    train_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='train',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='random', supervision='strong'),
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers)

    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='test',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='uniform', supervision='strong'),
        batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    pre_train_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='train',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='random', supervision='strong'),
        batch_size=config.pretrain_batch_size,
        shuffle=True, num_workers=config.num_workers)


    return train_loader, test_loader, pre_train_loader


def set_seed(config):
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.1):                #　　0.1
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):

        IA_refinement = self.NCE(
            torch.mean(contrast_pairs['IA'], 1),
            torch.mean(contrast_pairs['CA'], 1),
            contrast_pairs['CB']
        )

        IB_refinement = self.NCE(
            torch.mean(contrast_pairs['IB'], 1),
            torch.mean(contrast_pairs['CB'], 1),
            contrast_pairs['CA']
        )

        CA_refinement = self.NCE(
            torch.mean(contrast_pairs['CA'], 1),
            torch.mean(contrast_pairs['IA'], 1),
            contrast_pairs['CB']
        )

        CB_refinement = self.NCE(
            torch.mean(contrast_pairs['CB'], 1),
            torch.mean(contrast_pairs['IB'], 1),
            contrast_pairs['CA']
        )

        loss = IA_refinement + IB_refinement + CA_refinement + CB_refinement
        return loss


class ThumosTrainer():
    def __init__(self, config):
        # config
        self.config = config
        
        # network
        self.net = AICL(config)
        self.net = self.net.cuda()
        self.writter = SummaryWriter(config.log_path)
        self.softmax = nn.Softmax(dim=1)
        # data
        self.train_loader, self.test_loader, self.pre_train_loader = get_dataloaders(self.config)

        # loss, optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, betas=(0.9, 0.999), weight_decay=0.0005)
        self.criterion = CrossEntropyLoss()
        self.nce_criterion = InfoNCELoss()
        self.vid_pseudo_loss = LatentLoss()
        self.latent_loss = LatentLoss()
        self.Lgce = GeneralizedCE(q=self.config.q_val)

        # Memory module
        self.queue = Queue(queue_size=config.queue_size, embedding_dim=config.proj_dim, device='cuda')
        self.nn_queue = Queue(queue_size=config.queue_size, embedding_dim=config.proj_dim, device='cuda')
        self.initialized = False

        # parameters
        self.best_mAP = -1 # init
        self.step = 0
        self.total_loss_per_epoch = 0
        
    def initialize(self, embeddings):
        print('Initializing with ', embeddings.shape)
        self.queue.enqueue(embeddings)
        self.initialized = True
        return


    def sample_embeddings(self, embeddings):
        batch_size, t, feature_dim = embeddings.shape  # Assuming fixed t
        new_size = int(self.config.sampling_rate * t)
        sample_indexes = torch.randint(0, t, (batch_size, new_size), device=embeddings.device)
        # Use advanced indexing to preserve gradients
        sampled = embeddings[torch.arange(batch_size).unsqueeze(1), sample_indexes]
        return sampled  # Keeps gradients

    def test(self):
        self.net.eval()

        with torch.no_grad():
            model_filename = "CAS_Only.pkl"
            self.config.model_file = os.path.join(self.config.model_path, model_filename)
            _mean_ap, test_acc, mAp_dict = inference(self.net, self.config, self.test_loader, model_file=self.config.model_file)
            print("cls_acc={:.5f} map={:.5f}".format(test_acc*100, _mean_ap*100))
            if self.writter:
                self.writter.add_scalar('Test Performance/Accuracy', test_acc, self.step)
                self.writter.add_scalar('Test Performance/mAP@AVG', _mean_ap, self.step)
                for key, value in mAp_dict.items():
                  self.writter.add_scalar('mAp@tIOU/mAP@{:.1f}'.format(key), value, self.step)

    def calculate_pesudo_target(self, batch_size, label, topk_indices):
        cls_agnostic_gt = []
        cls_agnostic_neg_gt = []
        for b in range(batch_size):
            label_indices_b = torch.nonzero(label[b, :])[:,0]
            topk_indices_b = topk_indices[b, :, label_indices_b] # topk, num_actions
            cls_agnostic_gt_b = torch.zeros((1, 1, self.config.num_segments)).cuda()

            # positive examples
            for gt_i in range(len(label_indices_b)):
                cls_agnostic_gt_b[0, 0, topk_indices_b[:, gt_i]] = 1
            cls_agnostic_gt.append(cls_agnostic_gt_b)

        return torch.cat(cls_agnostic_gt, dim=0)  # B, 1, num_segments


    def calculate_all_losses1(self, contrast_pairs, contrast_pairs_r,contrast_pairs_f, cas_top, label, action_flow, action_rgb, cls_agnostic_gt, actionness1, actionness2):
        self.contrastive_criterion = ContrastiveLoss()
        loss_contrastive = self.contrastive_criterion(contrast_pairs) + self.contrastive_criterion(contrast_pairs_r) + self.contrastive_criterion(contrast_pairs_f)

        base_loss = self.criterion(cas_top, label)
        class_agnostic_loss = self.Lgce(action_flow.squeeze(1), cls_agnostic_gt.squeeze(1)) + self.Lgce(action_rgb.squeeze(1), cls_agnostic_gt.squeeze(1))

        modality_consistent_loss = 0.5 * F.mse_loss(action_flow, action_rgb) + 0.5 * F.mse_loss(action_rgb, action_flow)
        action_consistent_loss = 0.5 * F.mse_loss(actionness1, actionness2) + 0.5 * F.mse_loss(actionness2, actionness1)

        cost = self.config.classification_weight * base_loss + class_agnostic_loss  + self.config.modality_weight*modality_consistent_loss 
        + self.config.contrastive_weight*loss_contrastive + self.config.action_consistency_weight*action_consistent_loss
        
        # Module additional loss terms
        # cost += self.latent_weight * (loss_latent_inter + loss_latent_intra) # encoding-decoding loss
        # cost += self.nce_weight * loss_nce # snippetwise memory refinement loss
        # cost += self.pseudo_weight * loss_pseudo # videowise memory refinement loss
        if self.writter:
            self.writter.add_scalar('Loss/Action', base_loss.cpu().item(), self.step)
            self.writter.add_scalar('Loss/Class_Agnostic_Loss', class_agnostic_loss.cpu().item(), self.step)
            self.writter.add_scalar('Loss/Modality_Consistent_Loss', modality_consistent_loss.cpu().item(), self.step)
            self.writter.add_scalar('Loss/Action_Consistent_Loss', action_consistent_loss.cpu().item(), self.step)
            self.writter.add_scalar('Loss/Contrastive_Loss', loss_contrastive.cpu().item(), self.step)
            self.writter.add_scalar('Loss/Total_Base', cost.cpu().item(), self.step)
        return cost

    def evaluate(self, epoch=0):
        if self.step % self.config.detection_inf_step == 0:
            self.total_loss_per_epoch /= self.config.detection_inf_step

            with torch.no_grad():
                self.net = self.net.eval()
                mean_ap, test_acc, mAp_dict = inference(self.net, self.config, self.test_loader, model_file=None)
                self.net = self.net.train()

            if mean_ap > self.best_mAP:
                self.best_mAP = mean_ap
                torch.save(self.net.state_dict(), os.path.join(self.config.model_path, "CAS_Only.pkl"))

            if self.writter:
                self.writter.add_scalar('Test Performance/Accuracy', test_acc, self.step)
                self.writter.add_scalar('Test Performance/mAP@AVG', mean_ap, self.step)
                for key, value in mAp_dict.items():
                  self.writter.add_scalar('mAp@tIOU/mAP@{:.1f}'.format(key), value, self.step)


            print("epoch={:5d}  step={:5d}  Loss={:.4f}  cls_acc={:5.2f}  best_map={:5.2f}".format(
                    epoch, self.step, self.total_loss_per_epoch, test_acc * 100, self.best_mAP * 100))

            self.total_loss_per_epoch = 0

    def get_topk(self, cas):
        k_targets = cas.shape[1] // 8 # self.config.num_segments // 8
        _, topk_indices = torch.topk(cas, k_targets, dim=1)
        # _, topk_indices1 = torch.topk(combined_cas, r, dim=1)
        cas_top = torch.mean(torch.gather(cas, 1, topk_indices), dim=1)

        return cas_top, topk_indices

    def forward_pass(self, _data):
        (cas, action_flow, action_rgb, contrast_pairs,contrast_pairs_r,contrast_pairs_f, 
         actionness1, actionness2, aness_bin1, aness_bin2, all_embeddings ) = self.net(_data)

        combined_cas = misc_utils.instance_selection_function(torch.softmax(cas.detach(), -1),
                                                              action_flow.permute(0, 2, 1).detach(),
                                                              action_rgb.permute(0, 2, 1))

        cas_top, topk_indices = self.get_topk(combined_cas)
        cas_top = torch.mean(torch.gather(cas, 1, topk_indices), dim=1)
        return cas_top, topk_indices, action_flow, action_rgb, contrast_pairs,contrast_pairs_r,contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2, all_embeddings
    
    def forward_pass_from_embeddings(self, latent_embeddings):
        cas = self.net.forward_with_embeddings(latent_embeddings)
        cas_top, topk_indices = self.get_topk(cas)
        return cas_top
        
    def forward_pass_with_k_embeddings(self, topk_indices, distances):
        cas_targets = self.queue.get_fused_cas_targets(self.net, topk_indices, distances)
        cas_top, topk_action_indices = self.get_topk(cas_targets)
        return cas_top, cas_targets

    def calculate_module_losses(self, video_scores, pseudo_video_scores, input_feature, decoded_inter, decoded_intra, sampled_embeddings, positives, negatives):
        loss_pseudo = self.vid_pseudo_loss(video_scores, pseudo_video_scores)
        loss_nce = self.nce_criterion(sampled_embeddings, positives, negatives)
        loss_latent_inter = self.latent_loss(input_feature, decoded_inter)
        loss_latent_intra = self.latent_loss(input_feature, decoded_intra)
        loss_module = self.config.nce_weight * loss_nce + self.config.pseudo_weight * loss_pseudo + self.config.latent_loss_weight * (loss_latent_inter + loss_latent_intra)
        loss_dict = {
            'Loss/Total_Module': loss_module,
            'Loss/NCE': loss_nce,
            'Loss/Pseudo': loss_pseudo,
            'Loss/Latent_Inter': loss_latent_inter,
            'Loss/Latent_Intra': loss_latent_intra
        }
        if self.writter:
            for key, value in loss_dict.items():
              self.writter.add_scalar('{}'.format(key), value.cpu().item(), self.step)

        return loss_module

    def get_positives(self, intra_embeddings, temporal, embedding_dim, debug=False): # In future get K input
        batch_size, temporal, embedding_dim = intra_embeddings.shape
        intra_embeddings = intra_embeddings.reshape(batch_size * temporal, embedding_dim)
        nn_indices, nn_embeddings, nn_labels = self.queue.find_nearest_neighbors(intra_embeddings)
        nn_embeddings = nn_embeddings.reshape(batch_size, temporal, embedding_dim)
        nn_indices = nn_indices.reshape(batch_size, temporal)
        if nn_labels is not None:
          nn_labels = nn_labels.reshape(batch_size, temporal, 3)
        return nn_indices, nn_embeddings, nn_labels


    def get_positives_video_distance(self, full_embeddings, temporal, embedding_dim, k=1, debug=False):
        # In this function we will get the positives by using fft based distance calculation
        batch_size, temporal, embedding_dim = full_embeddings.shape
        polled_vids = batch_size
        distances, vid_indices = self.queue.find_nearest_vids(full_embeddings, self.config.sampled_vid_num)# Implement this
        #print('Vid indices shape: ', vid_indices.shape)
        #print('Distances shape: ', distances.shape)
        vid_embeddings = self.queue.getVidDataBatched(vid_indices)
        #if vid_labels is not None:
        #    vid_labels = vid_labels.reshape(batch_size, 3)
        return vid_embeddings, vid_indices, distances#, vid_labels
    

    def pretrain_encoder_decoder_step(self, net, loader_iter, step):
        net.train()
        data, label, _, _, _ = next(loader_iter)
        data = data.cuda()
        label = label.cuda()
        self.optimizer.zero_grad()
        cas, _, _, _, _, _, _, _, _, _, all_embeddings = net(data)
        criterion = LatentLoss()
        decoded_inter = all_embeddings['decoded_inter']
        decoded_intra = all_embeddings['decoded_intra']
        cost = self.config.latent_loss_pre * (criterion(data, decoded_inter) + criterion(data, decoded_intra))/2.0
        cost.backward()
        self.optimizer.step()
        self.writter.add_scalar('PRE_Latent Loss', cost.cpu().item(), step)
        return cost

    def pretrain_encoding(self):
        if self.config.pretrain_encoder_decoder:
            for step in range(1, self.config.pretrain_num_iters + 1):
                if (step - 1) % len(self.pre_train_loader) == 0:
                    loader_iter = iter(self.pre_train_loader)
                
                cost = self.pretrain_encoder_decoder_step(self.net, loader_iter, step)
                if step == 1 or step % self.config.print_freq == 0:
                    print(('PRETRAIN: Step: [{0:04d}/{1}]\t' \
                        'Loss {loss:.4f} \t'.format(
                        step, self.config.pretrain_num_iters, loss=cost.cpu().item())))
            del self.pre_train_loader
            del loader_iter

    def train(self):
        # resume training
        load_weight(self.net, self.config)

        # training
        for epoch in range(self.config.num_epochs):
            self.total_loss_per_epoch = 0
            for _data, _label, temp_anno, _, _ in self.train_loader:

                batch_size = _data.shape[0]
                _data, _label = _data.cuda(), _label.cuda()
                self.optimizer.zero_grad()

                # forward pass
                (cas_top, topk_indices, action_flow, action_rgb, contrast_pairs,contrast_pairs_r,contrast_pairs_f,
                 actionness1, actionness2, aness_bin1, aness_bin2, all_embeddings) = self.forward_pass(_data)
                
                # Sample Intra Embeddings
                intra_embeddings = all_embeddings['intra_embeddings']
                embedding_targets = self.sample_embeddings(intra_embeddings)
                if not self.initialized:
                    self.initialize(embedding_targets)

                # Snippet Contrastive Learning
                positive_indices, positives, positive_labels = self.get_positives(embedding_targets, self.config.num_segments, self.config.proj_dim)
                negatives, negative_indexes = self.queue.getNegatives(positive_indices)
                # Video contrastive Learning
                vid_positives, vid_positives_indices, distances = self.get_positives_video_distance(intra_embeddings, self.config.num_segments, self.config.proj_dim, self.config.fft_k)

                with torch.no_grad():
                    if(self.config.fft_k <= 1):
                        cas_top_pseudo = self.forward_pass_from_embeddings(vid_positives[0])
                    else:
                        cas_top_pseudo, cas_pseudo = self.forward_pass_with_k_embeddings(vid_positives_indices, distances)


                # calculate pseudo target
                cls_agnostic_gt = self.calculate_pesudo_target(batch_size, _label, topk_indices)
                

                # losses
                cost = self.calculate_all_losses1(contrast_pairs, contrast_pairs_r,contrast_pairs_f, 
                                                  cas_top, _label, action_flow, action_rgb, cls_agnostic_gt, actionness1, actionness2)
                loss_module = self.calculate_module_losses(self.softmax(cas_top), self.softmax(cas_top_pseudo), _data, 
                                                           all_embeddings['decoded_inter'], all_embeddings['decoded_intra'],
                                                                      embedding_targets, positives, negatives)
                cost += loss_module
                self.writter.add_scalar('Loss/Total', cost.cpu().item(), self.step) # Add all losses
                cost.backward()
                self.optimizer.step()

                self.total_loss_per_epoch += cost.cpu().item()
                self.step += 1

                # evaluation
                self.evaluate(epoch=epoch)
            if self.writter:
                self.writter.add_scalar('Loss/Total_Per_Epoch', self.total_loss_per_epoch, epoch)

def save_config(config):
    print('Saving config to {}'.format(config.log_path))
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    # save as txt
    with open(os.path.join(config.log_path, 'config.txt'), 'w') as f:
        for key, value in config.__dict__.items():
            f.write(f'{key}: {value}\n')

def main():
    args = parse_args()
    config = Config(args)
    set_seed(config)

    trainer = ThumosTrainer(config)

    if args.inference_only:
        trainer.test()
    else:
        save_config(config)
        trainer.pretrain_encoding()
        trainer.train()


if __name__ == '__main__':
    main()
