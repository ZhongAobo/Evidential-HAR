import torch
import torch.nn as nn
import numpy as np
import itertools    
from trainers.args import args

from models.models import classifier, ReverseLayerF, Discriminator, RandomLayer, Discriminator_CDAN, \
    codats_classifier, AdvSKM_Disc, CNN_ATTN
from models.loss import MMD_loss, CORAL, ConditionalEntropyLoss, VAT, LMMD_loss, HoMM_loss, NTXentLoss, SupConLoss
from utils import EMA
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import torch.nn. functional as F




import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE




def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

coff_list = [0.25, 0.25, 0.5, 1.0]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs, backbone):
        super(Algorithm, self).__init__()
        self.configs = configs

        self.cross_entropy = nn.CrossEntropyLoss()
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)


    # update function is common to all algorithms
    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            
            # training loop 
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())


            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        
        last_model = self.network.state_dict()

        return last_model, best_model
    
    # train loop vary from one method to another
    def training_epoch(self, *args, **kwargs):
        raise NotImplementedError
       

class NO_ADAPT(Algorithm):
    """
    Lower bound: train on source and test on target.
    """
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):
        for src_x, src_y in src_loader:
            
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)
            src_feat = self.feature_extractor(src_x)
            src_pred = []
            for i in range(len(src_feat)):
                src_pred.append(self.classifier(src_feat, i))
            # src_pred = self.classifier(src_feat)

            src_cls_loss = 0.0
            mean_pred = torch.zeros_like(src_pred[-1])
            for i, pred in enumerate(src_pred):
                src_cls_loss += self.cross_entropy(pred, src_y) * coff_list[i]
                mean_pred += pred * 1.0/len(src_pred)
            # src_cls_loss = self.cross_entropy(mean_pred, src_y)
            # src_cls_loss = self.cross_entropy(src_pred[-1], src_y)

            evi_loss = 0.0
            if args.dist == 'dir_dig' or args.dist == 'dir_edl':
                src_y_o = F.one_hot(src_y,num_classes=src_pred[-1].shape[-1]).float()
                # evi_loss += edl_digamma_loss(src_pred, src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)
                evi_loss += edl_digamma_loss(src_pred[-1], src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)
                # evi_loss += edl_digamma_loss(mean_pred, src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)

            loss = src_cls_loss + args.alpha * evi_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Src_cls_loss': src_cls_loss.item(), 'evi_loss': evi_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()
    

# class TARGET_ONLY(Algorithm):
#     """
#     Upper bound: train on target and test on target.
#     """

#     def __init__(self, backbone, configs, hparams, device):
#         super().__init__(configs, backbone)

#         # optimizer and scheduler
#         self.optimizer = torch.optim.Adam(
#             self.network.parameters(),
#             lr=hparams["learning_rate"],
#             weight_decay=hparams["weight_decay"]
#         )
#         self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
#         # hparams
#         self.hparams = hparams
#         # device
#         self.device = device

#     def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

#         for trg_x, trg_y in trg_loader:

#             trg_x, trg_y = trg_x.to(self.device), trg_y.to(self.device)

#             trg_feat = self.feature_extractor(trg_x)
#             trg_pred = self.classifier(trg_feat)

#             trg_cls_loss = self.cross_entropy(trg_pred, trg_y)

#             loss = trg_cls_loss

#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

#             losses = {'Trg_cls_loss': trg_cls_loss.item()}

#             for key, val in losses.items():
#                 avg_meter[key].update(val, 32)

#         self.lr_scheduler.step()


class Deep_Coral(Algorithm):
    """
    Deep Coral: https://arxiv.org/abs/1607.01719
    """
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # correlation alignment loss
        self.coral = CORAL()


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        # add if statement

        if len(src_loader) > len(trg_loader):
            joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        else:
            joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))


        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
            
            # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred = []
            for i in range(len(src_feat)):    #[B,128]
                src_pred.append(self.classifier(src_feat,i))
            
            evi_loss = 0.0
            if args.dist == 'dir_dig' or args.dist == 'dir_edl':
                src_y_o = F.one_hot(src_y,num_classes=src_pred[-1].shape[-1]).float()
                # evi_loss += edl_digamma_loss(src_pred, src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)
                evi_loss += edl_digamma_loss(src_pred[-1], src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)
    
            # extract target features
            trg_feat = self.feature_extractor(trg_x)

            # calculate source classification loss
            src_cls_loss = 0.0
            for i, pred in enumerate(src_pred):
                src_cls_loss += self.cross_entropy(pred, src_y) * coff_list[i]

            trg_feat = self.feature_extractor(trg_x)

            coral_loss = 0.0
            for i in range(len(src_feat)):
                coral_loss += self.coral(src_feat[i], trg_feat[i]) * coff_list[i]

            loss = self.hparams["coral_wt"] * coral_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss + evi_loss * args.alpha

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Total_loss': loss.item(), 'evi_loss': evi_loss.item(), 'Src_cls_loss': src_cls_loss.item(), 'coral_loss': coral_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class MMDA(Algorithm):
    """
    MMDA: https://arxiv.org/abs/1901.00282
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd = MMD_loss()
        self.coral = CORAL()
        self.cond_ent = ConditionalEntropyLoss()


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
            
            # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred = []
            for i in range(len(src_feat)):    #[B,128]
                src_pred.append(self.classifier(src_feat,i))
            
            evi_loss = 0.0
            if args.dist == 'dir_dig' or args.dist == 'dir_edl':
                src_y_o = F.one_hot(src_y,num_classes=src_pred[-1].shape[-1]).float()
                # evi_loss += edl_digamma_loss(src_pred, src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)
                evi_loss += edl_digamma_loss(src_pred[-1], src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)
    
            # extract target features
            trg_feat = self.feature_extractor(trg_x)

            # calculate source classification loss
            src_cls_loss = 0.0
            for i, pred in enumerate(src_pred):
                src_cls_loss += self.cross_entropy(pred, src_y) * coff_list[i]
            # src_cls_loss += self.cross_entropy(src_pred[-1], src_y)
            trg_feat = self.feature_extractor(trg_x)

            coral_loss = 0.0
            mmd_loss = 0.0
            cond_ent_loss = 0.0
            for i in range(len(src_feat)):
                coral_loss += self.coral(src_feat[i], trg_feat[i]) * coff_list[i]
                mmd_loss += self.mmd(src_feat[i], trg_feat[i]) * coff_list[i]
                cond_ent_loss += self.cond_ent(trg_feat[i]) * coff_list[i]

            loss = self.hparams["coral_wt"] * coral_loss + \
                self.hparams["mmd_wt"] * mmd_loss + \
                self.hparams["cond_ent_wt"] * cond_ent_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss + evi_loss * args.alpha

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'evi_loss': evi_loss.item(), 'Coral_loss': coral_loss.item(), 'MMD_loss': mmd_loss.item(),
                    'cond_ent_wt': cond_ent_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
            
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


class DANN(Algorithm):
    """
    DANN: https://arxiv.org/abs/1505.07818
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        
        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # self.lr_scheduler = StepLR(self.optimizer, step_size=args.step_size, gamma=args.lr_decay)
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Domain Discriminator
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):
        # Combine dataloaders
        # Method 1 (min len of both domains)
        # joint_loader = enumerate(zip(src_loader, trg_loader))

        # Method 2 (max len of both domains)
        # joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))


        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()
            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred = []
            for i in range(len(src_feat)):    #[B,128]
                src_pred.append(self.classifier(src_feat,i))
            
            evi_loss = 0.0
            if args.dist == 'dir_dig' or args.dist == 'dir_edl':
                src_y_o = F.one_hot(src_y,num_classes=src_pred[-1].shape[-1]).float()
                # evi_loss += edl_digamma_loss(src_pred, src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)
                evi_loss += edl_digamma_loss(src_pred[-1], src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)
    
            # extract target features
            trg_feat = self.feature_extractor(trg_x)

            # calculate source classification loss
            src_cls_loss = 0.0
            for i, pred in enumerate(src_pred):
                src_cls_loss += self.cross_entropy(pred, src_y) * coff_list[i]
            # src_cls_loss += self.cross_entropy(src_pred[-1], src_y)
            
            trg_feat = self.feature_extractor(trg_x)           

            src_domain_loss = 0.0
            for i, feat in enumerate(src_feat):
                src_feat_reversed = ReverseLayerF.apply(feat, alpha)
                src_domain_pred = self.domain_classifier(src_feat_reversed)
                src_domain_loss += self.cross_entropy(src_domain_pred, domain_label_src.long()) * coff_list[i]

            trg_domain_loss = 0.0
            for i, feat in enumerate(trg_feat):
                trg_feat_reversed = ReverseLayerF.apply(feat, alpha)
                trg_domain_pred = self.domain_classifier(trg_feat_reversed)
                trg_domain_loss += self.cross_entropy(trg_domain_pred, domain_label_trg.long()) * coff_list[i]

            domain_loss = src_domain_loss + trg_domain_loss ###########################

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + evi_loss * args.alpha + \
                self.hparams["domain_loss_wt"] * domain_loss


            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses =  {'Total_loss': loss.item(), 'evi_loss': evi_loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
           
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class CDAN(Algorithm):
    """
    CDAN: https://arxiv.org/abs/1705.10667
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)


        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment Losses
        self.criterion_cond = ConditionalEntropyLoss().to(device)

        self.domain_classifier = Discriminator_CDAN(configs)
        self.random_layer = RandomLayer([configs.features_len * configs.final_out_channels, configs.num_classes],
                                        configs.features_len * configs.final_out_channels)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"])

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            # prepare true domain labels
            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

            # source features and predictions
            src_feat = self.feature_extractor(src_x)
            src_pred = []
            for i in range(len(src_feat)):    #[B,128]
                src_pred.append(self.classifier(src_feat,i))

            evi_loss = 0.0
            if args.dist == 'dir_dig' or args.dist == 'dir_edl':
                src_y_o = F.one_hot(src_y,num_classes=src_pred[-1].shape[-1]).float()
                evi_loss += edl_digamma_loss(src_pred[-1], src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)

            # target features and predictions
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = []
            for i in range(len(trg_feat)):    #[B,128]
                trg_pred.append(self.classifier(trg_feat,i))
                  
            disc_loss = 0.0
            domain_loss = 0.0
            # concatenate features and predictions 最后一个总预测没有计算该loss
            # for i in range(len(src_feat)):    #[B,128]
            #     feat_concat = torch.cat((src_feat[i], trg_feat[i]), dim=0)
            #     pred_concat = torch.cat((src_pred[i], trg_pred[i]), dim=0)

            #     # Domain classification loss
            #     feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1)).detach()
            #     disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
            #     disc_loss += self.cross_entropy(disc_prediction, domain_label_concat) * coff_list[i]
            
            feat_concat = torch.cat((src_feat[-1], trg_feat[-1]), dim=0)
            pred_concat = torch.cat((src_pred[-1], trg_pred[-1]), dim=0)
            # Domain classification loss
            feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1)).detach()
            disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
            disc_loss += self.cross_entropy(disc_prediction, domain_label_concat)


            # update Domain classification
            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()
            
            for i in range(len(src_feat)):    #[B,128]
                # prepare fake domain labels for training the feature extractor
                domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
                domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
                domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

                # Repeat predictions after updating discriminator
                feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1))
                disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
                # loss of domain discriminator according to fake labels

                domain_loss += self.cross_entropy(disc_prediction, domain_label_concat) * coff_list[i]

            # prepare fake domain labels for training the feature extractor
            # domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
            # domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
            # domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

            # Repeat predictions after updating discriminator
            feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1))
            disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
            # loss of domain discriminator according to fake labels

            domain_loss += self.cross_entropy(disc_prediction, domain_label_concat)


            # Task classification  Loss
            src_cls_loss = 0.0
            for i, pred in enumerate(src_pred):
                src_cls_loss += self.cross_entropy(pred, src_y) * coff_list[i]

            # conditional entropy loss.
            loss_trg_cent = 0.0
            # for i, pred in enumerate(trg_pred):
            #     loss_trg_cent += self.criterion_cond(pred) * coff_list[i]
            loss_trg_cent += self.criterion_cond(trg_pred[-1])

            # total loss
            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["cond_ent_wt"] * loss_trg_cent + evi_loss * args.alpha

            # update feature extractor
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'evi_loss': evi_loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                    'cond_ent_loss': loss_trg_cent.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)
        self.lr_scheduler.step()

class DIRT(Algorithm):
    """
    DIRT-T: https://arxiv.org/abs/1802.08735
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.vat_loss = VAT(self.network, device).to(device)
        self.ema = EMA(0.998)
        self.ema.register(self.network)

        # Discriminator
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
       
    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            # prepare true domain labels
            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

            src_feat = self.feature_extractor(src_x)
            src_pred = []
            for i in range(len(src_feat)):    #[B,128]
                src_pred.append(self.classifier(src_feat,i))

            # target features and predictions
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = []
            for i in range(len(trg_feat)):    #[B,128]
                trg_pred.append(self.classifier(trg_feat,i))

            # concatenate features and predictions
            domain_loss = 0.0
            disc_loss = 0.0
            for i in range(len(src_feat)):
                feat_concat = torch.cat((src_feat[i], trg_feat[i]), dim=0)

                # Domain classification loss
                disc_prediction = self.domain_classifier(feat_concat.detach())
                disc_loss += self.cross_entropy(disc_prediction, domain_label_concat) * coff_list[i]

            # update Domain classification
            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()

            for i in range(len(src_feat)):
                # prepare fake domain labels for training the feature extractor
                domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
                domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
                domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

                # Repeat predictions after updating discriminator
                disc_prediction = self.domain_classifier(feat_concat)

            # loss of domain discriminator according to fake labels
                domain_loss += self.cross_entropy(disc_prediction, domain_label_concat) * coff_list[i]

            evi_loss = 0.0
            if args.dist == 'dir_dig' or args.dist == 'dir_edl':
                src_y_o = F.one_hot(src_y,num_classes=src_pred[-1].shape[-1]).float()
                evi_loss += edl_digamma_loss(src_pred[-1], src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)

            # Task classification  Loss
            src_cls_loss = 0.0
            for i, pred in enumerate(src_pred):
                src_cls_loss += self.cross_entropy(pred, src_y) * coff_list[i]

            # conditional entropy loss.
            
            #loss_trg_cent = 0.0
            #for i, pred in enumerate(trg_pred):
            #    loss_trg_cent += self.criterion_cond(pred) * coff_list[i]
            loss_trg_cent = self.criterion_cond(trg_pred[-1])

            # Virual advariarial training loss
            loss_src_vat = 0.0
            loss_trg_vat = 0.0
            # for i in range(len(src_pred)):
                # loss_src_vat += self.vat_loss(src_x, src_pred[i]) * coff_list[i]
                # loss_trg_vat += self.vat_loss(trg_x, trg_pred[i]) * coff_list[i]
            loss_src_vat += self.vat_loss(src_x, src_pred[-1])
            loss_trg_vat += self.vat_loss(trg_x, trg_pred[-1])
            total_vat = loss_src_vat + loss_trg_vat
            # total loss
            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["cond_ent_wt"] * loss_trg_cent + self.hparams["vat_loss_wt"] * total_vat + evi_loss * args.alpha

            # loss = args.a * src_cls_loss + args.b * domain_loss + \
            #     args.c * loss_trg_cent + args.d * total_vat + evi_loss * args.alpha

            # update exponential moving average
            self.ema(self.network)

            # update feature extractor
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'evi_loss': evi_loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                    'cond_ent_loss': loss_trg_cent.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()
###############################################################################
class DSAN(Algorithm):
    """
    DSAN: https://ieeexplore.ieee.org/document/9085896
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Alignment losses
        self.loss_LMMD = LMMD_loss(device=device, class_num=configs.num_classes).to(device)

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)        # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred = []
            for i in range(len(src_feat)):    #[B,128]
                src_pred.append(self.classifier(src_feat,i))

            # extract target features
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = []
            for i in range(len(trg_feat)):    #[B,128]
                trg_pred.append(self.classifier(trg_feat,i))

            # calculate lmmd loss
            domain_loss = 0.0
            for i in range(len(src_feat)):
                domain_loss += self.loss_LMMD.get_loss(src_feat[i], trg_feat[i], src_y, torch.nn.functional.softmax(trg_pred[i], dim=1)) * coff_list[i]

            
            evi_loss = 0.0
            if args.dist == 'dir_dig' or args.dist == 'dir_edl':
                src_y_o = F.one_hot(src_y,num_classes=src_pred[-1].shape[-1]).float()
                evi_loss += edl_digamma_loss(src_pred[-1], src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)

            # calculate source classification loss
            src_cls_loss = 0.0
            for i in range(len(src_pred)):
                src_cls_loss += self.cross_entropy(src_pred[i], src_y) * coff_list[i]

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss + evi_loss * args.alpha

            # loss = args.b * domain_loss + \
            #     args.a * src_cls_loss + evi_loss * args.alpha

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'evi_loss': evi_loss.item(), 'LMMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class HoMM(Algorithm):
    """
    HoMM: https://arxiv.org/pdf/1912.11976.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # aligment losses
        self.coral = CORAL()
        self.HoMM_loss = HoMM_loss()

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
            
            src_feat = self.feature_extractor(src_x)
            src_pred = []
            for i in range(len(src_feat)):    #[B,128]
                src_pred.append(self.classifier(src_feat,i))

            evi_loss = 0.0
            if args.dist == 'dir_dig' or args.dist == 'dir_edl':
                src_y_o = F.one_hot(src_y,num_classes=src_pred[-1].shape[-1]).float()
                evi_loss += edl_digamma_loss(src_pred[-1], src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)


            # extract target features
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = []
            for i in range(len(trg_feat)):    #[B,128]
                trg_pred.append(self.classifier(trg_feat,i))

            # calculate source classification loss
            src_cls_loss = 0.0
            for i in range(len(src_pred)):
                src_cls_loss += self.cross_entropy(src_pred[i], src_y) * coff_list[i]

            # calculate lmmd loss
            for i in range(len(src_feat)):
                domain_loss = self.HoMM_loss(src_feat[i], trg_feat[i]) * coff_list[i]

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss + evi_loss * args.alpha

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'evi_loss': evi_loss.item(), 'HoMM_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
            
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


class DDC(Algorithm):
    """
    DDC: https://arxiv.org/abs/1412.3474
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd_loss = MMD_loss()

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
            
            # extract source features
            src_feat = self.feature_extractor(src_x)
            
            src_pred = []
            for i in range(len(src_feat)):    #[B,128]
                src_pred.append(self.classifier(src_feat,i))
            
            evi_loss = 0.0
            if args.dist == 'dir_dig' or args.dist == 'dir_edl':
                src_y_o = F.one_hot(src_y,num_classes=src_pred[-1].shape[-1]).float()
                # evi_loss += edl_digamma_loss(src_pred, src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)
                evi_loss += edl_digamma_loss(src_pred[-1], src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)
    
            # extract target features
            trg_feat = self.feature_extractor(trg_x)

            # calculate source classification loss
            src_cls_loss = 0.0
            for i, pred in enumerate(src_pred):
                src_cls_loss += self.cross_entropy(pred, src_y) * coff_list[i]
            # src_cls_loss += self.cross_entropy(src_pred[-1], src_y)

            # calculate mmd loss
            domain_loss = 0.0
            for i, f in enumerate(src_feat):
                domain_loss += self.mmd_loss(src_feat[i], trg_feat[i]) * coff_list[i]

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss + evi_loss * args.alpha

            # print(loss)

            # loss = 1 * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'evi_loss': evi_loss.item(), 'MMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


            for key, val in losses.items():
                avg_meter[key].update(val, 32) 

        self.lr_scheduler.step()




class CoDATS(Algorithm):
    """
    CoDATS: https://arxiv.org/pdf/2005.10996.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # we replace the original classifier with codats the classifier
        # remember to use same name of self.classifier, as we use it for the model evaluation
        self.classifier = codats_classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


        # Domain classifier
        self.domain_classifier = Discriminator(configs)

        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
        
            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = []
            for i in range(len(src_feat)):    #[B,128]
                src_pred.append(self.classifier(src_feat,i))

            evi_loss = 0.0
            if args.dist == 'dir_dig' or args.dist == 'dir_edl':
                src_y_o = F.one_hot(src_y,num_classes=src_pred[-1].shape[-1]).float()
                evi_loss += edl_digamma_loss(src_pred[-1], src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)
                
            trg_feat = self.feature_extractor(trg_x)

            # Task classification  Loss
            src_cls_loss = 0.0
            for i, pred in enumerate(src_pred):
                src_cls_loss += self.cross_entropy(pred, src_y) * coff_list[i]

            # Domain classification loss
            # source
            src_domain_loss = 0.0
            for i in range(len(src_feat)):
                src_feat_reversed = ReverseLayerF.apply(src_feat[i], alpha)
                src_domain_pred = self.domain_classifier(src_feat_reversed)
                src_domain_loss += self.cross_entropy(src_domain_pred, domain_label_src.long()) * coff_list[i]

            # target
            trg_domain_loss = 0.0
            for i in range(len(trg_feat)):
                trg_feat_reversed = ReverseLayerF.apply(trg_feat[i], alpha)
                trg_domain_pred = self.domain_classifier(trg_feat_reversed)
                trg_domain_loss += self.cross_entropy(trg_domain_pred, domain_label_trg.long()) * coff_list[i]

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                self.hparams["domain_loss_wt"] * domain_loss +evi_loss * args.alpha

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses =  {'Total_loss': loss.item(), 'evi_loss': evi_loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class AdvSKM(Algorithm):
    """
    AdvSKM: https://www.ijcai.org/proceedings/2021/0378.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd_loss = MMD_loss()
        self.AdvSKM_embedder = AdvSKM_Disc(configs).to(device)
        self.optimizer_disc = torch.optim.Adam(
            self.AdvSKM_embedder.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features
            
            src_feat = self.feature_extractor(src_x)
            src_pred = []
            for i in range(len(src_feat)):    #[B,128]
                src_pred.append(self.classifier(src_feat,i))

            evi_loss = 0.0
            if args.dist == 'dir_dig' or args.dist == 'dir_edl':
                src_y_o = F.one_hot(src_y,num_classes=src_pred[-1].shape[-1]).float()
                evi_loss += edl_digamma_loss(src_pred[-1], src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)

            mmd_loss = 0.0
            for i in range(len(src_feat)):
                source_embedding_disc = self.AdvSKM_embedder(src_feat[i].detach())
                target_embedding_disc = self.AdvSKM_embedder(trg_feat[i].detach())
                mmd_loss -= self.mmd_loss(source_embedding_disc, target_embedding_disc) * coff_list[i]
            
            mmd_loss.requires_grad = True

            # update discriminator
            self.optimizer_disc.zero_grad()
            mmd_loss.backward()
            self.optimizer_disc.step()

            # calculate source classification loss
            src_cls_loss = 0.0
            for i, pred in enumerate(src_pred):
                src_cls_loss += self.cross_entropy(pred, src_y) * coff_list[i]

            # domain loss.
            mmd_loss_adv = 0.0
            for i in range(len(src_feat)):
                source_embedding_disc = self.AdvSKM_embedder(src_feat[i])
                target_embedding_disc = self.AdvSKM_embedder(trg_feat[i])
                mmd_loss_adv += self.mmd_loss(source_embedding_disc, target_embedding_disc)
            
            mmd_loss_adv.requires_grad = True

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * mmd_loss_adv + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss + evi_loss*args.alpha

            # update optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'MMD_loss': mmd_loss_adv.item(), 'evi_loss': evi_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
            for key, val in losses.items():
                    avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class SASA(Algorithm):
    
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # feature_length for classifier
        configs.features_len = 1
        self.classifier = classifier(configs)
        # feature length for feature extractor
        configs.features_len = 1
        self.feature_extractor = CNN_ATTN(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features

            # Extract features
            src_feat = self.feature_extractor(src_x)
            trg_feat = self.feature_extractor(trg_x)            

            # source classification loss
            src_pred = []
            for i in range(len(src_feat)):    #[B,128]
                src_pred.append(self.classifier(src_feat,i))

            src_cls_loss = 0.0
            for i, pred in enumerate(src_pred):
                src_cls_loss += self.cross_entropy(pred, src_y) * coff_list[i]

            evi_loss = 0.0
            if args.dist == 'dir_dig' or args.dist == 'dir_edl':
                src_y_o = F.one_hot(src_y,num_classes=src_pred[-1].shape[-1]).float()
                evi_loss += edl_digamma_loss(src_pred[-1], src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)

            # MMD loss
            domain_loss_intra = 0.0
            for i in range(len(src_feat)):
                domain_loss_intra += self.mmd_loss(src_struct=src_feat[i],
                                            tgt_struct=trg_feat[i], weight=self.hparams['domain_loss_wt']) * coff_list[i]

            # total loss
            total_loss = self.hparams['src_cls_loss_wt'] * src_cls_loss + domain_loss_intra + evi_loss * args.alpha

            # remove old gradients
            self.optimizer.zero_grad()
            # calculate gradients
            total_loss.backward()
            # update the weights
            self.optimizer.step()

            losses =  {'Total_loss': total_loss.item(), 'MMD_loss': domain_loss_intra.item(), 'evi_loss': evi_loss.item(),
                    'Src_cls_loss': src_cls_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()
    def mmd_loss(self, src_struct, tgt_struct, weight):
        delta = torch.mean(src_struct - tgt_struct, dim=-2)
        loss_value = torch.norm(delta, 2) * weight
        return loss_value


# class CoTMix(Algorithm):
#     def __init__(self, backbone, configs, hparams, device):
#         super().__init__(configs, backbone)

#          # optimizer and scheduler
#         self.optimizer = torch.optim.Adam(
#             self.network.parameters(),
#             lr=hparams["learning_rate"],
#             weight_decay=hparams["weight_decay"]
#         )
#         self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
#         # hparams
#         self.hparams = hparams
#         # device
#         self.device = device

#         # Aligment losses
#         self.contrastive_loss = NTXentLoss(device, hparams["batch_size"], 0.2, True)
#         self.entropy_loss = ConditionalEntropyLoss()
#         self.sup_contrastive_loss = SupConLoss(device)

#     def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

#         # Construct Joint Loaders 
#         joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
#         for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
#             src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features

#             # ====== Temporal Mixup =====================
#             src_dominant, trg_dominant = self.temporal_mixup(src_x, trg_x)

#             # ====== Source =====================
#             self.optimizer.zero_grad()

#             # Src original features
#             src_orig_feat = self.feature_extractor(src_x)
#             src_orig_logits = self.classifier(src_orig_feat)

#             evi_loss = 0.0
#             src_pred = src_orig_logits
#             if args.dist == 'nig':
#                 src_y_o = F.one_hot(src_y,num_classes=src_pred[-1].shape[-1]).float()
#                 src_pred,src_pred_v,src_pred_alpha,src_pred_beta = split(src_pred,src_pred,src_pred,src_pred)
#                 evi_loss += torch.mean(criterion_nig(src_pred,src_pred_v,src_pred_alpha,src_pred_beta,src_y_o,0.01))

#             elif args.dist == 'dir_dig' or args.dist == 'dir_edl':
#                 src_y_o = F.one_hot(src_y,num_classes=src_pred[-1].shape[-1]).float()
#                 evi_loss += edl_digamma_loss(src_pred, src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)
                

#             elif args.dist == 'dir_tmc':
#                 src_pred_alpha = src_pred + 1
#                 src_y_o = F.one_hot(src_y,num_classes=src_pred[-1].shape[-1]).float()
#                 evi_loss += kl_loss(src_y_o, src_pred_alpha, src_pred.shape[-1], epoch+1, 10.0)  
            
#             elif args.dist == 'dir_mse':
#                 src_y_o = F.one_hot(src_y,num_classes=src_pred[-1].shape[-1]).float()
#                 evi_loss += edl_mse_loss(src_pred, src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)

#             elif args.dist == 'dir_log':
#                 src_y_o = F.one_hot(src_y,num_classes=src_pred[-1].shape[-1]).float()
#                 evi_loss += edl_log_loss(src_pred, src_y_o, epoch+1, src_pred[-1].shape[-1], 10.0)

#             # Target original features
#             trg_orig_feat = self.feature_extractor(trg_x)
#             trg_orig_logits = self.classifier(trg_orig_feat)

#             # -----------  The two main losses
#             # Cross-Entropy loss
#             src_cls_loss = self.cross_entropy(src_orig_logits, src_y)
#             loss = src_cls_loss * round(self.hparams["src_cls_weight"], 2) + args.alpha * evi_loss

#             # Target Entropy loss
#             trg_entropy_loss = self.entropy_loss(trg_orig_logits)
#             loss += trg_entropy_loss * round(self.hparams["trg_entropy_weight"], 2)

#             # -----------  Auxiliary losses
#             # Extract source-dominant mixup features.
#             src_dominant_feat = self.feature_extractor(src_dominant)
#             src_dominant_logits = self.classifier(src_dominant_feat)

#             # supervised contrastive loss on source domain side
#             src_concat = torch.cat([src_orig_logits.unsqueeze(1), src_dominant_logits.unsqueeze(1)], dim=1)
#             src_supcon_loss = self.sup_contrastive_loss(src_concat, src_y)
#             loss += src_supcon_loss * round(self.hparams["src_supCon_weight"], 2)

#             # Extract target-dominant mixup features.
#             trg_dominant_feat = self.feature_extractor(trg_dominant)
#             trg_dominant_logits = self.classifier(trg_dominant_feat)

#             # Unsupervised contrastive loss on target domain side
#             trg_con_loss = self.contrastive_loss(trg_orig_logits, trg_dominant_logits)
#             loss += trg_con_loss * round(self.hparams["trg_cont_weight"], 2)

#             loss.backward()
#             self.optimizer.step()

#             losses =  {'Total_loss': loss.item(),
#                     'src_cls_loss': src_cls_loss.item(),
#                     'trg_entropy_loss': trg_entropy_loss.item(),
#                     'src_supcon_loss': src_supcon_loss.item(),
#                     'trg_con_loss': trg_con_loss.item()
#                     }
#             for key, val in losses.items():
#                 avg_meter[key].update(val, 32)

#         self.lr_scheduler.step()           

#     def temporal_mixup(self,src_x, trg_x):
        
#         mix_ratio = round(self.hparams["mix_ratio"], 2)
#         temporal_shift = self.hparams["temporal_shift"]
#         h = temporal_shift // 2  # half

#         src_dominant = mix_ratio * src_x + (1 - mix_ratio) * \
#                     torch.mean(torch.stack([torch.roll(trg_x, -i, 2) for i in range(-h, h)], 2), 2)

#         trg_dominant = mix_ratio * trg_x + (1 - mix_ratio) * \
#                     torch.mean(torch.stack([torch.roll(src_x, -i, 2) for i in range(-h, h)], 2), 2)
        
#         return src_dominant, trg_dominant
    


# # Untied Approaches: (MCD)
# class MCD(Algorithm):
#     """
#     Maximum Classifier Discrepancy for Unsupervised Domain Adaptation
#     MCD: https://arxiv.org/pdf/1712.02560.pdf
#     """

#     def __init__(self, backbone, configs, hparams, device):
#         super().__init__(configs, backbone)

#         self.feature_extractor = backbone(configs)
#         self.classifier = classifier(configs)
#         self.classifier2 = classifier(configs)

#         self.network = nn.Sequential(self.feature_extractor, self.classifier)


#         # optimizer and scheduler
#         self.optimizer_fe = torch.optim.Adam(
#             self.feature_extractor.parameters(),
#             lr=hparams["learning_rate"],
#             weight_decay=hparams["weight_decay"]
#         )
#                 # optimizer and scheduler
#         self.optimizer_c1 = torch.optim.Adam(
#             self.classifier.parameters(),
#             lr=hparams["learning_rate"],
#             weight_decay=hparams["weight_decay"]
#         )
#                 # optimizer and scheduler
#         self.optimizer_c2 = torch.optim.Adam(
#             self.classifier2.parameters(),
#             lr=hparams["learning_rate"],
#             weight_decay=hparams["weight_decay"]
#         )

#         self.lr_scheduler_fe = StepLR(self.optimizer_fe, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
#         self.lr_scheduler_c1 = StepLR(self.optimizer_c1, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
#         self.lr_scheduler_c2 = StepLR(self.optimizer_c2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

#         # hparams
#         self.hparams = hparams
#         # device
#         self.device = device

#         # Aligment losses
#         self.mmd_loss = MMD_loss()

#     def update(self, src_loader, trg_loader, avg_meter, logger):
#         # defining best and last model
#         best_src_risk = float('inf')
#         best_model = None

#         for epoch in range(1, self.hparams["num_epochs"] + 1):
            
#             # source pretraining loop 
#             self.pretrain_epoch(src_loader, avg_meter)

#             # training loop 
#             self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

#             # saving the best model based on src risk
#             if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
#                 best_src_risk = avg_meter['Src_cls_loss'].avg
#                 best_model = deepcopy(self.network.state_dict())


#             logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
#             for key, val in avg_meter.items():
#                 logger.debug(f'{key}\t: {val.avg:2.4f}')
#             logger.debug(f'-------------------------------------')
        
#         last_model = self.network.state_dict()

#         return last_model, best_model

#     def pretrain_epoch(self, src_loader,avg_meter):
#         for src_x, src_y in src_loader:
#             src_x, src_y = src_x.to(self.device), src_y.to(self.device)
          
#             src_feat = self.feature_extractor(src_x)
#             src_pred1 = self.classifier(src_feat)
#             src_pred2 = self.classifier2(src_feat)

#             src_cls_loss1 = self.cross_entropy(src_pred1, src_y)
#             src_cls_loss2 = self.cross_entropy(src_pred2, src_y)

#             loss = src_cls_loss1 + src_cls_loss2

#             self.optimizer_c1.zero_grad()
#             self.optimizer_c2.zero_grad()
#             self.optimizer_fe.zero_grad()

#             loss.backward()

#             self.optimizer_c1.step()
#             self.optimizer_c2.step()
#             self.optimizer_fe.step()

            
#             losses = {'Src_cls_loss': loss.item()}

#             for key, val in losses.items():
#                 avg_meter[key].update(val, 32)

#     def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

#         # Construct Joint Loaders 
#         joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

#         for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
#             src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
            

#             # extract source features
#             src_feat = self.feature_extractor(src_x)
#             src_pred1 = self.classifier(src_feat)
#             src_pred2 = self.classifier2(src_feat)

#             evi_loss = 0.0
#             if args.dist == 'nig':
#                 src_y_o = F.one_hot(src_y,num_classes=src_pred1.shape[-1]).float()

#                 src_pred1,src_pred1_v,src_pred1_alpha,src_pred1_beta = split(src_pred1,src_pred1,src_pred1,src_pred1)
#                 evi_loss += torch.mean(criterion_nig(src_pred1,src_pred1_v,src_pred1_alpha,src_pred1_beta,src_y_o,0.01))

#                 src_pred2,src_pred2_v,src_pred2_alpha,src_pred2_beta = split(src_pred2,src_pred2,src_pred2,src_pred2)
#                 evi_loss += torch.mean(criterion_nig(src_pred2,src_pred2_v,src_pred2_alpha,src_pred2_beta,src_y_o,0.01))

#             elif args.dist == 'dir_dig' or args.dist == 'dir_edl':
#                 src_y_o = F.one_hot(src_y,num_classes=src_pred1.shape[-1]).float()
#                 evi_loss += edl_digamma_loss(src_pred1, src_y_o, epoch+1, src_pred1.shape[-1], 10.0)
#                 evi_loss += edl_digamma_loss(src_pred2, src_y_o, epoch+1, src_pred1.shape[-1], 10.0)
#                 src_pred1 = F.softplus(src_pred1) + 1
#                 src_pred2 = F.softplus(src_pred2) + 1


#             elif args.dist == 'dir_tmc':
#                 src_pred1_alpha = src_pred1 + 1
#                 src_pred1_alpha = src_pred2 + 1
#                 src_y_o = F.one_hot(src_y,num_classes=src_pred1.shape[-1]).float()
#                 evi_loss += kl_loss(src_y_o, src_pred1_alpha, src_pred1.shape[-1], epoch+1, 10.0)  
#                 evi_loss += kl_loss(src_y_o, src_pred2_alpha, src_pred1.shape[-1], epoch+1, 10.0)  

            
#             elif args.dist == 'dir_mse':
#                 src_y_o = F.one_hot(src_y,num_classes=src_pred1.shape[-1]).float()
#                 evi_loss += edl_mse_loss(src_pred1, src_y_o, epoch+1, src_pred1.shape[-1], 10.0)
#                 evi_loss += edl_mse_loss(src_pred2, src_y_o, epoch+1, src_pred1.shape[-1], 10.0)

#             elif args.dist == 'dir_log':
#                 src_y_o = F.one_hot(src_y,num_classes=src_pred1.shape[-1]).float()
#                 evi_loss += edl_log_loss(src_pred1, src_y_o, epoch+1, src_pred1.shape[-1], 10.0)
#                 evi_loss += edl_log_loss(src_pred2, src_y_o, epoch+1, src_pred1.shape[-1], 10.0)
            

#             # source losses
#             src_cls_loss1 = self.cross_entropy(src_pred1, src_y)
#             src_cls_loss2 = self.cross_entropy(src_pred2, src_y)
#             loss_s = src_cls_loss1 + src_cls_loss2
            

#             # Freeze the feature extractor
#             for k, v in self.feature_extractor.named_parameters():
#                 v.requires_grad = False
#             # update C1 and C2 to maximize their difference on target sample
#             trg_feat = self.feature_extractor(trg_x) 
#             trg_pred1 = self.classifier(trg_feat.detach())
#             trg_pred2 = self.classifier2(trg_feat.detach())


#             loss_dis = self.discrepancy(trg_pred1, trg_pred2)

#             loss = loss_s - loss_dis + evi_loss * args.alpha
            
#             loss.backward()
#             self.optimizer_c1.step()
#             self.optimizer_c2.step()

#             self.optimizer_c1.zero_grad()
#             self.optimizer_c2.zero_grad()
#             self.optimizer_fe.zero_grad()

#             # Freeze the classifiers
#             for k, v in self.classifier.named_parameters():
#                 v.requires_grad = False
#             for k, v in self.classifier2.named_parameters():
#                 v.requires_grad = False
#                         # Freeze the feature extractor
#             for k, v in self.feature_extractor.named_parameters():
#                 v.requires_grad = True
#             # update feature extractor to minimize the discrepaqncy on target samples
#             trg_feat = self.feature_extractor(trg_x)        
#             trg_pred1 = self.classifier(trg_feat)
#             trg_pred2 = self.classifier2(trg_feat)


#             loss_dis_t = self.discrepancy(trg_pred1, trg_pred2)
#             domain_loss = self.hparams["domain_loss_wt"] * loss_dis_t 

#             domain_loss.backward()
#             self.optimizer_fe.step()

#             self.optimizer_fe.zero_grad()
#             self.optimizer_c1.zero_grad()
#             self.optimizer_c2.zero_grad()


#             losses =  {'Total_loss': loss.item(), 'MMD_loss': domain_loss.item()}

#             for key, val in losses.items():
#                 avg_meter[key].update(val, 32)

#         self.lr_scheduler_fe.step()
#         self.lr_scheduler_c1.step()
#         self.lr_scheduler_c2.step()

#     def discrepancy(self, out1, out2):

#         return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))





def evidence(x):
    return F.softplus(x)
def split(mu, logv, logalpha, logbeta):
    v = evidence(logv)
    alpha = evidence(logalpha) + 1
    beta = evidence(logbeta)
    return mu, v, alpha, beta
def criterion_nig(u, la, alpha, beta, y, risk):#u->δ la->γ alpha->α beta->β
    # our loss function
    om = 2 * beta * (1 + la)
    loss = sum(            #Eq.5 L_NLL(w)
        0.5 * torch.log(np.pi / la) - alpha * torch.log(om) + (alpha + 0.5) * torch.log(la * (u - y) ** 2 + om)) / u.shape[0]
    lossr = risk * sum(torch.abs(u - y) * (2 * la + alpha)) / u.shape[0]
    loss = loss + lossr    #Eq.6
    return loss




def kl_divergence(alpha, num_classes):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=alpha.device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl
def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step):
    S = torch.sum(alpha, dim=-1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=-1, keepdim=True)    #(4)
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return A + kl_div

def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step
):
    if isinstance(output, list):
        # coff = [0.5, 1.0]
        # coff = [0.25, 0.5, 1.0]
        loss = torch.zeros(1).float().to(target.device)
        for i, o in enumerate(output):
            evidence = F.softplus(o)
            alpha = evidence + 1    #[B,H,W,cls]
            loss += torch.mean(
                edl_loss(
                    torch.digamma, target, alpha, epoch_num, num_classes, annealing_step
                )
            )
    else:
        evidence = F.softplus(output)
        alpha = evidence + 1    #[B,H,W,cls]
        loss = torch.mean(
            edl_loss(
                torch.digamma, target, alpha, epoch_num, num_classes, annealing_step
            )
        )
    return loss

def loglikelihood_loss(y, alpha):
    S = torch.sum(alpha, dim=-1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=-1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=-1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood
def mse_loss(y, alpha, epoch_num, num_classes, annealing_step):
    loglikelihood = loglikelihood_loss(y, alpha)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return loglikelihood + kl_div
def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step):
    evidence = F.softplus(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step)
    )
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step):
    evidence = F.softplus(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step
        )
    )
    return loss













def kl_loss(label, alpha, c, global_step, annealing_step):
    E = alpha - 1
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1   #[B, C, H, W]
    kl_loss = annealing_coef * KL(alp, c)
    return torch.mean(kl_loss)

def KL(alpha, c):
    beta = torch.ones_like(alpha)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True) #[B,1,H,W]
    S_beta = torch.sum(beta, dim=1, keepdim=True)   #[B,1,H,W]
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl



def show_evidence(outputs, labels):
    num_classes = outputs.shape[-1]
    _, preds = torch.max(outputs, 1)    #[B]

    match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
    

    evidence = F.relu(outputs)
    alpha = evidence + 1
    u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
    total_evidence = torch.sum(evidence, 1, keepdim=True)
    mean_evidence = torch.mean(total_evidence)
    mean_evidence_succ = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * match
                        ) / torch.sum(match + 1e-20)
    mean_evidence_fail = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * (1 - match)
                        ) / (torch.sum(torch.abs(1 - match)) + 1e-20)
    
    return mean_evidence, mean_evidence_succ, mean_evidence_fail, u


def ECE(pred, true_labels, M=10):
    pred = F.softmax(pred, dim=1)

    confidences = torch.max(pred, axis=1)[0].detach().cpu().numpy()
    predicted_label = torch.argmax(pred, dim=1).cpu().numpy()
    true_labels = true_labels.detach().cpu().numpy()

    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = predicted_label==true_labels
    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        prop_in_bin = in_bin.astype(float).mean()
        
        # print(f'({bin_lower},{bin_upper}]:')
        accuracy_in_bin = accuracies[in_bin].astype(float).mean()
        avg_confidence_in_bin = confidences[in_bin].mean()
        print(accuracy_in_bin,prop_in_bin,avg_confidence_in_bin)

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece
