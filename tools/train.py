#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import math
import os
import sys
import time

sys.path.append('../../')

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from tools.log import log_dataset
from tools.sample import sliding_window
from tools.utils import (save_parameters)
from models.FGM import FGM
import torch.nn.functional as F


epsilon = 0.03

class Trainer():
    def __init__(self, model, options):
        self.model_name = options['model_name']
        self.save_dir = options['save_dir']
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.batch_size = options['batch_size']

        self.device = options['device']
        self.lr_step = options['lr_step']
        self.lr_decay_ratio = options['lr_decay_ratio']
        self.accumulation_step = options['accumulation_step']
        self.max_epoch = options['max_epoch']

        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.sample = options['sample']
        self.feature_num = options['feature_num']
        self.data_type = options['data_type']
        self.Event_TF_IDF = options['Event_TF_IDF']
        self.template = options['template']
        self.lmd = options['lmd']
        self.alpha = options['alpha']
        self.options = options

        os.makedirs(self.save_dir, exist_ok=True)

        if self.sample == 'session_window':
            if self.data_type == 'HDFS':
                train_logs, train_labels = sliding_window(self.data_dir, self.model_name, self.window_size,
                                                          datatype='train', data_type="HDFS",
                                                          template_file=self.template, token_attack=0, seq_attack=0,
                                                          attrs_flag=0)
                val_logs, val_labels = sliding_window(self.data_dir, self.model_name, self.window_size,
                                                      datatype='val', data_type="HDFS",
                                                      template_file=self.template, token_attack=0, seq_attack=0,
                                                      attrs_flag=0)

        elif self.sample == 'sliding_window':
            if self.data_type == 'BGL':
                train_logs, train_labels = sliding_window(self.data_dir, self.model_name, self.window_size,
                                                          datatype='train', data_type="BGL",
                                                          template_file=self.template, token_attack=0, seq_attack=0,
                                                          attrs_flag=0)
                val_logs, val_labels = sliding_window(self.data_dir, self.model_name, self.window_size,
                                                      datatype='val', data_type="BGL",
                                                      template_file=self.template, token_attack=0, seq_attack=0,
                                                      attrs_flag=0)
            elif self.data_type == 'Thunderbird':
                train_logs, train_labels = sliding_window(self.data_dir, self.model_name, self.window_size,
                                                          datatype='train', data_type='Thunderbird',
                                                          template_file=self.template, token_attack=0, seq_attack=0,
                                                          attrs_flag=0)
                val_logs, val_labels = sliding_window(self.data_dir, self.model_name, self.window_size,
                                                      datatype='val', data_type='Thunderbird',
                                                      template_file=self.template, token_attack=0, seq_attack=0,
                                                      attrs_flag=0)

        else:
            raise NotImplementedError
        # Format log data into the log_dataset class, which inherits from Dataset
        train_dataset = log_dataset(logs=train_logs,
                                    labels=train_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics)
        valid_dataset = log_dataset(logs=val_logs,
                                    labels=val_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics)

        del train_logs
        del val_logs
        gc.collect()
        # Load the dataset using DataLoader
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       pin_memory=True)

        self.num_train_log = len(train_dataset)
        self.num_valid_log = len(valid_dataset)

        print('Find %d train logs, %d validation logs' %
              (self.num_train_log, self.num_valid_log))
        print('Train batch size %d ,Validation batch size %d' %
              (options['batch_size'], options['batch_size']))
        self.model = model.to(self.device)
        if options['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=options['lr'],
                                             momentum=0.9)
        elif options['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=options['lr'],
                betas=(0.9, 0.999),
            )
        else:
            raise NotImplementedError

        self.start_epoch = 0
        self.best_loss = 1e10
        self.best_score = -1
        save_parameters(options, os.path.join(self.save_dir, "parameters.txt"))
        self.log = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]}
        }
        if options['resume_path'] is not None:
            if os.path.isfile(options['resume_path']):
                self.resume(options['resume_path'], load_optimizer=True)
            else:
                print("Checkpoint not found")

    def resume(self, path, load_optimizer=True):
        print("Resuming from {}".format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.log = checkpoint['log']
        self.model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "log": self.log,
            "best_score": self.best_score
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        save_path = os.path.join(self.save_dir, self.model_name + "_" + suffix + ".pth")
        torch.save(checkpoint, save_path)
        print("Save model checkpoint at {}".format(save_path))

    def save_log(self):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(os.path.join(self.save_dir, key + "_log.csv"),
                                            index=False)
            print("Log saved")
        except:
            print("Failed to save logs")

    intermediate_output = None

    def hook_fn(self, module, input, output):  # Register hook functions
        global intermediate_output
        intermediate_output = output
        return None

    def processExamples(self, tensor, label_, label):
        # Find indices of samples where the label matches the given label_
        indices = torch.nonzero(label == label_, as_tuple=False).squeeze()
        tensor = tensor[indices]  # Extract corresponding samples using the indices
        if tensor.size(0) == 0: return tensor  # Return if no samples with the specified label exist
        if tensor.dim() == 3:
            tensor = tensor.reshape(tensor.size(0), -1)  # Flatten the tensor
        else:
            tensor = tensor.reshape(1, tensor.size(0), tensor.size(1))
            tensor = tensor.reshape(1, -1)
        tensor = F.normalize(tensor.float(), p=2, dim=1)  # Normalize the tensor
        return tensor

    # Here, we treat anomalous samples as positive samples and normal samples as negative samples
    def contrastive_loss(self, ori, per, label):
        t = 0.5  # temperature coefficient,default=0.05, t_BGL=0.3
        ori_with_label_0 = self.processExamples(ori, 0, label)
        if ori_with_label_0.size(0) == 0: return None  # Handle the case where no normal samples exist
        ori_with_label_1 = self.processExamples(ori, 1, label)
        if ori_with_label_1.size(0) == 0: return None  # Handle the case where no anomalous samples exist
        per_with_label_0 = self.processExamples(per, 0, label)
        per_with_label_1 = self.processExamples(per, 1, label)

        loss_ori = 0
        loss_adv = 0
        loss_contra = 0
        # Function to calculate cosine similarity
        cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        for i in range(ori_with_label_1.size(0)):
            sim_pos_posadv = cosine_similarity(ori_with_label_1[i].reshape(1, -1), per_with_label_1[i].reshape(1, -1))
            sim_pos_posadv = torch.exp(sim_pos_posadv / t)
            sim_pos_posadv = torch.sum(sim_pos_posadv)
            if per_with_label_0.size(0) >= 1:
                ori_with_label_1_ = ori_with_label_1[i].unsqueeze(0).expand_as(per_with_label_0)
                sim_pos_negadv = cosine_similarity(ori_with_label_1_, per_with_label_0)
                sim_pos_negadv = torch.exp(sim_pos_negadv / t)
                sim_pos_negadv = torch.sum(sim_pos_negadv)
                per_with_label_1_ = per_with_label_1[i].unsqueeze(0).expand_as(ori_with_label_0)
                sim_posadv_neg = cosine_similarity(per_with_label_1_, ori_with_label_0)
                sim_posadv_neg = torch.exp(sim_posadv_neg / t)
                sim_posadv_neg = torch.sum(sim_posadv_neg)
            loss_contra += torch.log(sim_pos_posadv / (sim_pos_negadv + sim_posadv_neg))
            loss_ori += torch.log(sim_pos_posadv / (sim_pos_posadv + sim_pos_negadv))
            loss_adv += torch.log(sim_pos_posadv / (sim_pos_posadv + sim_posadv_neg))
        loss_contra = -1 * loss_contra / ori_with_label_1.size(0)
        return loss_contra

    def adversarial_train(self, epoch):
        self.log['train']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print("Starting epoch: %d | phase: adversarial train | ⏰: %s | Learning rate: %f" %
              (epoch, start, lr))
        self.log['train']['lr'].append(lr)
        self.log['train']['time'].append(start)
        self.model.train()
        self.optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(self.train_loader, desc="\r")
        num_batch = len(self.train_loader)
        loss_total = 0
        fgm = FGM()
        for i, (log, label) in enumerate(tbar):
            if self.quantitatives:
                features = log['Quantitatives'].clone().detach().to(self.device)
            if self.semantics:
                features = log['Semantics'].clone().detach().to(self.device)
            # Set features as a variable that can compute gradients
            features.requires_grad = True
            output = self.model(features=features, device=self.device)
            # Calculate the loss for the original samples
            loss = criterion(output, label.to(self.device))
            loss /= self.accumulation_step
            loss.backward(retain_graph=True)  # Backpropagate to get the normal gradient
            mask = (features != 0).float() # Shape is (batch_size, seq_len), 1 for non-zero parts, 0 for padding
            fgm.attack(features=features, mask=mask, epsilon=epsilon)  # Add adversarial perturbation to the embeddings
            output2 = self.model(features=features, device=self.device)
            loss_perturbed = criterion(output2, label.to(self.device))
            total_batch_loss = (loss + loss_perturbed) / 2
            loss_total += (float(loss) + float(loss_perturbed)) / 2
            total_batch_loss /= self.accumulation_step
            total_batch_loss.backward()
            if (i + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            tbar.set_description("Train loss: %.5f" % (loss_total / (i + 1)))
        self.log['train']['loss'].append(loss_total / num_batch)

    def consistency_loss(self, ori, per):
        # Compute the Euclidean distance for each position
        euclidean_distance = torch.norm(ori - per, dim=-1)
        mean_distance = euclidean_distance.mean()
        return mean_distance

    def ATCC_adversarial_train(self, epoch):
        self.log['train']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print("Starting epoch: %d | phase: ATCC train | ⏰: %s | Learning rate: %f" %
              (epoch, start, lr))
        self.log['train']['lr'].append(lr)
        self.log['train']['time'].append(start)
        self.model.train()
        self.optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(self.train_loader, desc="\r")
        num_batch = len(self.train_loader)
        loss_total = 0
        fgm = FGM()
        if self.model_name == 'logrobust':
            self.model.rnn.register_forward_hook(self.hook_fn)
        elif self.model_name == 'mlog':
            self.model.identity.register_forward_hook(self.hook_fn)
        for i, (log, label) in enumerate(tbar):
            if self.quantitatives:
                features = log['Quantitatives'].clone().detach().to(self.device)
            if self.semantics:
                features = log['Semantics'].clone().detach().to(self.device)
            # Set features as a variable that can compute gradients
            features.requires_grad = True
            global intermediate_output
            output = self.model(features=features, device=self.device)
            if self.model_name == 'logrobust':
                original_example = intermediate_output[0]  # Obtain the original samples
            elif self.model_name == 'mlog':
                original_example = intermediate_output.unsqueeze(1)
            # Calculate the loss for the original samples
            loss = criterion(output, label.to(self.device))
            loss /= self.accumulation_step
            loss.backward(retain_graph=True)  # Backpropagate to get the normal gradient
            mask = (features != 0).float()   # Shape is (batch_size, seq_len), 1 for non-zero parts, 0 for padding
            fgm.attack(features=features, mask=mask, epsilon=epsilon) # Add adversarial perturbation to the embeddings
            output2 = self.model(features=features, device=self.device)
            loss_perturbed = criterion(output2, label.to(self.device))
            if self.model_name == 'logrobust':
                perturbed_example = intermediate_output[0]  # Get the perturbed example
            elif self.model_name == 'mlog':
                perturbed_example = intermediate_output.unsqueeze(1)

            loss_distance = self.consistency_loss(original_example, perturbed_example)  # compute consistency loss
            loss_contra = self.contrastive_loss(original_example, perturbed_example, label)  # Compute contrastive loss
            if loss_contra:
                total_batch_loss = (1 - self.lmd) * (loss + loss_perturbed) / 2 + self.alpha * loss_contra + \
                                   (self.lmd - self.alpha) * loss_distance
            else:
                print('loss_contra is None')
                total_batch_loss = (loss + loss_perturbed) / 2
            loss_total += float(total_batch_loss)
            total_batch_loss /= self.accumulation_step
            total_batch_loss.backward()
            if (i + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            tbar.set_description("Train loss: %.5f" % (loss_total / (i + 1)))
        self.log['train']['loss'].append(loss_total / num_batch)

    def ordinary_train(self, epoch):
        self.log['train']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print("Starting epoch: %d | phase: ordinary train | ⏰: %s | Learning rate: %f" %
              (epoch, start, lr))
        self.log['train']['lr'].append(lr)
        self.log['train']['time'].append(start)
        self.model.train()
        self.optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(self.train_loader, desc="\r")
        num_batch = len(self.train_loader)
        loss_total = 0
        for i, (log, label) in enumerate(tbar):
            if self.quantitatives:
                features = log['Quantitatives'].clone().detach().to(self.device)
            if self.semantics:
                features = log['Semantics'].clone().detach().to(self.device)
            output = self.model(features=features, device=self.device)
            loss = criterion(output, label.to(self.device))
            loss_total += float(loss)
            loss /= self.accumulation_step
            loss.backward()
            if (i + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            tbar.set_description("Train loss: %.5f" % (loss_total / (i + 1)))
        self.log['train']['loss'].append(loss_total / num_batch)

    def valid(self, epoch):
        self.model.eval()
        self.log['valid']['epoch'].append(epoch)
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log['valid']['lr'].append(lr)
        start = time.strftime("%H:%M:%S")
        print("Starting epoch: %d | phase: valid | ⏰: %s " % (epoch, start))
        self.log['valid']['time'].append(start)
        total_losses = 0
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(self.valid_loader, desc="\r")
        num_batch = len(self.valid_loader)
        label_list = []
        TP, FP, FN, TN = 0, 0, 0, 0
        for i, (log, label) in enumerate(tbar):
            with torch.no_grad():
                if self.quantitatives:
                    features = log['Quantitatives'].clone().detach().to(self.device)
                if self.semantics:
                    features = log['Semantics'].clone().detach().to(self.device)
                output = self.model(features=features, device=self.device)
                loss = criterion(output, label.to(self.device))
                total_losses += float(loss)
                output = F.softmax(output)[:, 0].cpu().detach().numpy()
                predicted = (output < 0.5).astype(int)
                label = np.array([y.cpu() for y in label])
                label_list.extend(label)
                TP += ((predicted == 1) * (label == 1)).sum()
                FP += ((predicted == 1) * (label == 0)).sum()
                FN += ((predicted == 0) * (label == 1)).sum()
                TN += ((predicted == 0) * (label == 0)).sum()
        print("TP:", TP, "FP", FP)
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {},true positive (TP): {},true negative (TN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(FP, FN, TP, TN, P, R, F1))
        print("Validation loss:", total_losses / num_batch)
        self.log['valid']['loss'].append(total_losses / num_batch)

        if total_losses / num_batch < self.best_loss:
            self.best_loss = total_losses / num_batch
            self.save_checkpoint(epoch,
                                 save_optimizer=False,
                                 suffix="bestloss")

    def start_train(self, env_name, is_attack):
        if is_attack == 1 or is_attack == 2:
            print('Re-initialising the optimizer')
            if self.options['optimizer'] == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),  #
                                                 lr=self.options['lr'],
                                                 momentum=0.9)
            elif self.options['optimizer'] == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.options['lr'],
                    betas=(0.9, 0.999),
                )
            else:
                raise NotImplementedError

        for epoch in range(self.start_epoch, self.max_epoch):
            if epoch == 0:
                self.optimizer.param_groups[0]['lr'] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                self.optimizer.param_groups[0]['lr'] *= 2
            if epoch in self.lr_step:
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio
            if is_attack == 1:
                print("====start general adversarial training====")
                self.adversarial_train(epoch)
            elif is_attack == 2:
                print("====start ATCC adversarial training====")
                self.ATCC_adversarial_train(epoch)
            elif is_attack == 0:
                print("====start ordinary training====")
                self.ordinary_train(epoch)

            if epoch >= 0 and epoch % 2 == 0:
                self.valid(epoch)
                self.save_checkpoint(epoch,
                                     save_optimizer=True,
                                     suffix="epoch" + str(epoch))
            self.save_checkpoint(epoch, save_optimizer=True, suffix="last")
            self.save_log()
