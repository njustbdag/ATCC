#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
import sys

sys.path.append('../../')
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from tools.log import log_dataset
from tools.sample import sliding_window
from sklearn.metrics import confusion_matrix
from captum.attr import IntegratedGradients


class Predicter():
    def __init__(self, model, options):
        self.model_name = options['model_name']
        self.data_dir = options['data_dir']
        self.device = options['device']
        self.model = model
        self.model_path = options['model_path']
        self.window_size = options['window_size']
        self.num_candidates = options['num_candidates']
        self.num_classes = options['num_classes']
        self.input_size = options['input_size']
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.batch_size = options['batch_size']
        self.data_type = options['data_type']
        self.Event_TF_IDF = options['Event_TF_IDF']
        self.template = options['template']
        self.ig = IntegratedGradients(model)

    def read_json(filename):
        with open(filename, 'r') as load_f:
            file_dict = json.load(load_f)
        return file_dict

    def predict_supervised(self, token_attack, seq_attack, attrs_flag):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        label_list = []
        output_list = []
        n_steps = 50
        attributions_output = []
        print('model_path: {}'.format(self.model_path))
        if self.data_type == 'HDFS':
            test_logs, test_labels = sliding_window(self.data_dir, self.model_name, self.window_size,
                                                    datatype='predict', data_type='HDFS',
                                                    template_file=self.template, token_attack=token_attack,
                                                    seq_attack=seq_attack, attrs_flag=attrs_flag)
        elif self.data_type == 'BGL':
            test_logs, test_labels = sliding_window(self.data_dir, self.model_name, self.window_size,
                                                    datatype='predict', data_type='BGL',
                                                    template_file=self.template, token_attack=token_attack,
                                                    seq_attack=seq_attack, attrs_flag=attrs_flag)
        elif self.data_type == 'Thunderbird':
            test_logs, test_labels = sliding_window(self.data_dir, self.model_name, self.window_size,
                                                    datatype='predict', data_type='Thunderbird',
                                                    template_file=self.template, token_attack=token_attack,
                                                    seq_attack=seq_attack, attrs_flag=attrs_flag)
        test_dataset = log_dataset(logs=test_logs,
                                   labels=test_labels,
                                   seq=self.sequentials,
                                   quan=self.quantitatives,
                                   sem=self.semantics)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      pin_memory=True)
        del test_logs
        tbar = tqdm(self.test_loader, desc="\r")
        TP, FP, FN, TN = 0, 0, 0, 0
        for i, (log, label) in enumerate(tbar):
            if self.quantitatives:
                features = log['Quantitatives'].clone().detach().to(self.device)
            if self.semantics:
                features = log['Semantics'].clone().detach().to(self.device)
            output = self.model(features=features, device=self.device)
            dims = output.dim()
            if dims == 1:
                output_softmax = F.softmax(output)
                output = output_softmax.cpu().detach().numpy()
            else:
                output_softmax = F.softmax(output, 1)
                output = output_softmax[:, 0].cpu().detach().numpy()

            threshold = 0.5
            predicted = (output < threshold).astype(int)
            label = np.array([y.cpu() for y in label])
            label_list.extend(label)
            output_list.extend(np.array([(1 - y) for y in output]))
            tn, fp, fn, tp = confusion_matrix(label, predicted, labels=[0, 1]).ravel()
            TN += tn
            FP += fp
            FN += fn
            TP += tp

            if attrs_flag != 0:
                with torch.backends.cudnn.flags(enabled=False):
                    target = torch.tensor(label, dtype=torch.int64).to(self.device)
                    attributions, delta = self.ig.attribute(features, n_steps=n_steps, target=target,
                                                            return_convergence_delta=True)
                    attributions = attributions.cpu().detach().numpy()
                # Obtain positions where label and predicted are the same, and save the values of attrs at the
                # corresponding positions
                pos = np.where(((label == 1) & (predicted == 1)) | ((label == 0) & (predicted == 0)))
                for idx in range(features.shape[0]):
                    if idx not in pos[0]:
                        attributions[idx] = np.zeros_like(attributions[0])
                attributions_output.extend(list(attributions))
        if attrs_flag != 0:
            file_dir = os.path.join(os.path.dirname(self.model_path), 'attrs')
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            if seq_attack == 0:
                np.savez_compressed(os.path.join(file_dir, f'attributions_IG_step{n_steps}_ac_{token_attack}'),
                                    np.array(attributions_output))
            else:
                np.savez_compressed(os.path.join(file_dir, f'attributions_IG_step{n_steps}_ac_{seq_attack}'),
                                    np.array(attributions_output))

        str2 = "TP: " + str(TP) + " FP: " + str(FP)
        print(str2)
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        str3 = 'false positive (FP): {}, false negative (FN): {}, true positive (TP): {}, ' \
               'true negative (TN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
            FP, FN, TP, TN, P, R, F1)
        print(str3)
        with open(self.model_name + '_output.log', 'a') as log_file:
            log_file.write(self.model_path + '\n')
            log_file.write(str2 + '\n')
            log_file.write(str3 + '\n')
