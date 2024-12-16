#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys

sys.path.append('../')

from tools.predict import Predicter
from tools.train import Trainer
from tools.utils import *
from models.logrobust import *

# Config Parameters

options = dict()
options['data_dir'] = '../data/'
options['device'] = "cuda"

# Sample
options['sample'] = "session_window"
options['window_size'] = 50

# Features
options['sequentials'] = False
options['quantitatives'] = False
options['semantics'] = True
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics']])
options['pre_train'] = "../data/HDFS/fastText_weight.json"
options['vocab_size'] = 122

# Model
options['input_size'] = 300
options['hidden_size'] = 512
options['num_layers'] = 1
options['num_classes'] = 2
options['num_directions'] = 2

# Train
options['batch_size'] = 32
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 40
options['lr_step'] = (40, 50)
options['lr_decay_ratio'] = 0.1
options['Event_TF_IDF'] = 0

options['resume_path'] = None
options['model_name'] = "logrobust"
options['Event_TF_IDF'] = 0
options['template'] = "../data/HDFS/HDFS.log_templates.csv"

# Predict
options['num_candidates'] = -1
options['data_type'] = "HDFS"

seed_everything(seed=1234)
base_epoch = 20

def train(is_attack=0):
    Model = logrobust(input_size=options['input_size'],
                      hidden_size=options['hidden_size'],
                      num_layers=options['num_layers'],
                      num_keys=options['num_classes'],
                      window_size=options['window_size'],
                      num_directions=options['num_directions'],
                      pre_train=options['pre_train'],
                      vocab_size=options['vocab_size'])
    trainer = Trainer(Model, options)
    trainer.start_train('LogRobust_HDFS', is_attack)


def predict(token_attack, seq_attack, attrs_flag=0):
    Model = logrobust(input_size=options['input_size'],
                      hidden_size=options['hidden_size'],
                      num_layers=options['num_layers'],
                      num_keys=options['num_classes'],
                      window_size=options['window_size'],
                      num_directions=options['num_directions'],
                      pre_train=options['pre_train'],
                      vocab_size=options['vocab_size'])
    predicter = Predicter(Model, options)
    predicter.predict_supervised(token_attack, seq_attack, attrs_flag=attrs_flag)


if __name__ == "__main__":
    last_epoch = "logrobust_last.pth"
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'])
    # Specifies the training mode:
    # 0: Normal training, 1: Regular adversarial training, 2: ATCC training. Default is normal training.
    parser.add_argument('--attack', type=int, default=0)
    # Specifies whether to attack template tokens: 0 - no attack, 1 - attack. Default is no attack.
    parser.add_argument('--token', type=int, default=0)
    # Specifies whether to attack template sequences: 0 - no attack, 1 - attack. Default is no attack.
    parser.add_argument('--seq', type=int, default=0)
    # Specifies whether to enable attribution: 0 - disable, 1 - enable. Default is disabled.
    parser.add_argument('--attrs', type=int, default=0)
    parser.add_argument('--lmd', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--iter', type=int, default=2)
    args = parser.parse_args()
    options['lmd'] = args.lmd
    options['alpha'] = args.alpha

    if args.mode == 'train':
        if args.attack == 0:
            options['resume_path'] = os.path.join("../result/logrobust_HDFS/", f"logrobust_epoch20.pth")# None
            options['save_dir'] = "../result/logrobust_HDFS/"
            options['max_epoch'] = base_epoch + 3
            train(is_attack=0)
        if args.attack == 1:
            options['resume_path'] = os.path.join("../result/logrobust_HDFS/", f"logrobust_epoch{base_epoch}.pth")
            options['save_dir'] = os.path.join("../result/logrobust_adversarial_HDFS/")
            options['max_epoch'] = base_epoch + 3
            train(is_attack=1)
        if args.attack == 2:
            options['resume_path'] = os.path.join("../result/logrobust_HDFS/", f"logrobust_epoch{base_epoch}.pth")
            options['save_dir'] = os.path.join("../result/logrobust_ATCC_HDFS/")
            options['max_epoch'] = base_epoch + 3
            options['lmd'] = args.lmd
            options['alpha'] = args.alpha
            train(is_attack=2)
    else:
        if args.attack == 0:
            options['model_path'] = os.path.join("../result/logrobust_HDFS/", f"logrobust_epoch{base_epoch+2}.pth")
        if args.attack == 1:
            options['model_path'] = os.path.join("../result/logrobust_adversarial_HDFS/", f"logrobust_epoch{base_epoch+2}.pth")
        if args.attack == 2:
            options['model_path'] = os.path.join("../result/logrobust_ATCC_HDFS/", f"logrobust_epoch{base_epoch+2}.pth")
        if args.attrs == 0:
            options['batch_size'] = 1000
        predict(args.token, args.seq, args.attrs)
