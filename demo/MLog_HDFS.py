# -*- coding: utf-8 -*-

import argparse
import os.path
import sys

sys.path.append('../')

from tools.predict import Predicter
from tools.train import Trainer
from tools.utils import *
from models.mlog import *

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
options['pre_train'] = "../data/HDFS/templates_semantic.json"

# Model
options['input_size'] = 768
options['hidden_size'] = 648
options['num_layers'] = 2
options['num_classes'] = 2
options['cnn_length'] = 3
# Train
options['batch_size'] = 32
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.001  # Initial learning rate for the model
options['max_epoch'] = 21
options['lr_step'] = (40, 50)
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "mlog"
options['save_dir'] = "../result/mlog/"
options['Event_TF_IDF'] = 1
options['template'] = "../data/HDFS/HDFS.log_templates.csv"

# adversarial
options['lmd'] = 0.2
options['alpha'] = 0.05

# Predict
options['num_candidates'] = -1
options['data_type'] = 'HDFS'
options['filter_num'] = 32
options['filter_size'] = '2,3,4'
options['pool'] = True
seed_everything(seed=1234)
base_epoch = 20


def train(is_attack=0):
    Model = MLog(in_size2=200,
                 input_sz=options['input_size'],
                 hidden_sz=options['hidden_size'],
                 cnn_length=options['cnn_length'],
                 mog_iteration=2,
                 filter_num=options['filter_num'],
                 filter_sizes=options['filter_size'],
                 pre_train=options['pre_train'])
    trainer = Trainer(Model, options)
    trainer.start_train('MLog_HDFS', is_attack)


def predict(token_attack, seq_attack, attrs_flag=0):
    Model = MLog(in_size2=200,
                 input_sz=options['input_size'],
                 hidden_sz=options['hidden_size'],
                 mog_iteration=2,
                 cnn_length=options["cnn_length"],
                 filter_num=options['filter_num'],
                 filter_sizes=options['filter_size'],
                 pre_train=options['pre_train'])
    predicter = Predicter(Model, options)
    predicter.predict_supervised(token_attack, seq_attack, attrs_flag=attrs_flag)


if __name__ == '__main__':
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
    iter = args.iter

    if args.mode == 'train':
        if args.attack == 0:
            options['resume_path'] = None  
            options['save_dir'] = "../result/mlog_HDFS/"
            options['max_epoch'] = base_epoch + 3
            train(is_attack=0)
        if args.attack == 1:
            options['resume_path'] = os.path.join("../result/mlog_HDFS/", f"mlog_epoch{base_epoch}.pth")
            options['save_dir'] = os.path.join("../result/mlog_adversarial_HDFS/")
            options['max_epoch'] = base_epoch + 3
            train(is_attack=1)
        if args.attack == 2:
            options['resume_path'] = os.path.join("../result/mlog_HDFS/", f"mlog_epoch{base_epoch}.pth")
            options['save_dir'] = os.path.join("../result/mlog_ATCC_HDFS/")
            options['max_epoch'] = base_epoch + 3
            options['lmd'] = args.lmd
            options['alpha'] = args.alpha
            train(is_attack=2)
    else:
        if args.attack == 0:
            options['model_path'] = os.path.join("../result/mlog_HDFS/", f"mlog_epoch{base_epoch + iter}.pth")
        if args.attack == 1:
            options['model_path'] = os.path.join("../result/mlog_adversarial_HDFS/",
                                                 f"mlog_epoch{base_epoch + iter}.pth")
        if args.attack == 2:
            options['model_path'] = os.path.join("../result/mlog_ATCC_HDFS/", f"mlog_epoch{base_epoch + iter}.pth")
        if args.attrs == 0:
            options['batch_size'] = 1000
        else:
            options['batch_size'] = 100
        predict(args.token, args.seq, args.attrs)
