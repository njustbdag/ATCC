# ATCC

 
 
This is the basic implementation of our submission in TNSM: **ATCC: Adversarial Training-Based Approach for Improving Log Anomaly Detection Model Robustness**.
- [ATCC](#ATCC)
  * [Description](#description)
  * [Base Models](#Base Models)
  * [Datasets](#datasets)
  * [Log Parser](#Log Parser)
  * [Running](#Running)

## Description

The ATCC method is a novel enhancement for log anomaly detection, designed to address the challenges posed by continuously evolving log templates and noisy data. By combining adversarial training with contrastive learning, ATCC improves both the detection accuracy and robustness of existing log anomaly detection models. The method involves two stages: first, training a base model with its original objective function, and then fine-tuning it using a novel objective function that incorporates contrastive loss and consistency loss for improved robustness.

This open-source code implements ATCC and demonstrates its application to enhance three representative supervised log anomaly detection models. Extensive experiments on widely used benchmark datasets show that ATCC improves model performance in terms of both detection accuracy and robustness against noisy and evolving logs.

## Base Models
We implemented ATCC on three log anomaly detection base models to enhance their robustness.

`LogRobust` is a log anomaly detection method based on an attention-based Bi-LSTM model, which extracts 
semantic information from log events and captures the context of log sequences, implemented in the 
code from [deep-loglizer](https://github.com/logpai/deep-loglizer).

`MLog` is a hybrid deep neural network for log sequence anomaly detection that combines transformer-based event 
representation with an LSTM-CNN architecture to capture both global and local sequential patterns, implemented in the 
code from [MLog](https://github.com/njustbdag/MLog).

`LightLog` is a lightweight log anomaly detection method, utilizing low-dimensional 
semantic vector space and a lightweight temporal convolutional network, implemented in the code from 
[Light](https://github.com/Aquariuaa/LightLog).

Since the source code of `LightLog`  is implemented using the TensorFlow, in order to adapt it to our method 
and facilitate comparative experiments, we provide an implementation of `LightLog` in the PyTorch in our code.



## Datasets

We implemented `ATCC` on HDFS, BGL and Thunderbird from [LogHub](https://github.com/logpai/loghub).

## Log Parser
We use the log parser [Drain](https://github.com/logpai/Drain3) to parse the raw logs and obtain log templates.

### Running
Here are the steps to execute the ATCC method using the LogRobust model as an example:

Step 0: Move the training preparation data to the corresponding dataset directory `./data/`. In this project, we provide the HDFS version of the data, including the semantic vectors for the three base models, log templates, as well as the training, validation, and test sets.

Step 1: Enter `python LogRobust_HDFS.py --train 0` to get the base trained LogRobust model.

Step 2: Enter `python LogRobust_HDFS.py --train 2` to fine-tune the model using ATCC.

Step 3: Enter `python LogRobust_HDFS.py --predict 0` to evaluate the accuracy of the base LogRobust model, or enter `python LogRobust_HDFS.py --predict 2` to evaluate the accuracy of the fine-tuned LogRobust model using ATCC.


Additionally, if you are interested, you can also enter `python LogRobust_HDFS.py --train 1` and `python LogRobust_HDFS.py --predict 1` to train and evaluate the LogRobust model fine-tuned with general adversarial training.

Enter `python LogRobust_HDFS.py --predict 0 --attrs 1`  to obtain feature attributions for the base LogRobust model.

If you wish to adjust the model training parameters, you can modify these two parameters by adding options like `--lmd 0.2` and `--alpha 0.05` at the end of the command.