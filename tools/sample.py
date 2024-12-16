import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import gc
import pickle
from joblib import dump, load
from pytorch_pretrained_bert import BertTokenizer
from tools.utils import *
from scipy.sparse import csr_matrix, lil_matrix

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


def trp(l, n, INT=0):
    """ Truncate or pad a list """
    r = l[:n]

    if len(r) < n:
        r.extend(list([INT]) * (n - len(r)))

    return r


def count_train_idf(df_train, df_templates):
    df_train_idf = pd.DataFrame(index=range(1, len(df_templates) + 1), columns=['EventId', 'Occurrences', 'idf'])
    df_train_idf['EventId'] = df_train_idf.index
    df_train_idf['Occurrences'] = pd.Series([0] * (len(df_templates) + 1))
    df_train_idf['idf'] = pd.Series([0] * (len(df_templates) + 1))
    count_dict = {}
    for i in range(len(df_train)):
        sequence = df_train.loc[i, 'Sequence']
        numbers = list(map(int, sequence.split()))
        for num in numbers:
            if num in count_dict:
                count_dict[num] += 1
            else:
                count_dict[num] = 1
    for num, count in count_dict.items():
        df_train_idf.loc[num, 'Occurrences'] = count

    for i in range(1, len(df_train_idf) + 1):
        if df_train_idf.loc[i, 'Occurrences'] != 0:
            df_train_idf.loc[i, 'idf'] = np.log(
                float(df_train_idf['Occurrences'].sum()) / df_train_idf.loc[i, 'Occurrences'] + 1)
    scaler = MinMaxScaler(feature_range=(1, 2))
    df_tmp = df_train_idf[df_train_idf['idf'] != 0]
    scaled_data = scaler.fit_transform(df_train_idf[df_train_idf['idf'] != 0][['idf']])
    df_tmp.loc[:, 'idf'] = scaled_data

    for i in range(1, len(df_train_idf) + 1):
        if df_train_idf.loc[i, 'idf'] == 0:
            df_train_idf.loc[i, 'idf'] = 1
        else:
            df_train_idf.loc[i, 'idf'] = df_tmp.loc[i, 'idf']
    return df_train_idf


def indice_to_templates(sequence, indice_templates):
    sequence = sequence.split()
    return [indice_templates[int(i)] for i in sequence]


def count_tf_idf(train_df, templates, indice_templates):
    import tools.preprocessing as preprocessing
    extractor = preprocessing.FeatureExtractor()
    extractor.vocab.build_vocab(templates['EventTemplate'].tolist())
    sequences_list = train_df['Sequence'].apply(lambda x: indice_to_templates(x, indice_templates))
    templates_set = [item for sublist in sequences_list for item in sublist]
    extractor.vocab.fit_tfidf(templates_set)
    return extractor


# The sliding window and session window were used during the previous dataset partitioning.
# Here, we load the already partitioned dataset without needing to differentiate between window types.
def sliding_window(data_dir, model_name, window_size, datatype, data_type, template_file, token_attack, seq_attack,
                   attrs_flag):
    target = datatype  # 'train', 'valid', 'predict','explain'
    if attrs_flag != 0 and target == 'predict':
        target = 'explain'
        explain_sample = 50000  # HDFS, BGL: 50k, Thunderbird: 30k
    # Only use the training set to calculate the event_idf for mlog or tf-idf for logrobust
    df_train = pd.read_csv(os.path.join(data_dir, data_type, 'robust_log_train.csv'))
    df_template = pd.read_csv(template_file)
    output_dir_event_idf = os.path.join(data_dir + data_type + "/event_idf.csv")
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []

    if datatype == "train":
        input_data = data_dir + data_type + "/robust_log_train.csv"
    if datatype == "val":
        input_data = data_dir + data_type + "/robust_log_valid.csv"
    if datatype == "predict":
        if seq_attack == 1:
            input_data = data_dir + data_type + "/attacked/robust_log_test.csv"
            print(input_data)
        else:
            input_data = data_dir + data_type + "/robust_log_test.csv"

    if model_name == "mlog":
        if os.path.exists(output_dir_event_idf):
            df_train_idf = pd.read_csv(output_dir_event_idf, index_col=0)
        else:
            df_train_idf = count_train_idf(df_train, df_template)
            df_train_idf.to_csv(output_dir_event_idf)
        if token_attack == 0:
            template_semantic = read_json(os.path.join(data_dir, data_type, 'templates_semantic.json'))
            idf_semantics = {eventId - 1: df_train_idf.loc[eventId, 'idf'] * np.array(template_semantic[eventId - 1])
                             for eventId in range(1, len(template_semantic) + 1)}
        elif token_attack == 1:
            template_semantic = read_json(os.path.join(data_dir, data_type, 'attacked', 'templates_semantic.json'))
            idf_semantics = {eventId - 1: 1 * np.array(template_semantic[eventId - 1])
                             for eventId in range(1, len(template_semantic) + 1)}
        zero_vector = np.array([0] * 768)

    if model_name == "logrobust":
        # This step will produce the size of the vocab
        if token_attack == 0:
            # If using attacked templates, the attacked templates corresponding to the index_templates need to be
            # regenerated after generating the extractor
            template_semantic = np.array(read_json(os.path.join(data_dir, data_type, 'templates_semantic_300.json')))
        elif token_attack == 1:
            template_semantic = np.array(
                read_json(os.path.join(data_dir, data_type, 'attacked', 'templates_semantic_300.json')))
        zero_vector = np.array([0] * 300)
    print(template_file)
    input_df = pd.read_csv(input_data)  # Depends on whether it's the training set, validation set, or test set
    if target == 'explain':
        # Randomly select {explain_sample} samples, the random seed has been fixed
        input_df = input_df.sample(explain_sample).reset_index(drop=True)

    seq_len = window_size
    labels = input_df['label'].tolist()

    for i in tqdm(range(len(input_df))):
        try:
            ori_seq = list(map(int, input_df.loc[i, 'Sequence'].split()))
            Sequential_pattern = trp(ori_seq, seq_len)
            if model_name == "mlog":
                Semantic_pattern = [
                    idf_semantics[eventId - 1] if eventId != 0 else zero_vector
                    for eventId in Sequential_pattern
                ]
            if model_name == "logrobust":
                Semantic_pattern = [
                    template_semantic[eventId - 1] if eventId != 0 else zero_vector
                    for eventId in Sequential_pattern
                ]
            result_logs['Semantics'].append(Semantic_pattern)
        except Exception as e:
            print(e)
            print(i)
            print(input_df.loc[i, 'Sequence'])
    print('Number of sessions({}): {}'.format(input_data, len(result_logs['Semantics'])))
    return result_logs, labels
