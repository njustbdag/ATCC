import sys

sys.path.append('../')

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.lightlog import TCN
from sklearn.decomposition import PCA
import argparse
from models.FGM import FGM
from tools.utils import *
import logging
from captum.attr import IntegratedGradients
from tools.atcc_tools import *
import time

max_len = 100
seed_everything(seed=1234)

logging.basicConfig(
    filename='lightlog_HDFS_result.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_ppa(semantic_file):
    ppa_result = []
    with open(semantic_file) as f:
        gdp_list = json.load(f)
        value = list(gdp_list.values())

        estimator = PCA(n_components=20)
        pca_result = estimator.fit_transform(value)

        result = pca_result - np.mean(pca_result, axis=0)
        pca = PCA(n_components=20)
        pca_result = pca.fit_transform(result)
        U = pca.components_

        for i, x in enumerate(result):
            for u in U[0:7]:
                x = x - np.dot(u.transpose(), x) * u
            ppa_result.append(list(x))

    ppa_result = np.array(ppa_result)
    return ppa_result


def trp(l, n, INT=0):
    r = l[:n]

    if len(r) < n:
        r.extend(list([INT]) * (n - len(r)))

    return r


def read_data(path, semantics, ig_flag=False):
    logs_series = pd.read_csv(path)
    if ig_flag:
        logs_series = logs_series.sample(n=50000, random_state=42)
    logs_series = logs_series.values
    label = logs_series[:, 1]
    logs_data = logs_series[:, 0]
    logs = []
    ppa_result = get_ppa(semantics)
    zero_vector = np.array([0] * 20)
    for i in range(0, len(logs_data)):
        try:
            data = logs_data[i]
            data = [int(n) for n in data.split()]
            data = trp(data, max_len)
            padding = [
                ppa_result[j - 1] if j != 0 else zero_vector
                for j in data
            ]
            logs.append(padding)

        except Exception as e:
            print(i, e)
    logs = np.array(logs)
    x = logs
    y = label
    x = np.reshape(x, (x.shape[0], x.shape[1], 20))
    y = np.array(y, dtype=int)
    return x, y


def hook_fn(module, input, output):
    global intermediate_output
    intermediate_output = output
    return None


def predict(_model, _test_loader, _epoch=-1):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    with torch.no_grad():
        for _inputs, _labels in _test_loader:
            _outputs = _model(_inputs)
            _, _predicted = torch.max(_outputs, 1)
            TP += ((_predicted == 1) & (_labels == 1)).sum().item()  # 预测为1且标签为1
            FP += ((_predicted == 1) & (_labels == 0)).sum().item()  # 预测为1且标签为0
            TN += ((_predicted == 0) & (_labels == 0)).sum().item()  # 预测为0且标签为0
            FN += ((_predicted == 0) & (_labels == 1)).sum().item()  # 预测为0且标签为1
    if _epoch >= 0:
        line = f'Epoch [{_epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}'
        print(line)
        logging.info(line)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print_lines = [
        f'TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}',
        f'Precision: {precision * 100:.3f}%, Recall: {recall * 100:.3f}%, F1 Score: {f1_score * 100:.3f}%'
    ]
    for line in print_lines:
        print(line)
    for line in print_lines:
        logging.info(line)


def predict_ig(_model, _test_loader, adv_attack=0, model_type=0):
    attributions_output = []
    ig = IntegratedGradients(model)
    for _inputs, _labels in _test_loader:
        _outputs = _model(_inputs)
        _, _predicted = torch.max(_outputs, 1)
        target = _labels.clone().detach()
        attributions, delta = ig.attribute(_inputs, n_steps=50, target=target,
                                           return_convergence_delta=True)
        attributions = attributions.cpu().detach().numpy()
        # Obtain positions where label and predicted are the same, and save the values of attrs at the corresponding
        # positions
        pos = np.where(((np.array(_labels.cpu()) == 1) & (np.array(_predicted.cpu()) == 1)) | (
                (np.array(_labels.cpu()) == 0) & (np.array(_predicted.cpu()) == 0)))
        for idx in range(_inputs.shape[0]):
            if idx not in pos[0]:
                attributions[idx] = np.zeros_like(attributions[0])
        attributions_output.extend(list(attributions))
    if model_type == 0:
        file_dir = os.path.join('../result/lightlog_HDFS/', 'attrs')
    elif model_type == 1:
        file_dir = os.path.join('../result/lightlog_adversarial_HDFS/', 'attrs')
    elif model_type == 2:
        file_dir = os.path.join('../result/lightlog_mix_HDFS/', 'attrs')

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    np.savez_compressed(os.path.join(file_dir, f'attributions_IG_step50_ac_{adv_attack}'),
                        np.array(attributions_output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict', 'ig'], default='train')
    # Specifies the training mode:
    # 0: Normal training, 1: Regular adversarial training, 2: ATCC training. Default is normal training.
    parser.add_argument('--attack', type=int, default=0)
    # Specifies whether to attack template tokens: 0 - no attack, 1 - attack. Default is no attack.
    parser.add_argument('--token', type=int, default=0)
    # Specifies whether to attack template sequences: 0 - no attack, 1 - attack. Default is no attack.
    parser.add_argument('--seq', type=int, default=0)
    # Specifies whether to enable attribution: 0 - disable, 1 - enable. Default is disabled.
    parser.add_argument('--num_epoch', type=int, default=2)
    parser.add_argument('--lmd', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.05)

    args = parser.parse_args()

    lmd = args.lmd
    alpha = args.alpha
    if args.token == 0:
        semantic_path = '../data/HDFS/hdfs_semantic_word2vec.json'
    elif args.token == 1:
        semantic_path = '../data/HDFS/attacked/hdfs_semantic_word2vec.json'

    if args.seq == 0:
        test_path = '../data/HDFS/robust_log_test.csv'
    elif args.seq == 1:
        test_path = '../data/HDFS/attacked/robust_log_test.csv'

    device = 'cuda'
    if args.mode == 'train':
        train_path = '../data/HDFS/robust_log_train.csv'
        valid_path = '../data/HDFS/robust_log_valid.csv'
        train_x, train_y = read_data(train_path, semantic_path)
        valid_x, valid_y = read_data(valid_path, semantic_path)
        train_x_tensor = torch.tensor(train_x, dtype=torch.float32).to(device)
        train_y_tensor = torch.tensor(train_y, dtype=torch.long).to(device)
        valid_x_tensor = torch.tensor(valid_x, dtype=torch.float32).to(device)
        valid_y_tensor = torch.tensor(valid_y, dtype=torch.long).to(device)

        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
        valid_dataset = TensorDataset(valid_x_tensor, valid_y_tensor)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # 64
        valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False) # 64

        model = TCN().to(device=device)
        criterion = nn.CrossEntropyLoss()

        if args.attack == 0:
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            num_epochs = 120
            for epoch in range(num_epochs):
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                start = time.strftime("%H:%M:%S")
                line = f"phase: ordinary train | {start} | Learning rate: {lr}"
                print(line)
                logging.info(line)

                model.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                model.eval()
                predict(model, valid_loader, epoch)
                torch.save(model.state_dict(), f'../result/lightlog_HDFS/tcn_model_epoch{epoch}.pth')
            torch.save(model.state_dict(), f'../result/lightlog_HDFS/tcn_model_last.pth')
        elif args.attack == 1:
            optimizer = optim.Adam(model.parameters(), lr=0.0003)
            model.load_state_dict(torch.load('../result/lightlog_HDFS/tcn_model_epoch98.pth'))
            num_epochs = 2
            fgm = FGM()
            for epoch in range(num_epochs):
                model.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    inputs.requires_grad = True
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward(retain_graph=True)
                    mask = (inputs != 0).float()
                    fgm.attack(features=inputs, mask=mask, epsilon=0.03)
                    outputs2 = model(inputs)
                    loss_perturbed = criterion(outputs2, labels)
                    total_loss = (loss + loss_perturbed) / 2
                    total_loss.backward()
                    fgm.restore(inputs)
                    optimizer.step()

                model.eval()
                predict(model, valid_loader, epoch)
                if epoch > 0 and epoch % 2 == 0:
                    torch.save(model.state_dict(), f'../result/lightlog_adversarial_HDFS/tcn_model_epoch{epoch}.pth')
            torch.save(model.state_dict(), f'../result/lightlog_adversarial_HDFS/tcn_model_last.pth')
        elif args.attack == 2:
            optimizer = optim.Adam(model.parameters(), lr=0.0003)
            model.load_state_dict(torch.load('../result/lightlog_HDFS/tcn_model_epoch98.pth'))
            num_epochs = 2
            fgm = FGM()
            model.resblock2.register_forward_hook(hook_fn)

            for epoch in range(num_epochs):
                model.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    inputs.requires_grad = True
                    global intermediate_output
                    outputs = model(inputs)
                    original_example = intermediate_output
                    loss = criterion(outputs, labels)
                    loss.backward(retain_graph=True)
                    mask = (inputs != 0).float()
                    fgm.attack(features=inputs, mask=mask, epsilon=0.03)
                    outputs2 = model(inputs)
                    perturbed_example = intermediate_output
                    loss_perturbed = criterion(outputs2, labels)
                    loss_distance = consistency_loss(original_example, perturbed_example)  # compute consistency loss
                    loss_contra = contrastive_loss(original_example, perturbed_example, labels)  # Compute contrastive loss
                    if loss_contra:
                        total_loss = (1 - lmd) * (loss + loss_perturbed) / 2 + alpha * loss_contra + \
                                     (lmd - alpha) * loss_distance
                    else:
                        print('loss_contra is None')
                        total_loss = (loss + loss_perturbed) / 2
                    total_loss.backward()
                    optimizer.step()

                model.eval()
                predict(model, valid_loader, epoch)
                torch.save(model.state_dict(), f'../result/lightlog_mix_HDFS/tcn_model_epoch{epoch}.pth')
            torch.save(model.state_dict(), f'../result/lightlog_mix_HDFS/tcn_model_last.pth')


    elif args.mode == 'predict':
        test_x, test_y = read_data(test_path, semantic_path)
        test_x_tensor = torch.tensor(test_x, dtype=torch.float32).to(device)
        test_y_tensor = torch.tensor(test_y, dtype=torch.long).to(device)
        test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        model = TCN().to(device)
        print('epoch:', args.num_epoch)
        if args.attack == 0:
            logging.info(f'epoch: {args.num_epoch}')
            if args.token == 1:
                logging.info(f'token attack')
            if args.seq == 1:
                logging.info(f'seq attack')
            model.load_state_dict(
                torch.load(f'../result/lightlog_HDFS/tcn_model_epoch{args.num_epoch}.pth', weights_only=True))
        elif args.attack == 1:
            model.load_state_dict(torch.load(f'../result/lightlog_adversarial_HDFS/tcn_model_last.pth',
                                             weights_only=True))
        elif args.attack == 2:
            model.load_state_dict(torch.load(f'../result/lightlog_mix_HDFS/tcn_model_last.pth',
                                             weights_only=True))
        model.eval()
        predict(model, test_loader)
    elif args.mode == 'ig':
        test_x, test_y = read_data(test_path, semantic_path, ig_flag=True)
        test_x_tensor = torch.tensor(test_x, dtype=torch.float32).to(device)
        test_y_tensor = torch.tensor(test_y, dtype=torch.long).to(device)
        test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
        test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
        model = TCN().to(device)
        if args.attack == 0:
            model.load_state_dict(
                torch.load(f'../result/lightlog_HDFS/tcn_model_epoch{args.num_epoch}.pth', weights_only=True))
        elif args.attack == 1:
            model.load_state_dict(torch.load(f'../result/lightlog_adversarial_HDFS/tcn_model_last.pth',
                                             weights_only=True))
        elif args.attack == 2:
            model.load_state_dict(
                torch.load(f'../result/lightlog_mix_HDFS/tcn_model_last.pth', weights_only=True))
        if args.token != 0:
            predict_ig(model, test_loader, args.token, args.attack)
        elif args.seq != 0:
            predict_ig(model, test_loader, args.seq, args.attack)
        else:
            predict_ig(model, test_loader, args.token, args.attack)
