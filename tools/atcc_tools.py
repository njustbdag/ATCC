import torch
import torch.nn as nn
import torch.nn.functional as F

def consistency_loss(ori, per):
    # Compute the Euclidean distance for each position
    euclidean_distance = torch.norm(ori - per, dim=-1)
    mean_distance = euclidean_distance.mean()
    return mean_distance


def processExamples(tensor, label_, label):
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


def contrastive_loss(ori, per, label):
    t = 0.5
    ori_with_label_0 = processExamples(ori, 0, label)
    if ori_with_label_0.size(0) == 0: return None  # Handle the case where no normal samples exist
    ori_with_label_1 = processExamples(ori, 1, label)
    if ori_with_label_1.size(0) == 0: return None  # Handle the case where no anomalous samples exist
    per_with_label_0 = processExamples(per, 0, label)
    per_with_label_1 = processExamples(per, 1, label)

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
