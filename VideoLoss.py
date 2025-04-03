import torch
import torch.nn.functional as F

def improved_contrastive_loss(features, labels, temperature):
    device = features.device
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T) / temperature

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    logits_mask = torch.ones_like(mask) - torch.eye(features.size(0), device=device)
    mask = mask * logits_mask

    exp_sim = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
    return -mean_log_prob_pos.mean()