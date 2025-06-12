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

def hierarchical_contrastive_loss(features, labels, temperature=0.07, alpha=1.0, beta=1.0):
    device = features.device
    features = F.normalize(features, dim=1)
    sim_matrix = torch.matmul(features, features.T) / temperature
    B = features.size(0)

    # 공통: 자기 자신 제외
    logits_mask = torch.ones((B, B), device=device) - torch.eye(B, device=device)

    # -------------------
    # Level 1: class 3 vs others
    # -------------------
    label_mask_L1 = (labels == 3).view(-1, 1)  # shape (B, 1)
    mask_L1_pos = torch.eq(label_mask_L1, label_mask_L1.T).float()  # True if both in class 3 or both not
    mask_L1_pos = mask_L1_pos * logits_mask  # remove self-comparisons

    exp_sim = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    mean_log_prob_pos_L1 = (mask_L1_pos * log_prob).sum(dim=1) / (mask_L1_pos.sum(dim=1) + 1e-12)
    loss_L1 = -mean_log_prob_pos_L1.mean()

    # -------------------
    # Level 2: class 2 vs [0,1,4]
    # -------------------
    valid_L2 = (labels != 3)  # only use class 0,1,2,4
    idx_L2 = valid_L2.nonzero(as_tuple=False).squeeze()

    if idx_L2.numel() > 1:
        f_L2 = features[idx_L2]
        l_L2 = labels[idx_L2]
        sim_L2 = torch.matmul(f_L2, f_L2.T) / temperature
        B2 = f_L2.size(0)

        logits_mask_L2 = torch.ones((B2, B2), device=device) - torch.eye(B2, device=device)

        label_mask_L2 = (l_L2 == 2).view(-1, 1)
        mask_L2_pos = torch.eq(label_mask_L2, label_mask_L2.T).float() * logits_mask_L2

        exp_sim_L2 = torch.exp(sim_L2) * logits_mask_L2
        log_prob_L2 = sim_L2 - torch.log(exp_sim_L2.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob_pos_L2 = (mask_L2_pos * log_prob_L2).sum(dim=1) / (mask_L2_pos.sum(dim=1) + 1e-12)
        loss_L2 = -mean_log_prob_pos_L2.mean()
    else:
        loss_L2 = 0.0  # no valid samples for level 2

    return alpha * loss_L1 + beta * loss_L2


import torch
import torch.nn.functional as F

def group_contrastive_loss(features, labels, temperature=0.07):
    """
    3-way group contrastive loss for:
    - group 0: class 0,1,4
    - group 1: class 2
    - group 2: class 3
    """
    device = features.device
    B = features.size(0)
    
    # Normalize feature vectors
    features = F.normalize(features, dim=1)
    sim_matrix = torch.matmul(features, features.T) / temperature

    # --------------------------
    # Group mapping: label → group
    # --------------------------
    group_map = {0: 0, 1: 1, 2: 1, 3: 0, 4: 0}
    group_labels = torch.tensor([group_map[int(l)] for l in labels], device=device)

    # Positive pair mask (same group)
    group_labels = group_labels.view(-1, 1)
    pos_mask = torch.eq(group_labels, group_labels.T).float()

    # Remove self-similarity
    logits_mask = torch.ones_like(pos_mask) - torch.eye(B, device=device)
    pos_mask = pos_mask * logits_mask
    sim_matrix = sim_matrix.masked_fill(torch.eye(B, device=device).bool(), -1e9)

    # Contrastive loss
    exp_sim = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-12)
    loss = -mean_log_prob_pos.mean()
    
    return loss

import torch
import torch.nn.functional as F

def multilabel_soft_contrastive_loss(embeddings, labels, temperature=0.1):
    """
    embeddings: (N, D) float tensor, feature vectors
    labels: (N, C) binary multi-hot labels
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature  # (N, N)

    # Compute label similarity matrix using Jaccard index
    # Intersection: a @ b.T
    inter = (labels @ labels.T).float()  # (N, N)
    union = ((labels.unsqueeze(1) + labels.unsqueeze(0)) > 0).sum(dim=2).float()  # (N, N)
    jaccard_sim = inter / (union + 1e-8)  # Avoid divide by zero

    # Remove self-similarity
    mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
    sim_matrix = sim_matrix[mask].view(sim_matrix.size(0), -1)
    jaccard_sim = jaccard_sim[mask].view(jaccard_sim.size(0), -1)

    # Apply soft contrastive loss: log(sum(exp(sim) * jaccard_sim)) - sim of anchor
    exp_sim = torch.exp(sim_matrix)
    weighted_sim = exp_sim * jaccard_sim
    loss = -torch.log((weighted_sim.sum(dim=1) + 1e-8) / (exp_sim.sum(dim=1) + 1e-8))

    return loss.mean()
