import torch


class FGM:
    def __init__(self):
        pass

    def attack(self, features, mask, epsilon=0.03):
        """
        :param features: Original embedding vectors, shape (batch_size, seq_len, emb_dim)
        :param mask: Mask, shape (batch_size, seq_len), padded parts are 0, valid parts are 1
        :param epsilon: Perturbation magnitude
        """
        grad = features.grad
        if grad is not None:
            norm = torch.norm(grad, p=2, dim=-1, keepdim=True)
            # Avoid division by zero
            norm = torch.where(norm == 0, torch.ones_like(norm), norm)
            # Calculate perturbation and add it to the original features
            r_at = epsilon * grad / norm
            # Use mask to exclude the padded parts
            r_at = r_at * mask  # Expand the mask dimensions to match r_at
            features.data.add_(r_at)  # Add perturbation to generate perturbed features
