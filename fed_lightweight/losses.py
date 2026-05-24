from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def orthogonality_loss(role: torch.Tensor, style: torch.Tensor, weak: bool = True) -> torch.Tensor:
    """
    role: [batch, seq, dim]
    style: [batch, seq, dim]
    weak: If True, only constrain the mean vectors (weaker constraint)
          If False, constrain all positions (stronger constraint)
    """
    if weak:
        # Weak constraint: only constrain the mean vectors
        # This allows local correlations while encouraging global separation
        role_mean = role.mean(dim=[0, 1])  # [dim]
        style_mean = style.mean(dim=[0, 1])  # [dim]
        # Dot product of mean vectors (scalar)
        return torch.abs(role_mean @ style_mean)
    else:
        # Strong constraint: constrain all positions (original)
        role_flat = role.reshape(-1, role.size(-1))
        style_flat = style.reshape(-1, style.size(-1))
        prod = torch.matmul(role_flat.t(), style_flat)
        return torch.norm(prod, p="fro")


def mmd_loss(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Maximum Mean Discrepancy with RBF kernel.
    x, y: [batch, dim]
    """
    xx = _rbf_kernel(x, x, sigma)
    yy = _rbf_kernel(y, y, sigma)
    xy = _rbf_kernel(x, y, sigma)
    return xx.mean() + yy.mean() - 2 * xy.mean()


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    x = x.unsqueeze(1)  # [n,1,d]
    y = y.unsqueeze(0)  # [1,m,d]
    diff = x - y
    dist_sq = (diff * diff).sum(-1)
    return torch.exp(-dist_sq / (2 * sigma * sigma))


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, lambd: float):
        ctx.lambd = lambd
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def apply_grl(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReversal.apply(x, lambd)


def prototype_alignment_loss(
    local_proto: torch.Tensor,
    global_proto: torch.Tensor,
    loss_type: str = "mse"
) -> torch.Tensor:
    """
    Align local client prototype to global average prototype.

    This is the key loss for privacy protection: by aligning all client prototypes
    to the global average, they become indistinguishable from each other.

    Args:
        local_proto: [dim] - Current client's prototype (computed from batch)
        global_proto: [dim] - Global average prototype (from server aggregation)
        loss_type: "mse" for L2 loss, "cosine" for cosine distance

    Returns:
        Scalar loss value
    """
    if loss_type == "mse":
        # L2 distance: push local prototype towards global
        return F.mse_loss(local_proto, global_proto)
    elif loss_type == "cosine":
        # Cosine distance: 1 - cos_sim
        cos_sim = F.cosine_similarity(local_proto.unsqueeze(0), global_proto.unsqueeze(0))
        return 1.0 - cos_sim.squeeze()
    else:
        # Combined: both direction and magnitude alignment
        mse = F.mse_loss(local_proto, global_proto)
        cos_sim = F.cosine_similarity(local_proto.unsqueeze(0), global_proto.unsqueeze(0))
        return mse + (1.0 - cos_sim.squeeze())


class RunningPrototype:
    """
    Maintains a running estimate of the client prototype during training.

    Uses exponential moving average to smooth the prototype estimate,
    which is more stable than computing from each batch.
    """
    def __init__(self, dim: int, momentum: float = 0.99, device: str = "cuda"):
        self.dim = dim
        self.momentum = momentum
        self.device = device
        self.proto = None  # Will be initialized on first update
        self.count = 0

    def update(self, batch_proto: torch.Tensor):
        """
        Update running prototype with batch prototype.

        Args:
            batch_proto: [dim] - Prototype computed from current batch
        """
        batch_proto = batch_proto.detach()

        if self.proto is None:
            self.proto = batch_proto.clone()
        else:
            # Exponential moving average
            self.proto = self.momentum * self.proto + (1 - self.momentum) * batch_proto

        self.count += 1

    def get(self) -> Optional[torch.Tensor]:
        """Get current running prototype estimate."""
        return self.proto

    def reset(self):
        """Reset the running prototype."""
        self.proto = None
        self.count = 0


def prototype_confusion_loss(
    role_embeddings: torch.Tensor,
    global_proto: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Strong prototype confusion loss to prevent client attribution.

    This loss has two components:
    1. Centroid alignment: Push batch mean towards global prototype
    2. Variance reduction: Reduce within-batch variance to make embeddings more uniform

    Args:
        role_embeddings: [B, T, D] - Role embeddings from current batch
        global_proto: [D] - Global average prototype
        attention_mask: [B, T] - Optional attention mask

    Returns:
        Scalar loss value
    """
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
        # Masked mean pooling
        sum_emb = (role_embeddings * mask).sum(dim=[0, 1])
        count = mask.sum().clamp(min=1.0)
        batch_proto = sum_emb / count

        # Variance: mean squared distance from batch prototype
        diff = role_embeddings - batch_proto.unsqueeze(0).unsqueeze(0)
        var = ((diff ** 2) * mask).sum() / count
    else:
        batch_proto = role_embeddings.mean(dim=[0, 1])
        diff = role_embeddings - batch_proto.unsqueeze(0).unsqueeze(0)
        var = (diff ** 2).mean()

    # 1. Alignment loss: push batch prototype to global
    align_loss = F.mse_loss(batch_proto, global_proto)

    # 2. Cosine alignment for direction
    cos_sim = F.cosine_similarity(batch_proto.unsqueeze(0), global_proto.unsqueeze(0))
    cos_loss = 1.0 - cos_sim.squeeze()

    # 3. Variance reduction (optional, can be weighted separately)
    # Lower variance means more uniform embeddings, harder to distinguish
    var_loss = var * 0.1  # Small weight to not collapse embeddings completely

    return align_loss + cos_loss + var_loss


def embedding_noise_regularization(
    embeddings: torch.Tensor,
    noise_scale: float = 0.1,
    training: bool = True,
) -> torch.Tensor:
    """
    Add noise to embeddings during training for privacy.

    Args:
        embeddings: [B, T, D] or [B, D]
        noise_scale: Standard deviation of Gaussian noise
        training: Whether in training mode

    Returns:
        Noisy embeddings (same shape as input)
    """
    if training and noise_scale > 0:
        noise = torch.randn_like(embeddings) * noise_scale
        return embeddings + noise
    return embeddings


# =============================================================================
# Spherical Uniform Alignment
# =============================================================================
#
# Mathematical principle:
#   1. Normalize prototypes to unit hypersphere: p̂ᵢ = pᵢ / ||pᵢ||
#   2. Align all normalized prototypes to global average direction
#   3. Minimize directional differences between prototypes
#
# =============================================================================


def spherical_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize vector to unit hypersphere.

    Args:
        x: [D] or [B, D] or [B, T, D]
        eps: Small constant to prevent division by zero

    Returns:
        Normalized vector with norm = 1
    """
    norm = x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)
    return x / norm


def spherical_alignment_loss(
    local_proto: torch.Tensor,
    global_proto: torch.Tensor,
    lambda_direction: float = 1.0,
    lambda_uniformity: float = 0.5,
) -> torch.Tensor:
    """
    Spherical uniform alignment loss.

    Aligns local prototype to global prototype on the unit sphere.
    This is more effective than plain MSE alignment because:
    1. Eliminates magnitude information (KNN/LogReg cannot exploit magnitude differences)
    2. Enforces directional consistency (maximizes cosine similarity)

    Mathematical form:
        L = λ_dir * (1 - cos(p̂_local, p̂_global)) + λ_uni * ||p̂_local - p̂_global||²

    Args:
        local_proto: [D] - Local client prototype
        global_proto: [D] - Global average prototype
        lambda_direction: Direction alignment loss weight
        lambda_uniformity: Uniformity loss weight

    Returns:
        Scalar loss value

    Defense effect:
        - KNN: cos(p̂ᵢ, p̂ⱼ) → 1, distance differences vanish
        - LogReg: All points cluster together, cannot be linearly separated
    """
    # Normalize to unit sphere
    local_normed = spherical_normalize(local_proto)
    global_normed = spherical_normalize(global_proto)

    # Direction alignment loss: 1 - cos_sim (closer to 0 is better)
    cos_sim = F.cosine_similarity(
        local_normed.unsqueeze(0),
        global_normed.unsqueeze(0)
    ).squeeze()
    direction_loss = 1.0 - cos_sim

    # Uniformity loss: L2 distance on sphere
    uniformity_loss = F.mse_loss(local_normed, global_normed)

    return lambda_direction * direction_loss + lambda_uniformity * uniformity_loss


def spherical_confusion_loss(
    role_embeddings: torch.Tensor,
    global_proto: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    lambda_centroid: float = 1.0,
    lambda_dispersion: float = 0.1,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Spherical confusion loss - stronger privacy protection.

    Not only aligns centroids, but also reduces embedding dispersion,
    preventing attackers from exploiting distribution information.

    Mathematical form:
        L = λ_c * (1 - cos(μ̂, p̂_global)) + λ_d * mean(||êᵢ - μ̂||²)

    Where:
        μ̂ = normalize(mean(embeddings)) is the normalized batch centroid
        êᵢ = normalize(embeddingᵢ) is the normalized individual embedding

    Args:
        role_embeddings: [B, T, D] - Role embeddings
        global_proto: [D] - Global prototype
        attention_mask: [B, T] - Attention mask
        lambda_centroid: Centroid alignment weight
        lambda_dispersion: Dispersion penalty weight
        temperature: Temperature parameter (unused, reserved for future extension)

    Returns:
        Scalar loss value

    """
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
        # Compute masked mean
        sum_emb = (role_embeddings * mask).sum(dim=[0, 1])
        count = mask.sum().clamp(min=1.0)
        batch_centroid = sum_emb / count  # [D]
    else:
        batch_centroid = role_embeddings.mean(dim=[0, 1])  # [D]

    # Normalize
    centroid_normed = spherical_normalize(batch_centroid)
    global_normed = spherical_normalize(global_proto)

    # Centroid alignment loss
    cos_sim = F.cosine_similarity(
        centroid_normed.unsqueeze(0),
        global_normed.unsqueeze(0)
    ).squeeze()
    centroid_loss = 1.0 - cos_sim

    # Dispersion loss: reduce embedding dispersion around centroid
    # This prevents attackers from exploiting embedding distribution differences
    if attention_mask is not None:
        # Normalize each embedding
        emb_normed = spherical_normalize(role_embeddings)  # [B, T, D]
        # Compute distance to centroid
        diff = emb_normed - centroid_normed.unsqueeze(0).unsqueeze(0)
        dispersion = ((diff ** 2) * mask).sum() / count
    else:
        emb_normed = spherical_normalize(role_embeddings)
        diff = emb_normed - centroid_normed.unsqueeze(0).unsqueeze(0)
        dispersion = (diff ** 2).mean()

    return lambda_centroid * centroid_loss + lambda_dispersion * dispersion


class SphericalRunningPrototype:
    """
    Maintains running estimate of prototype on unit sphere.

    Unlike regular RunningPrototype, this class:
    1. Always keeps prototype normalized
    2. Uses spherical interpolation instead of linear interpolation
    """

    def __init__(self, dim: int, momentum: float = 0.99, device: str = "cuda"):
        self.dim = dim
        self.momentum = momentum
        self.device = device
        self.proto = None  # Normalized prototype
        self.count = 0

    def update(self, batch_proto: torch.Tensor):
        """
        Update running prototype (on sphere).

        Args:
            batch_proto: [D] - Current batch prototype
        """
        # Normalize input
        batch_normed = spherical_normalize(batch_proto.detach())

        if self.proto is None:
            self.proto = batch_normed.clone()
        else:
            # Spherical interpolation: linear interpolation then normalize
            interpolated = self.momentum * self.proto + (1 - self.momentum) * batch_normed
            self.proto = spherical_normalize(interpolated)

        self.count += 1

    def get(self) -> Optional[torch.Tensor]:
        """Get current normalized running prototype."""
        return self.proto

    def get_similarity_to_global(self, global_proto: torch.Tensor) -> float:
        """Compute cosine similarity with global prototype."""
        if self.proto is None:
            return 0.0
        global_normed = spherical_normalize(global_proto)
        sim = F.cosine_similarity(
            self.proto.unsqueeze(0),
            global_normed.unsqueeze(0)
        )
        return sim.item()

    def reset(self):
        """Reset running prototype."""
        self.proto = None
        self.count = 0

