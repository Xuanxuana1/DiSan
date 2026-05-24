from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    pretrained_model_path: str = "../long-t5-tglobal-base"
    max_seq_length: int = 1536
    role_dim: int = 256
    style_dim: int = 256
    lora_rank: int = 8
    use_grl: bool = False
    use_mmd: bool = True
    dropout: float = 0.1
    # New: two-stream architecture control
    enable_two_stream: bool = True  # Enable role/style disentanglement
    use_residual: bool = True  # Use residual connection to preserve information flow
    mix_weight_init: float = 0.5  # Initial weight for residual mixing (α in fused = α*fuse + (1-α)*h)
    # Residual mode for privacy protection:
    # - 'full': original residual connection (may leak client info via hidden)
    # - 'gated': use learnable gate to control how much hidden info passes through
    # - 'projected': project hidden through same bottleneck before residual
    # - 'none': disable residual connection entirely (safest for privacy)
    residual_mode: str = 'full'  # Default to 'full' for privacy
    # Client adversarial classifier
    adversary_hidden_dim: int = 128  # Hidden dim for adversary MLP
    num_clients: int = 7  # Number of clients for adversarial training
    # LoRA configuration for communication-efficient federated learning
    use_lora: bool = False  # Enable LoRA for efficient fine-tuning
    lora_r: int = 8  # LoRA rank (lower = fewer params, higher = more capacity)
    lora_alpha: int = 32  # LoRA scaling factor
    lora_dropout: float = 0.1  # LoRA dropout


@dataclass
class TrainingConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 1400
    batch_size: int = 16
    grad_accum: int = 4
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    # Deprecated: lambda_align (MMD alignment) replaced by lambda_proto (prototype alignment)
    lambda_align: float = 0.0  # Deprecated, use lambda_proto instead
    lambda_orth: float = 0.1  # Reduced for weak orthogonality constraint
    lambda_proto: float = 0.0  # Prototype-based alignment (Phase 2 only)
    grl_lambda: float = 0.1
    lambda_priv_start: float = 0.1
    lambda_priv_end: float = 1.0
    lambda_priv_warmup_steps: int = 2000
    save_every: int = 1000
    log_every: int = 10
    device: str = "cuda"
    val_split_ratio: float = 0.15
    early_stop_patience: int = 3
    # New: alignment and loss control
    align_max_pairs: int = 1024  # Max pairs for MMD (if still used)
    use_weak_orth: bool = True  # Use weak orthogonality constraint (mean vectors only)
    # Client adversarial training for Role privacy
    lambda_adv: float = 0.3  # Weight for adversarial loss (0.3 balances privacy and quality)
    lambda_adv_warmup_steps: int = 500  # Warmup steps before full adversarial training
    adv_loss_clamp: float = 5.0  # Clamp adversarial loss to prevent explosion
    # Prototype alignment for privacy protection (Phase 2+)
    # This aligns client-level prototypes to global average, making them indistinguishable
    lambda_proto_align: float = 1.0  # Weight for prototype alignment loss
    proto_align_warmup_steps: int = 100  # Warmup steps before full alignment
    proto_update_interval: int = 5  # Update running prototype every N steps
    # Spherical Uniform Alignment
    use_spherical_align: bool = True  # Use spherical alignment instead of plain MSE alignment
    lambda_sphere_direction: float = 1.0  # Direction alignment weight
    lambda_sphere_uniformity: float = 0.5  # Uniformity weight
    lambda_sphere_dispersion: float = 0.1  # Dispersion penalty weight


@dataclass
class DataConfig:
    clients: List[str] = field(default_factory=list)
    data_root: str = "../data"
    src_field: str = "original_text"
    tgt_field: str = "rewritten_text"


@dataclass
class PrototypeConfig:
    entity_types: List[str] = field(
        default_factory=lambda: ["PER", "ORG", "LOC", "ID", "DATE", "AMT"]
    )
    include_covariance: bool = False


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prototypes: PrototypeConfig = field(default_factory=PrototypeConfig)
    output_dir: str = "../checkpoints/fed_disp"
    strategy: str = "local"  # options: local, oneshot_avg, oneshot_proto
    seed: int = 42
    precision: Optional[str] = None  # e.g., "bf16" or "fp16"
    gliner_model_dir: Optional[str] = None

