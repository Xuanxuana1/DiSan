from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. Install with: pip install peft")

try:
    from .config import ModelConfig, TrainingConfig
    from .losses import orthogonality_loss, apply_grl
except ImportError:
    from config import ModelConfig, TrainingConfig
    from losses import orthogonality_loss, apply_grl


class ClientAdversary(nn.Module):
    """
    Client adversarial classifier for Role embedding.
    Goal: Make Role embedding unable to predict client identity.

    Uses Gradient Reversal Layer (GRL) to train encoder to fool this classifier.
    """
    def __init__(self, input_dim: int, num_clients: int, hidden_dim: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_clients)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq, dim] or [batch, dim]
        Returns: logits [batch, num_clients]
        """
        if x.dim() == 3:
            # Mean pooling over sequence
            x = x.mean(dim=1)  # [batch, dim]
        return self.classifier(x)


class FedDisPModel(nn.Module):
    def __init__(self, model_config: ModelConfig, num_clients: int = 7):
        super().__init__()
        self.model_config = model_config
        self.num_clients = num_clients

        # Load base model
        self.base = AutoModelForSeq2SeqLM.from_pretrained(model_config.pretrained_model_path)
        self.base.config.max_length = model_config.max_seq_length

        # LoRA configuration
        use_lora = getattr(model_config, 'use_lora', False)
        self.use_lora = use_lora

        if use_lora:
            if not PEFT_AVAILABLE:
                raise RuntimeError("PEFT library required for LoRA. Install with: pip install peft")

            lora_r = getattr(model_config, 'lora_r', 8)
            lora_alpha = getattr(model_config, 'lora_alpha', 32)
            lora_dropout = getattr(model_config, 'lora_dropout', 0.1)

            # Apply LoRA to attention layers
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q", "v"],  # Query and Value projections
                modules_to_save=None,  # Don't save full modules, only LoRA weights
            )

            # Wrap base model with LoRA
            self.base = get_peft_model(self.base, lora_config)

            # Freeze non-LoRA parameters
            for name, param in self.base.named_parameters():
                if "lora_" not in name:
                    param.requires_grad = False

            print(f"LoRA enabled: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            self.base.print_trainable_parameters()

        # Only initialize two-stream components if enabled
        if model_config.enable_two_stream:
            # Get hidden size - handle both regular and PEFT-wrapped models
            if hasattr(self.base, 'config'):
                hidden_size = self.base.config.d_model
            else:
                hidden_size = self.base.base_model.config.d_model

            self.role_proj = nn.Linear(hidden_size, model_config.role_dim)
            self.style_proj = nn.Linear(hidden_size, model_config.style_dim)
            # Use residual connection to preserve information flow
            self.use_residual = getattr(model_config, 'use_residual', True)
            self.residual_mode = getattr(model_config, 'residual_mode', 'projected')

            # Fuse layer maps [role; style] back to hidden_size
            self.fuse = nn.Linear(model_config.role_dim + model_config.style_dim, hidden_size)

            if self.use_residual:
                # Learnable mixing weight for residual
                mix_init = getattr(model_config, 'mix_weight_init', 0.5)
                self.mix_weight = nn.Parameter(torch.tensor(mix_init))

                if self.residual_mode == 'gated':
                    # Gated residual: learnable gate to control info flow from hidden
                    # Gate is computed from fused representation
                    self.residual_gate = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.Sigmoid()
                    )
                elif self.residual_mode == 'projected':
                    # Projected residual: hidden goes through same bottleneck
                    # This ensures hidden info also passes through role/style separation
                    # We project hidden -> role_dim + style_dim -> hidden_size
                    # This creates an information bottleneck for the residual path
                    self.hidden_to_bottleneck = nn.Linear(hidden_size, model_config.role_dim + model_config.style_dim)
                    self.bottleneck_to_hidden = nn.Linear(model_config.role_dim + model_config.style_dim, hidden_size)
            self.dropout = nn.Dropout(model_config.dropout)

            # Client adversarial classifiers for privacy
            # 1. Role adversary: ensure role doesn't encode client info
            self.role_client_adversary = ClientAdversary(
                input_dim=model_config.role_dim,
                num_clients=num_clients,
                hidden_dim=getattr(model_config, 'adversary_hidden_dim', 128)
            )
            # 2. Fused adversary: ensure final fused representation doesn't leak client info
            # This is critical when using residual connections!
            self.fused_client_adversary = ClientAdversary(
                input_dim=hidden_size,
                num_clients=num_clients,
                hidden_dim=getattr(model_config, 'adversary_hidden_dim', 128)
            )

    def _get_encoder(self):
        """Get encoder, handling both regular and PEFT-wrapped models."""
        if self.use_lora:
            return self.base.base_model.encoder
        return self.base.encoder

    def _get_base_model(self):
        """Get base model for generation, handling both regular and PEFT-wrapped models."""
        if self.use_lora:
            return self.base
        return self.base

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        training_config: TrainingConfig,
        compute_loss: bool = True,
        client_id: Optional[int] = None,  # For adversarial training
    ) -> Tuple[Seq2SeqLMOutput, Dict[str, torch.Tensor]]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss_dict: Dict[str, torch.Tensor] = {}

        if self.model_config.enable_two_stream:
            # Two-stream path: role/style disentanglement
            encoder = self._get_encoder()
            encoder_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = encoder_outputs.last_hidden_state
            role = self.dropout(self.role_proj(hidden))
            style = self.dropout(self.style_proj(hidden))

            # NOTE: GRL is NOT applied here in fusion path anymore.
            # GRL only works when paired with a classifier that produces gradients.
            # The adversarial training (GRL + ClientAdversary) is applied separately below.
            # Here we just concatenate role and style for generation.
            fused = torch.cat([role, style], dim=-1)
            fused = self.fuse(fused)

            # Residual connection: preserve information flow
            # Different modes to balance generation quality vs privacy:
            # - 'full': direct residual (may leak client info)
            # - 'gated': learnable gate controls info flow
            # - 'projected': hidden passes through bottleneck (privacy-preserving)
            # - 'none': no residual (safest for privacy, may hurt generation)
            if self.use_residual:
                if self.residual_mode == 'full':
                    residual = hidden
                elif self.residual_mode == 'gated':
                    # Gated: learnable gate controls how much hidden info passes through
                    gate = self.residual_gate(fused)
                    residual = gate * hidden
                elif self.residual_mode == 'projected':
                    # Projected: hidden goes through information bottleneck
                    # This forces the residual to also pass through dimensionality reduction
                    # preventing raw hidden state from leaking client information
                    bottleneck = self.dropout(self.hidden_to_bottleneck(hidden))
                    residual = self.bottleneck_to_hidden(bottleneck)
                else:
                    # Unknown mode, default to no residual
                    residual = torch.zeros_like(fused)

                # Mix fused representation with residual
                fused = self.mix_weight * fused + (1 - self.mix_weight) * residual

            fused_outputs = BaseModelOutput(last_hidden_state=fused)

            # For PEFT models, we need to call differently
            if self.use_lora:
                outputs = self.base(
                    encoder_outputs=fused_outputs,
                    attention_mask=attention_mask,
                    labels=batch.get("labels"),
                )
            else:
                outputs = self.base(
                    encoder_outputs=fused_outputs,
                    attention_mask=attention_mask,
                    labels=batch.get("labels"),
                )

            # Expose role and style embeddings for downstream losses
            loss_dict["_role"] = role
            loss_dict["_style"] = style

            if compute_loss and batch.get("labels") is not None:
                seq_loss = outputs.loss
                loss_dict["seq2seq"] = seq_loss

                # Orthogonality: use weak constraint by default (constrain mean vectors only)
                # This allows local correlations while encouraging global separation
                use_weak_orth = getattr(training_config, 'use_weak_orth', True)
                orth = orthogonality_loss(role, style, weak=use_weak_orth)
                loss_dict["orth"] = orth * training_config.lambda_orth

                total_loss = seq_loss + loss_dict["orth"]

                # Client adversarial loss: make representations unable to predict client
                # Two-pronged approach:
                # 1. Role adversary: ensure role embedding doesn't encode client
                # 2. Fused adversary: ensure final output (with residual) doesn't leak client
                lambda_adv = getattr(training_config, 'lambda_adv', 0.0)
                if lambda_adv > 0.0 and client_id is not None:
                    batch_size = role.size(0)
                    client_labels = torch.full(
                        (batch_size,), client_id,
                        dtype=torch.long, device=role.device
                    )

                    # 1. Role adversary: prevent role from encoding client
                    role_with_grl = apply_grl(role, training_config.grl_lambda)
                    role_adv_logits = self.role_client_adversary(role_with_grl)
                    role_adv_loss = nn.functional.cross_entropy(role_adv_logits, client_labels)

                    # 2. Fused adversary: prevent final representation from leaking client
                    # This is CRITICAL when using residual connections!
                    # Even if role is clean, residual from hidden may leak info
                    fused_with_grl = apply_grl(fused, training_config.grl_lambda)
                    fused_adv_logits = self.fused_client_adversary(fused_with_grl)
                    fused_adv_loss = nn.functional.cross_entropy(fused_adv_logits, client_labels)

                    # Clamp individual adversarial losses to prevent explosion
                    # When adversary can't predict client, CE loss can explode
                    adv_loss_clamp = getattr(training_config, 'adv_loss_clamp', 5.0)
                    role_adv_loss = torch.clamp(role_adv_loss, max=adv_loss_clamp)
                    fused_adv_loss = torch.clamp(fused_adv_loss, max=adv_loss_clamp)

                    # Combined adversarial loss
                    # Weight fused_adv more when using residual (since that's where leakage occurs)
                    if self.use_residual and self.residual_mode != 'none':
                        adv_loss = 0.3 * role_adv_loss + 0.7 * fused_adv_loss
                    else:
                        adv_loss = role_adv_loss  # No residual, only role matters

                    loss_dict["adv_role"] = role_adv_loss * lambda_adv
                    loss_dict["adv_fused"] = fused_adv_loss * lambda_adv
                    loss_dict["adv"] = adv_loss * lambda_adv
                    total_loss = total_loss + loss_dict["adv"]

                    # Monitor both adversary accuracies
                    with torch.no_grad():
                        role_pred = role_adv_logits.argmax(dim=-1)
                        fused_pred = fused_adv_logits.argmax(dim=-1)
                        loss_dict["_adv_acc_role"] = (role_pred == client_labels).float().mean()
                        loss_dict["_adv_acc_fused"] = (fused_pred == client_labels).float().mean()
                        # Legacy: keep _adv_acc for backward compatibility
                        loss_dict["_adv_acc"] = loss_dict["_adv_acc_fused"]

                loss_dict["total"] = total_loss
                outputs.loss = total_loss
        else:
            # Baseline path: plain encoder-decoder (no role/style disentanglement)
            outputs = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch.get("labels"),
            )
            if compute_loss and batch.get("labels") is not None:
                seq_loss = outputs.loss
                loss_dict["seq2seq"] = seq_loss
                loss_dict["total"] = seq_loss
                outputs.loss = seq_loss

        return outputs, loss_dict

    @torch.no_grad()
    def encode_role(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoder = self._get_encoder()
        encoder_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = encoder_outputs.last_hidden_state
        return self.role_proj(hidden)

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **generate_kwargs):
        """
        Generate using the fused encoder (role + style) or raw encoder, depending on enable_two_stream.
        This ensures inference uses the same architecture as training.
        """
        if self.model_config.enable_two_stream:
            # Two-stream path: use role/style fusion
            encoder = self._get_encoder()
            encoder_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = encoder_outputs.last_hidden_state

            # Apply role/style projection and fusion (same as forward)
            role = self.role_proj(hidden)
            style = self.style_proj(hidden)

            # GRL is not applied during inference (no gradients)
            fused = torch.cat([role, style], dim=-1)
            fused = self.fuse(fused)

            # Apply residual connection in inference (same logic as forward)
            if self.use_residual:
                if self.residual_mode == 'full':
                    residual = hidden
                elif self.residual_mode == 'gated':
                    gate = self.residual_gate(fused)
                    residual = gate * hidden
                elif self.residual_mode == 'projected':
                    bottleneck = self.hidden_to_bottleneck(hidden)
                    residual = self.bottleneck_to_hidden(bottleneck)
                else:
                    residual = torch.zeros_like(fused)

                fused = self.mix_weight * fused + (1 - self.mix_weight) * residual

            fused_outputs = BaseModelOutput(last_hidden_state=fused)

            # Generate using fused encoder outputs
            base_model = self._get_base_model()
            return base_model.generate(
                encoder_outputs=fused_outputs,
                attention_mask=attention_mask,
                **generate_kwargs
            )
        else:
            # Baseline path: direct generation from base model
            base_model = self._get_base_model()
            return base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs
            )

    def get_trainable_state_dict(self, exclude_style: bool = True) -> Dict[str, torch.Tensor]:
        """
        Get only trainable parameters for communication-efficient federated learning.
        With LoRA, this is ~100x smaller than full model.

        Args:
            exclude_style: If True, exclude style_proj weights (keep style local for privacy).
                          This follows the Fed-Masks design where only Role is shared.
        """
        state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                # Skip style_proj to keep style representation local (privacy)
                if exclude_style and "style_proj" in name:
                    continue
                state_dict[name] = param.data.clone()
        return state_dict

    def load_trainable_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """
        Load only trainable parameters (for federated aggregation).
        """
        current_state = self.state_dict()
        for name, param in state_dict.items():
            if name in current_state:
                current_state[name].copy_(param)
            elif strict:
                raise KeyError(f"Unexpected key: {name}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def get_communication_size(model: nn.Module) -> float:
    """Calculate communication size in MB for trainable parameters."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = trainable_params * 4  # float32
    size_mb = size_bytes / (1024 * 1024)
    return size_mb
