"""
Federated Training with Server-side Prototype Adversarial Training (Parallel Version).

This version supports parallel client training on multiple GPUs.
Each client trains on a separate GPU simultaneously.
"""

import argparse
import os
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import queue

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig

from .config import ModelConfig, TrainingConfig
from .data import ClientDataLoaders, JsonlSeq2SeqDataset, _collate_pad_with_gliner
from .federated import average_state_dicts
from .model import FedDisPModel, ClientAdversary
from .losses import (
    apply_grl,
    prototype_alignment_loss,
    prototype_confusion_loss,
    RunningPrototype,
    spherical_alignment_loss,
    spherical_confusion_loss,
    SphericalRunningPrototype,
)


@dataclass
class ClientPayload:
    """What each client uploads to server after local training."""
    client_name: str
    client_id: int
    model_state: Dict[str, torch.Tensor]
    role_prototype: torch.Tensor
    num_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel Federated training")
    parser.add_argument("--clients", type=str, nargs="+", required=True)
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="../checkpoints/fed_disp_federated")
    parser.add_argument("--devices", type=int, nargs="+", default=[4, 5, 6], help="GPU IDs to use")
    parser.add_argument("--lr", type=float, default=1e-4)  # Lower default lr to prevent instability
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--local_steps", type=int, default=200)
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--lambda_orth", type=float, default=0.2)
    parser.add_argument("--grl_lambda", type=float, default=0.3, help="GRL lambda for gradient reversal (increased from 0.1)")
    parser.add_argument("--server_adv_steps", type=int, default=30, help="Server adversary training steps per round (reduced from 100)")
    parser.add_argument("--fedprox_mu", type=float, default=0.1, help="FedProx proximal term coefficient (0.1 for better generation quality)")
    parser.add_argument("--warmup_steps", type=int, default=20, help="LR warmup steps after aggregation")
    parser.add_argument("--residual_mode", type=str, default="projected")
    parser.add_argument("--gliner_model_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_two_stream", action="store_true", default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--adv_loss_clamp", type=float, default=5.0, help="Clamp adversarial loss to prevent explosion")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name for output directory")
    parser.add_argument("--min_gpu_memory_gb", type=float, default=22.0, help="Min free GPU memory (GB) to start a client")
    # Privacy loss parameters
    parser.add_argument("--lambda_priv_start", type=float, default=0.5, help="Initial privacy loss weight")
    parser.add_argument("--lambda_priv_end", type=float, default=3.0, help="Final privacy loss weight")
    parser.add_argument("--lambda_priv_warmup_steps", type=int, default=100, help="Steps to warmup privacy loss")
    # Prototype alignment for privacy protection
    parser.add_argument("--lambda_proto_align", type=float, default=1.0, help="Weight for prototype alignment loss")
    parser.add_argument("--proto_align_warmup_steps", type=int, default=50, help="Warmup steps before prototype alignment")
    parser.add_argument("--proto_align_start_round", type=int, default=2, help="Start prototype alignment from this round (1-indexed)")
    # Prototype-level adversarial training
    parser.add_argument("--lambda_proto_adv", type=float, default=0.2, help="Weight for prototype-level adversarial loss (reduced from 0.5)")
    parser.add_argument("--proto_adv_warmup_steps", type=int, default=100, help="Warmup steps for prototype adversarial loss")
    # Spherical Uniform Alignment - against KNN and LogReg attacks
    parser.add_argument("--use_spherical_align", action="store_true", default=False,
                       help="Use spherical alignment (stronger against KNN/LogReg attacks)")
    parser.add_argument("--lambda_sphere_direction", type=float, default=1.0,
                       help="Weight for direction alignment in spherical loss")
    parser.add_argument("--lambda_sphere_dispersion", type=float, default=0.1,
                       help="Weight for dispersion penalty (prevents Bootstrap attack)")
    parser.add_argument("--proto_noise_scale", type=float, default=0.0,
                       help="Gaussian noise scale to add to prototype before upload (noise perturbation defense, not true DP)")
    # LoRA for communication-efficient federated learning
    parser.add_argument("--use_lora", action="store_true", default=False,
                       help="Use LoRA for efficient fine-tuning (reduces communication ~100x)")
    parser.add_argument("--lora_r", type=int, default=8,
                       help="LoRA rank (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha scaling factor (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout (default: 0.1)")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def privacy_mass_loss(logits: torch.Tensor, vocab_mask: torch.Tensor, decoder_attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Privacy mass loss: penalize probability mass on PII tokens.

    Args:
        logits: Model output logits [B, T, V]
        vocab_mask: Mask indicating PII tokens in vocabulary [B, V]
        decoder_attention_mask: Attention mask for decoder [B, T]

    Returns:
        Scalar loss value
    """
    probs = torch.softmax(logits, dim=-1)  # [B, T, V]
    pii_mass = (probs * vocab_mask[:, None, :]).sum(dim=-1)  # [B, T]
    if decoder_attention_mask is not None:
        mask = decoder_attention_mask.float()
        denom = mask.sum()
        if denom.item() == 0:
            return pii_mass.mean()
        return (pii_mass * mask).sum() / denom
    return pii_mass.mean()


def client_training_worker(
    client_name: str,
    client_id: int,
    gpu_id: int,
    data_files: List[str],
    global_state_path: str,  # Path to global state file
    server_adv_state_path: str,  # Path to adversary state file
    output_path: str,  # Path to save client output
    model_config: ModelConfig,
    training_config: TrainingConfig,
    num_steps: int,
    num_clients: int,
    batch_size: int,
    fedprox_mu: float = 0.1,  # FedProx proximal term coefficient
    warmup_steps: int = 20,  # Warmup steps after aggregation
    lambda_priv_start: float = 0.5,  # Privacy loss initial weight
    lambda_priv_end: float = 3.0,  # Privacy loss final weight
    lambda_priv_warmup_steps: int = 1000,  # Privacy loss warmup steps
    global_client_proto_path: Optional[str] = None,  # Path to global average prototype
    lambda_proto_align: float = 1.0,  # Prototype alignment loss weight
    proto_align_warmup_steps: int = 50,  # Warmup steps for prototype alignment
    lambda_proto_adv: float = 0.2,  # Prototype-level adversarial loss weight (reduced)
    proto_adv_warmup_steps: int = 100,  # Warmup steps for prototype adversarial loss
    use_spherical_align: bool = False,  # Use spherical alignment (against KNN/LogReg attacks)
    lambda_sphere_direction: float = 1.0,  # Direction alignment weight
    lambda_sphere_dispersion: float = 0.1,  # Dispersion penalty weight
    proto_noise_scale: float = 0.0,  # Noise scale added to prototype before upload
):
    """
    Worker function for parallel client training.
    Each worker runs on a separate GPU.
    Uses files instead of Queue to avoid multiprocessing issues with large tensors.
    """
    import sys
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)

        print(f"[{client_name}] Starting on GPU {gpu_id}", flush=True)
        sys.stdout.flush()

        # Load global state and adversary state from files
        global_state = torch.load(global_state_path, map_location='cpu')
        server_adv_state = torch.load(server_adv_state_path, map_location='cpu') if os.path.exists(server_adv_state_path) else None

        # Load global client prototype for alignment (if available)
        global_client_proto = None
        if global_client_proto_path and os.path.exists(global_client_proto_path):
            global_client_proto = torch.load(global_client_proto_path, map_location=device)
            print(f"[{client_name}] Loaded global prototype for alignment", flush=True)

        # Initialize running prototype tracker
        # Always initialize for prototype-level adversarial training (even without global proto)
        # Choose prototype tracker based on whether spherical alignment is used
        running_proto = None
        if model_config.enable_two_stream:
            if use_spherical_align:
                # Spherical prototype tracker: always keeps prototype normalized
                running_proto = SphericalRunningPrototype(
                    dim=model_config.role_dim,
                    momentum=0.99,
                    device=str(device)
                )
                print(f"[{client_name}] Using SPHERICAL prototype alignment (anti-KNN/LogReg)", flush=True)
            else:
                running_proto = RunningPrototype(
                    dim=model_config.role_dim,
                    momentum=0.99,
                    device=str(device)
                )

        # Load tokenizer and create dataset
        tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_path, use_fast=True)
        tokenizer.model_max_length = model_config.max_seq_length
        hf_config = AutoConfig.from_pretrained(model_config.pretrained_model_path)

        dataset = JsonlSeq2SeqDataset(
            data_files, tokenizer, model_config,
            src_field="original_text", tgt_field="rewritten_text"
        )

        collate = _collate_pad_with_gliner(
            tokenizer, None,
            max_source_length=model_config.max_seq_length,
            max_target_length=model_config.max_seq_length,
            vocab_size=hf_config.vocab_size,
        )

        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
        )

        # Create model and load global weights (only trainable parameters)
        model = FedDisPModel(model_config, num_clients).to(device)
        model.load_trainable_state_dict(global_state, strict=False)

        # Load server's adversary
        if hasattr(model, 'role_client_adversary') and server_adv_state:
            model.role_client_adversary.load_state_dict(server_adv_state)
            for p in model.role_client_adversary.parameters():
                p.requires_grad = False

        # Training
        model.train()

        # Keep a copy of global model parameters for FedProx
        global_params = {name: param.clone().detach() for name, param in model.named_parameters()}

        # Use lower learning rate with warmup
        base_lr = training_config.lr
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr * 0.1,  # Start with lower lr
            weight_decay=training_config.weight_decay
        )

        # Learning rate scheduler with warmup
        def get_lr_multiplier(step):
            if step < warmup_steps:
                return 0.1 + 0.9 * (step / warmup_steps)  # Warmup from 0.1 to 1.0
            return 1.0

        step = 0
        epoch = 0
        role_embeddings = []
        num_samples = 0

        while step < num_steps:
            epoch += 1
            for batch in train_loader:
                if step >= num_steps:
                    break
                step += 1

                # Update learning rate with warmup
                lr_mult = get_lr_multiplier(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = base_lr * lr_mult

                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                outputs, loss_dict = model(batch, training_config, client_id=client_id)
                loss = loss_dict.get("total", outputs.loss)

                # FedProx: Add proximal term to prevent client drift
                if fedprox_mu > 0:
                    proximal_term = 0.0
                    for name, param in model.named_parameters():
                        if name in global_params:
                            proximal_term += ((param - global_params[name]) ** 2).sum()
                    loss = loss + (fedprox_mu / 2) * proximal_term

                # Privacy loss: penalize probability mass on PII tokens
                loss_priv = torch.tensor(0.0, device=device)
                logits = outputs.logits
                vocab_mask = batch.get("vocab_mask")
                dec_attn = batch.get("decoder_attention_mask")
                if vocab_mask is not None and logits is not None:
                    # Warmup privacy loss weight
                    lambda_priv = lambda_priv_start
                    if lambda_priv_warmup_steps > 0:
                        progress = min(step / lambda_priv_warmup_steps, 1.0)
                        lambda_priv = lambda_priv_start + progress * (lambda_priv_end - lambda_priv_start)
                    loss_priv = privacy_mass_loss(logits, vocab_mask, dec_attn) * lambda_priv
                    loss = loss + loss_priv

                # Update running prototype (always, for prototype-level adversarial training)
                if running_proto is not None and "_role" in loss_dict:
                    role = loss_dict["_role"]  # [B, T, d_role]
                    attn_mask = batch["attention_mask"]  # [B, T]
                    mask_expanded = attn_mask.unsqueeze(-1).float()
                    batch_proto = (role * mask_expanded).sum(dim=[0, 1]) / mask_expanded.sum().clamp(min=1.0)
                    running_proto.update(batch_proto)

                # Prototype alignment loss (only when global prototype is available)
                # Choose alignment method based on use_spherical_align
                loss_proto_align = torch.tensor(0.0, device=device)
                if (
                    global_client_proto is not None
                    and "_role" in loss_dict
                    and lambda_proto_align > 0.0
                ):
                    role = loss_dict["_role"]
                    attn_mask = batch["attention_mask"]

                    # Compute alignment loss with warmup
                    if running_proto is not None and running_proto.get() is not None and step >= proto_align_warmup_steps:
                        warmup_progress = min(
                            (step - proto_align_warmup_steps) / max(proto_align_warmup_steps, 1),
                            1.0
                        )
                        lambda_align = lambda_proto_align * warmup_progress

                        if use_spherical_align:
                            # Spherical confusion loss: stronger privacy protection against KNN and LogReg attacks
                            # 1. Normalize prototype to unit sphere, eliminating magnitude information
                            # 2. Align direction so that cos(p_i, p_j) → 1
                            # 3. Reduce dispersion to prevent Bootstrap attack from exploiting distribution differences
                            loss_proto_align = spherical_confusion_loss(
                                role, global_client_proto, attn_mask,
                                lambda_centroid=lambda_sphere_direction,
                                lambda_dispersion=lambda_sphere_dispersion,
                            )
                        else:
                            # Original prototype confusion loss
                            loss_proto_align = prototype_confusion_loss(
                                role, global_client_proto, attn_mask
                            )
                        loss = loss + lambda_align * loss_proto_align

                # Prototype-level adversarial training
                # KEY: Use running_proto (EMA) instead of batch_proto for more stable training
                # Running proto approximates the true client prototype over many batches
                loss_proto_adv = torch.tensor(0.0, device=device)
                if (
                    "_role" in loss_dict
                    and lambda_proto_adv > 0.0
                    and hasattr(model, 'role_client_adversary')
                    and server_adv_state is not None
                ):
                    role = loss_dict["_role"]  # [B, T, d_role]
                    attn_mask = batch["attention_mask"]

                    # Compute batch prototype
                    mask_exp = attn_mask.unsqueeze(-1).float()
                    batch_proto_for_adv = (role * mask_exp).sum(dim=[0, 1]) / mask_exp.sum().clamp(min=1.0)

                    # Use running prototype if available (more stable, closer to true client proto)
                    # Otherwise fall back to batch prototype
                    if running_proto is not None and running_proto.get() is not None:
                        # Detach running proto and add current batch contribution with gradient
                        # This allows gradient to flow through current batch while using stable estimate
                        alpha = 0.1  # Weight for current batch contribution
                        proto_for_adv = (1 - alpha) * running_proto.get().detach() + alpha * batch_proto_for_adv
                    else:
                        proto_for_adv = batch_proto_for_adv

                    # Apply GRL: reverse gradient to fool adversary
                    grl_lambda = training_config.grl_lambda
                    proto_grl = apply_grl(proto_for_adv, grl_lambda)

                    # Predict client ID from prototype (adversary tries to identify client)
                    proto_adv_logits = model.role_client_adversary(proto_grl.unsqueeze(0))  # [1, num_clients]
                    client_label = torch.tensor([client_id], device=device)

                    # Adversarial loss: model learns to make prototype unidentifiable
                    loss_proto_adv = nn.functional.cross_entropy(proto_adv_logits, client_label)
                    loss_proto_adv = torch.clamp(loss_proto_adv, max=training_config.adv_loss_clamp)

                    # Apply warmup for prototype adversarial loss to prevent sudden spikes
                    if step < proto_adv_warmup_steps:
                        warmup_factor = step / max(proto_adv_warmup_steps, 1)
                    else:
                        warmup_factor = 1.0
                    loss = loss + warmup_factor * lambda_proto_adv * loss_proto_adv

                if "_role" in loss_dict:
                    role = loss_dict["_role"]
                    role_mean = role.mean(dim=1).detach()
                    role_embeddings.append(role_mean)
                    num_samples += role_mean.size(0)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                if step % 50 == 0:
                    # Log individual loss components
                    l_seq = loss_dict.get("seq2seq", torch.tensor(0.0)).item()
                    l_orth = loss_dict.get("orth", torch.tensor(0.0)).item()
                    l_priv = loss_priv.item() if torch.is_tensor(loss_priv) else loss_priv
                    l_align = loss_proto_align.item() if torch.is_tensor(loss_proto_align) else loss_proto_align
                    l_proto_adv = loss_proto_adv.item() if torch.is_tensor(loss_proto_adv) else loss_proto_adv
                    # Compute similarity to global prototype
                    proto_sim = 0.0
                    if running_proto is not None and running_proto.get() is not None and global_client_proto is not None:
                        proto_sim = torch.nn.functional.cosine_similarity(
                            running_proto.get().unsqueeze(0),
                            global_client_proto.unsqueeze(0)
                        ).item()
                    print(f"[{client_name}][GPU{gpu_id}] step={step}/{num_steps} "
                          f"loss={loss.item():.4f} l_seq={l_seq:.4f} l_orth={l_orth:.4f} "
                          f"l_priv={l_priv:.4f} l_align={l_align:.4f} l_p_adv={l_proto_adv:.4f} p_sim={proto_sim:.4f} "
                          f"lr={lr_mult*base_lr:.2e}", flush=True)

        # Compute prototype for upload to server
        # Use running_proto (EMA, normalized) for consistency with training alignment
        # This prevents prototype attacks from exploiting magnitude/distribution info
        if running_proto is not None and running_proto.get() is not None:
            # SphericalRunningPrototype returns normalized prototype
            prototype = running_proto.get().cpu()
        elif role_embeddings:
            # Fallback: normalize the mean of collected embeddings
            all_roles = torch.cat(role_embeddings, dim=0)
            prototype = all_roles.mean(dim=0)
            prototype = F.normalize(prototype, dim=0).cpu()
        else:
            prototype = torch.zeros(model_config.role_dim)

        # Add Gaussian noise perturbation defense (not true DP, no formal privacy guarantee)
        # Theoretical basis: noise magnitude σ√d ≈ inter-prototype distance makes KNN/LogReg attacks fail
        # With σ=0.01, d=256: noise angular perturbation ≈ 9°, comparable to client separation ≈ 12°
        if proto_noise_scale > 0:
            noise = torch.randn_like(prototype) * proto_noise_scale
            prototype = prototype + noise
            # Re-normalize to stay on unit sphere
            prototype = F.normalize(prototype, dim=0)

        # Move only trainable parameters to CPU (LoRA weights + heads)
        # This reduces communication from ~1.2GB to ~5MB with LoRA
        model_state = {k: v.cpu() for k, v in model.get_trainable_state_dict().items()}

        # Save output to file
        output_data = {
            'client_name': client_name,
            'client_id': client_id,
            'model_state': model_state,
            'role_prototype': prototype,
            'num_samples': num_samples,
            'success': True
        }
        torch.save(output_data, output_path)
        print(f"[{client_name}] Training complete, samples={num_samples}", flush=True)

    except Exception as e:
        print(f"[{client_name}] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Save error state
        torch.save({'success': False, 'client_name': client_name}, output_path)


class FederatedServer:
    """Server for federated learning with prototype-based adversarial training."""

    def __init__(self, model_config: ModelConfig, num_clients: int, device: torch.device):
        self.device = device
        self.num_clients = num_clients
        self.model_config = model_config
        self.global_model: Optional[FedDisPModel] = None

        self.prototype_adversary = ClientAdversary(
            input_dim=model_config.role_dim,
            num_clients=num_clients,
            hidden_dim=128
        ).to(device)

        self.adv_optimizer = torch.optim.Adam(
            self.prototype_adversary.parameters(), lr=1e-3
        )

    def initialize_global_model(self):
        self.global_model = FedDisPModel(self.model_config, self.num_clients).to(self.device)
        print(f"[Server] Global model initialized")

    def aggregate_models(self, payloads: List[ClientPayload]) -> Dict[str, torch.Tensor]:
        if not payloads:
            return self.global_model.get_trainable_state_dict()

        total_samples = sum(p.num_samples for p in payloads)
        weights = [p.num_samples / total_samples for p in payloads]

        avg_state = {}
        for key in payloads[0].model_state.keys():
            avg_state[key] = sum(
                w * p.model_state[key].float() for w, p in zip(weights, payloads)
            )

        return avg_state

    def train_adversary_on_prototypes(
        self,
        payloads: List[ClientPayload],
        num_steps: int,
        early_stop_acc: float = 0.85,  # Stop if accuracy exceeds this threshold
        early_stop_patience: int = 5,  # Stop if no improvement for this many steps
    ) -> float:
        """
        Train adversary on client prototypes with early stopping.

        Early stopping prevents the adversary from becoming too strong,
        which would make the adversarial loss saturate at the clamp value.
        """
        if len(payloads) < 2:
            return 0.0

        prototypes = torch.stack([p.role_prototype.to(self.device) for p in payloads])
        labels = torch.tensor([p.client_id for p in payloads], device=self.device)

        self.prototype_adversary.train()
        total_loss = 0.0
        actual_steps = 0
        best_loss = float('inf')
        patience_counter = 0

        for step in range(num_steps):
            self.adv_optimizer.zero_grad()
            logits = self.prototype_adversary(prototypes)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            self.adv_optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss
            actual_steps += 1

            # Early stopping check every step
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == labels).float().mean().item()

            # Stop if accuracy is too high (adversary too strong)
            if acc >= early_stop_acc:
                print(f"[Server] Adversary early stop: acc={acc:.3f} >= {early_stop_acc} at step {step+1}")
                break

            # Stop if loss plateaus (adversary converged)
            if current_loss < best_loss - 1e-4:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"[Server] Adversary early stop: loss plateaued at step {step+1}")
                    break

        with torch.no_grad():
            preds = self.prototype_adversary(prototypes).argmax(dim=-1)
            acc = (preds == labels).float().mean().item()

        avg_loss = total_loss / max(actual_steps, 1)
        print(f"[Server] Adversary: loss={avg_loss:.4f}, acc={acc:.3f} (steps={actual_steps}/{num_steps})")
        return avg_loss

    def get_adversary_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.prototype_adversary.state_dict().items()}

    def compute_global_client_prototype(self, payloads: List[ClientPayload]) -> Optional[torch.Tensor]:
        """
        Compute global average of client prototypes.
        This is used for prototype alignment to make clients indistinguishable.
        """
        if not payloads:
            return None

        # Weighted average by number of samples
        total_samples = sum(p.num_samples for p in payloads)
        if total_samples == 0:
            # Simple average if no sample counts
            prototypes = torch.stack([p.role_prototype for p in payloads])
            global_proto = prototypes.mean(dim=0)
        else:
            weights = [p.num_samples / total_samples for p in payloads]
            global_proto = sum(
                w * p.role_prototype for w, p in zip(weights, payloads)
            )

        # Print similarity analysis
        print(f"\n[Server] Computing global prototype from {len(payloads)} clients")
        print("[Server] Client prototype similarities to global average:")
        for p in payloads:
            sim = torch.nn.functional.cosine_similarity(
                p.role_prototype.unsqueeze(0),
                global_proto.unsqueeze(0)
            ).item()
            print(f"  {p.client_name}: {sim:.4f}")

        return global_proto


def get_client_data_files(client: str, data_root: str) -> List[str]:
    """Get training files for a client."""
    files = []
    client_root = os.path.join(data_root, client)

    for d in [client_root, os.path.join(client_root, "train")]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".jsonl"):
                    files.append(os.path.join(d, f))

    if not files and os.path.isdir(data_root):
        for f in os.listdir(data_root):
            if f.endswith(".jsonl") and f.startswith(f"{client}_"):
                files.append(os.path.join(data_root, f))

    return files


def get_gpu_free_memory_torch(logical_gpu_id: int) -> float:
    """Get free memory in GB for a logical GPU using PyTorch."""
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(logical_gpu_id)
        return free_bytes / (1024 ** 3)  # Convert to GB
    except Exception as e:
        print(f"Warning: Could not query GPU {logical_gpu_id} memory: {e}")
    return 0.0


def find_available_gpu(logical_devices: List[int], busy_gpus: set, min_free_gb: float = 22.0) -> Optional[int]:
    """Find a logical GPU with at least min_free_gb of free memory that is not busy."""
    for logical_id in logical_devices:
        if logical_id in busy_gpus:
            continue  # Skip GPUs already running a client
        free_gb = get_gpu_free_memory_torch(logical_id)
        if free_gb >= min_free_gb:
            return logical_id
    return None


def federated_training_parallel(
    clients: List[str],
    data_root: str,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    devices: List[int],
    num_rounds: int,
    local_steps: int,
    server_adv_steps: int,
    output_dir: str,
    batch_size: int,
    fedprox_mu: float = 0.1,
    warmup_steps: int = 20,
    min_gpu_memory_gb: float = 22.0,
    lambda_priv_start: float = 0.5,
    lambda_priv_end: float = 3.0,
    lambda_priv_warmup_steps: int = 1000,
    lambda_proto_align: float = 1.0,
    proto_align_warmup_steps: int = 50,
    proto_align_start_round: int = 2,
    lambda_proto_adv: float = 0.2,
    proto_adv_warmup_steps: int = 100,
    use_spherical_align: bool = False,
    lambda_sphere_direction: float = 1.0,
    lambda_sphere_dispersion: float = 0.1,
    proto_noise_scale: float = 0.0,
):
    """Main federated training loop with parallel client training."""
    mp.set_start_method('spawn', force=True)

    num_clients = len(clients)
    client_to_id = {c: i for i, c in enumerate(clients)}

    # Get data files for each client
    client_data_files = {c: get_client_data_files(c, data_root) for c in clients}
    valid_clients = [c for c in clients if client_data_files[c]]

    # When CUDA_VISIBLE_DEVICES is set, PyTorch sees logical GPUs 0, 1, 2, ...
    # The --devices argument is only used to determine how many GPUs to use
    num_available_gpus = torch.cuda.device_count()
    logical_devices = list(range(num_available_gpus))  # [0, 1, 2] when 3 GPUs visible

    print(f"=== Parallel Federated Training ===")
    print(f"Clients: {valid_clients}")
    print(f"GPUs: {num_available_gpus} available")
    print(f"Rounds: {num_rounds}, Local steps: {local_steps}")

    # Server uses first available GPU (logical index 0)
    server_device = torch.device(f"cuda:0")
    server = FederatedServer(model_config, num_clients, server_device)
    server.initialize_global_model()

    os.makedirs(output_dir, exist_ok=True)

    for round_idx in range(num_rounds):
        print(f"\n{'='*50}")
        print(f"Round {round_idx + 1}/{num_rounds}")
        print(f"{'='*50}")

        # Save global state and adversary state to temp files
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        global_state_path = os.path.join(temp_dir, f"round_{round_idx}_global.pt")
        server_adv_path = os.path.join(temp_dir, f"round_{round_idx}_adv.pt")

        torch.save({k: v.cpu() for k, v in server.global_model.get_trainable_state_dict().items()}, global_state_path)
        torch.save(server.get_adversary_state(), server_adv_path)

        # Check if we should use prototype alignment this round
        # Prototype alignment starts from proto_align_start_round (1-indexed)
        use_proto_align = (round_idx + 1) >= proto_align_start_round
        global_client_proto_path = None
        if use_proto_align:
            # Use prototype from previous round (if it exists)
            prev_proto_path = os.path.join(output_dir, f"round_{round_idx}", "global_client_proto.pt")
            if os.path.exists(prev_proto_path):
                global_client_proto_path = prev_proto_path
                print(f"[Server] Using global prototype from round {round_idx} for alignment")
            else:
                print(f"[Server] No global prototype available yet, skipping alignment this round")

        # Parallel client training - batch clients to fit available GPUs
        # If more clients than GPUs, run in batches to avoid OOM
        output_paths = []
        num_gpus = len(logical_devices)

        # Split clients into batches based on available GPUs
        client_batches = []
        for i in range(0, len(valid_clients), num_gpus):
            client_batches.append(valid_clients[i:i + num_gpus])

        print(f"  Training {len(valid_clients)} clients in {len(client_batches)} batch(es) on {num_gpus} GPUs")

        for batch_idx, client_batch in enumerate(client_batches):
            if len(client_batches) > 1:
                print(f"  Batch {batch_idx + 1}/{len(client_batches)}: {[c.split('_')[-1] for c in client_batch]}")

            processes = []
            batch_output_paths = []

            for i, client in enumerate(client_batch):
                gpu_id = logical_devices[i]  # Each client in batch gets its own GPU
                client_id = client_to_id[client]
                output_path = os.path.join(temp_dir, f"round_{round_idx}_{client}.pt")
                output_paths.append(output_path)
                batch_output_paths.append(output_path)

                p = mp.Process(
                    target=client_training_worker,
                    args=(
                        client,
                        client_id,
                        gpu_id,
                        client_data_files[client],
                        global_state_path,
                        server_adv_path,
                        output_path,
                        model_config,
                        training_config,
                        local_steps,
                        num_clients,
                        batch_size,
                        fedprox_mu,
                        warmup_steps,
                        lambda_priv_start,
                        lambda_priv_end,
                        lambda_priv_warmup_steps,
                        global_client_proto_path,  # Pass global prototype path
                        lambda_proto_align if use_proto_align else 0.0,  # Only enable if this round uses alignment
                        proto_align_warmup_steps,
                        lambda_proto_adv,  # Prototype-level adversarial training weight
                        proto_adv_warmup_steps,  # Warmup steps for prototype adversarial loss
                        use_spherical_align,  # Spherical alignment switch
                        lambda_sphere_direction,  # Direction alignment weight
                        lambda_sphere_dispersion,  # Dispersion penalty weight
                        proto_noise_scale,  # Noise scale for prototype upload
                    )
                )
                p.start()
                processes.append(p)

            # Wait for this batch to complete before starting next batch
            for p in processes:
                p.join()

        # Load results from files
        payloads = []
        for output_path in output_paths:
            if os.path.exists(output_path):
                data = torch.load(output_path, map_location='cpu')
                if data.get('success', False):
                    payload = ClientPayload(
                        client_name=data['client_name'],
                        client_id=data['client_id'],
                        model_state=data['model_state'],
                        role_prototype=data['role_prototype'],
                        num_samples=data['num_samples']
                    )
                    payloads.append(payload)
                # Clean up temp file
                os.remove(output_path)

        # Clean up temp global files
        if os.path.exists(global_state_path):
            os.remove(global_state_path)
        if os.path.exists(server_adv_path):
            os.remove(server_adv_path)

        print(f"\n[Server] Received {len(payloads)}/{len(valid_clients)} client updates")

        # Aggregate models (only trainable parameters)
        if payloads:
            aggregated_state = server.aggregate_models(payloads)
            server.global_model.load_trainable_state_dict(aggregated_state, strict=False)

            # Train adversary on prototypes
            server.train_adversary_on_prototypes(payloads, server_adv_steps)

            # Compute global client prototype for next round's alignment
            global_client_proto = server.compute_global_client_prototype(payloads)

        # Save checkpoint (only trainable parameters for lightweight storage)
        round_dir = os.path.join(output_dir, f"round_{round_idx + 1}")
        os.makedirs(round_dir, exist_ok=True)
        torch.save(server.global_model.get_trainable_state_dict(), os.path.join(round_dir, "global_model.pt"))
        torch.save(server.get_adversary_state(), os.path.join(round_dir, "adversary.pt"))

        if payloads:
            protos = {p.client_name: p.role_prototype for p in payloads}
            torch.save(protos, os.path.join(round_dir, "prototypes.pt"))

            # Save global client prototype for next round's alignment
            if global_client_proto is not None:
                torch.save(global_client_proto, os.path.join(round_dir, "global_client_proto.pt"))
                print(f"[Server] Saved global client prototype for round {round_idx + 2} alignment")

    # Save final model (only trainable parameters)
    torch.save(server.global_model.get_trainable_state_dict(), os.path.join(output_dir, "final_model.pt"))
    print(f"\n=== Training complete ===")


def main():
    args = parse_args()
    set_seed(args.seed)

    model_config = ModelConfig()
    model_config.enable_two_stream = args.enable_two_stream
    model_config.residual_mode = args.residual_mode
    model_config.num_clients = len(args.clients)
    # LoRA configuration
    model_config.use_lora = args.use_lora
    model_config.lora_r = args.lora_r
    model_config.lora_alpha = args.lora_alpha
    model_config.lora_dropout = args.lora_dropout

    training_config = TrainingConfig()
    training_config.lr = args.lr
    training_config.lambda_orth = args.lambda_orth
    training_config.lambda_adv = 0.0  # Disabled - only using prototype-level adversarial training
    training_config.grl_lambda = args.grl_lambda
    training_config.batch_size = args.batch_size
    training_config.max_grad_norm = args.max_grad_norm
    training_config.adv_loss_clamp = args.adv_loss_clamp

    # Set output directory with experiment name if provided
    output_dir = args.output_dir
    if args.exp_name:
        output_dir = os.path.join(args.output_dir, args.exp_name)

    federated_training_parallel(
        clients=args.clients,
        data_root=args.data_root,
        model_config=model_config,
        training_config=training_config,
        devices=args.devices,
        num_rounds=args.num_rounds,
        local_steps=args.local_steps,
        server_adv_steps=args.server_adv_steps,
        output_dir=output_dir,
        batch_size=args.batch_size,
        fedprox_mu=args.fedprox_mu,
        warmup_steps=args.warmup_steps,
        min_gpu_memory_gb=args.min_gpu_memory_gb,
        lambda_priv_start=args.lambda_priv_start,
        lambda_priv_end=args.lambda_priv_end,
        lambda_priv_warmup_steps=args.lambda_priv_warmup_steps,
        lambda_proto_align=args.lambda_proto_align,
        proto_align_warmup_steps=args.proto_align_warmup_steps,
        proto_align_start_round=args.proto_align_start_round,
        lambda_proto_adv=args.lambda_proto_adv,
        proto_adv_warmup_steps=args.proto_adv_warmup_steps,
        use_spherical_align=args.use_spherical_align,
        lambda_sphere_direction=args.lambda_sphere_direction,
        lambda_sphere_dispersion=args.lambda_sphere_dispersion,
        proto_noise_scale=args.proto_noise_scale,
    )


if __name__ == "__main__":
    main()
