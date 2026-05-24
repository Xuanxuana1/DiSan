import json
import os
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn

from .config import PrototypeConfig


def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("No state dicts provided for averaging.")
    avg_state = {}
    keys = state_dicts[0].keys()
    for k in keys:
        tensors = [sd[k] for sd in state_dicts if k in sd]
        avg_state[k] = torch.mean(torch.stack(tensors, dim=0), dim=0)
    return avg_state


def filter_shared_weights(model: nn.Module, include_decoder: bool = True) -> Dict[str, torch.Tensor]:
    state = model.state_dict()
    shared = {}
    for name, tensor in state.items():
        if name.startswith("style_proj"):
            continue  # keep style local
        if not include_decoder and name.startswith("base.model.decoder"):
            continue
        shared[name] = tensor.cpu()
    return shared


def load_shared_weights(model: nn.Module, shared_state: Dict[str, torch.Tensor]):
    model_state = model.state_dict()
    for name, tensor in shared_state.items():
        if name in model_state:
            model_state[name] = tensor
    model.load_state_dict(model_state)


def compute_prototypes(
    role_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    entity_masks: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    role_embeddings: [batch, seq, dim]
    attention_mask: [batch, seq]
    entity_masks: dict[type] -> [batch, seq] binary masks
    """
    protos: Dict[str, torch.Tensor] = {}
    for ent_type, mask in entity_masks.items():
        if mask is None:
            continue
        mask = mask.unsqueeze(-1)  # [batch, seq,1]
        denom = mask.sum(dim=(0, 1)).clamp(min=1)
        summed = (role_embeddings * mask).sum(dim=(0, 1))
        protos[ent_type] = summed / denom
    return protos


def merge_prototypes(
    client_protos: List[Dict[str, torch.Tensor]], proto_config: PrototypeConfig
) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    all_keys = set(proto_config.entity_types)
    for proto in client_protos:
        all_keys.update(proto.keys())
    for ent in all_keys:
        tensors = [p[ent] for p in client_protos if ent in p]
        if tensors:
            merged[ent] = torch.mean(torch.stack(tensors, dim=0), dim=0)
    return merged


def save_payload(path: str, state: Dict[str, torch.Tensor], prototypes: Dict[str, torch.Tensor]):
    os.makedirs(path, exist_ok=True)
    torch.save(state, os.path.join(path, "shared_weights.pt"))
    torch.save(prototypes, os.path.join(path, "prototypes.pt"))
    meta = {"num_params": sum(t.numel() for t in state.values()), "num_prototypes": len(prototypes)}
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_payload(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    shared = torch.load(os.path.join(path, "shared_weights.pt"), map_location="cpu")
    protos = torch.load(os.path.join(path, "prototypes.pt"), map_location="cpu")
    return shared, protos

