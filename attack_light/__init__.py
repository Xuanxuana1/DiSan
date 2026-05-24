"""
Attack experiments for FedDisP privacy evaluation (LoRA-compatible version).

EXP-1: Representation Probe - Does role/style embedding leak client info?
EXP-2: Prototype Attack - Can attacker infer client from role prototypes?
EXP-3: Stylometry Attack - Can attacker infer client from sanitized evidence?

Modified for fed_lightweight LoRA models.
"""

from .data_utils import prepare_attack_data, create_matched_test_set
from .attackers import TFIDFAttacker, NeuralAttacker, EmbeddingProbe
from .exp1_representation_probe import run_exp2_probe
from .exp2_prototype_attack import run_exp3_prototype
from .exp3a_text_stylometry import run_text_stylometry_experiment

__all__ = [
    'prepare_attack_data',
    'create_matched_test_set',
    'TFIDFAttacker',
    'NeuralAttacker',
    'EmbeddingProbe',
    'run_exp2_probe',
    'run_exp3_prototype',
    'run_text_stylometry_experiment',
]
