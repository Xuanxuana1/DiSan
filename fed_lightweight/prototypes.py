from typing import Dict, List

import torch


# Full list of fine-grained PII labels observed in the dataset
PII_LABELS: List[str] = [
    "name",
    "first_name",
    "last_name",
    "company",
    "customer_id",
    "employee_id",
    "user_name",
    "street_address",
    "local_latlng",
    "email",
    "phone_number",
    "date",
    "date_of_birth",
    "date_time",
    "time",
    "ssn",
    "passport_number",
    "driver_license_number",
    "account_pin",
    "credit_card_number",
    "credit_card_security_code",
    "bank_routing_number",
    "iban",
    "bban",
    "swift_bic_code",
    "ipv4",
    "ipv6",
    "api_key",
    "password",
]

# Macro-type prototypes: group multiple fine-grained labels into a smaller set
MACRO_TYPES: List[str] = [
    "NAME",
    "ORG",
    "ID",
    "CONTACT",
    "ADDR",
    "TIME",
    "CARD",
    "NET",
    "CRED",
]

# Mapping from fine-grained label -> macro-type
LABEL_TO_MACRO: Dict[str, str] = {
    # names
    "name": "NAME",
    "first_name": "NAME",
    "last_name": "NAME",
    "user_name": "ID",
    # org / company
    "company": "ORG",
    # ids / identifiers
    "customer_id": "ID",
    "employee_id": "ID",
    "ssn": "ID",
    "passport_number": "ID",
    "driver_license_number": "ID",
    "account_pin": "ID",
    "api_key": "ID",
    "password": "ID",
    # contact channels
    "email": "CONTACT",
    "phone_number": "CONTACT",
    # address / location
    "street_address": "ADDR",
    "local_latlng": "ADDR",
    # time / date
    "date": "TIME",
    "date_of_birth": "TIME",
    "date_time": "TIME",
    "time": "TIME",
    # card / bank
    "credit_card_number": "CARD",
    "credit_card_security_code": "CARD",
    "bank_routing_number": "CARD",
    "iban": "CARD",
    "bban": "CARD",
    "swift_bic_code": "CARD",
    # network
    "ipv4": "NET",
    "ipv6": "NET",
}

# Indices are defined over macro-types, not fine-grained labels
MACRO_TO_INDEX: Dict[str, int] = {m: i for i, m in enumerate(MACRO_TYPES)}



def empty_proto_accumulators(dim: int, device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Create zeroed accumulators for prototypes on macro-types.
    Returns a dict: macro_type -> {"sum": tensor[dim], "count": scalar tensor}
    """
    acc: Dict[str, Dict[str, torch.Tensor]] = {}
    for m in MACRO_TYPES:
        acc[m] = {
            "sum": torch.zeros(dim, device=device, dtype=torch.float32),
            "count": torch.zeros(1, device=device, dtype=torch.float32),
        }
    return acc


def finalize_prototypes(acc: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Turn accumulators into mean prototypes. Macro-types with zero count are skipped.
    """
    protos: Dict[str, torch.Tensor] = {}
    for m, d in acc.items():
        cnt = d["count"].item()
        if cnt > 0:
            protos[m] = d["sum"] / d["count"].clamp(min=1.0)
    return protos



