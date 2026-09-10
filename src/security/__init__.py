"""Controlled security experiments and V0.1 integrity facade."""

from src.security.attack_generator import apply_attack, generate_attack, restore_original
from src.security.experiment_data import (
    DevelopmentExperimentData,
    PreparedAttackSplit,
    prepare_attack_split,
    prepare_development_experiment_data,
    prepare_test_experiment_data,
)
from src.security.ground_truth import assert_no_attack_metadata

__all__ = [
    "apply_attack",
    "assert_no_attack_metadata",
    "DevelopmentExperimentData",
    "generate_attack",
    "PreparedAttackSplit",
    "prepare_attack_split",
    "prepare_development_experiment_data",
    "prepare_test_experiment_data",
    "restore_original",
]
