"""Unit tests for the selective trajectory-replacement gate, using hand-
constructed risk/deviation values so every branch of the two-condition gate
is exercised explicitly."""
import torch

from src.trajectory_selection import select_trajectory

TAU_RISK = 0.5
TAU_DEV = 2.0


def test_no_replacement_when_nominal_is_safe():
    # Nominal risk below tau_risk -> must never replace, even if a candidate
    # looks great.
    nominal_risk = torch.tensor([0.1])
    candidate_risk = torch.tensor([[0.01, 0.02]])
    candidate_dev = torch.tensor([[0.5, 0.5]])
    result = select_trajectory(nominal_risk, candidate_risk, candidate_dev, TAU_RISK, TAU_DEV)
    assert result.replaced[0].item() is False


def test_no_replacement_when_no_candidate_qualifies_on_risk():
    # Nominal unsafe, but every candidate is also >= tau_risk -> no replacement.
    nominal_risk = torch.tensor([0.9])
    candidate_risk = torch.tensor([[0.6, 0.8]])
    candidate_dev = torch.tensor([[0.1, 0.1]])
    result = select_trajectory(nominal_risk, candidate_risk, candidate_dev, TAU_RISK, TAU_DEV)
    assert result.replaced[0].item() is False


def test_no_replacement_when_only_safe_candidate_deviates_too_much():
    # Nominal unsafe, one candidate has low risk but deviates past tau_dev ->
    # no replacement (component-wise constraint fails on deviation).
    nominal_risk = torch.tensor([0.9])
    candidate_risk = torch.tensor([[0.1, 0.9]])
    candidate_dev = torch.tensor([[5.0, 0.1]])  # low-risk candidate deviates too much
    result = select_trajectory(nominal_risk, candidate_risk, candidate_dev, TAU_RISK, TAU_DEV)
    assert result.replaced[0].item() is False


def test_replacement_when_both_conditions_hold():
    nominal_risk = torch.tensor([0.9])
    candidate_risk = torch.tensor([[0.9, 0.1, 0.3]])
    candidate_dev = torch.tensor([[0.1, 1.0, 1.5]])
    result = select_trajectory(nominal_risk, candidate_risk, candidate_dev, TAU_RISK, TAU_DEV)
    assert result.replaced[0].item() is True
    # candidate 1 (risk=0.1, dev=1.0) is the lowest-risk qualifying candidate
    assert result.selected_index[0].item() == 1


def test_batched_mixed_decisions():
    nominal_risk = torch.tensor([0.1, 0.9, 0.9])
    candidate_risk = torch.tensor([
        [0.05, 0.05],   # scene 0: nominal already safe -> ignored regardless
        [0.6, 0.7],     # scene 1: nominal unsafe, no candidate qualifies on risk
        [0.2, 0.4],     # scene 2: nominal unsafe, candidate 0 qualifies
    ])
    candidate_dev = torch.tensor([
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    result = select_trajectory(nominal_risk, candidate_risk, candidate_dev, TAU_RISK, TAU_DEV)
    assert result.replaced.tolist() == [False, False, True]
    assert result.selected_index[2].item() == 0
    assert result.selected_index[0].item() == -1
    assert result.selected_index[1].item() == -1
