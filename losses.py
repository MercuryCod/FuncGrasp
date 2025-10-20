"""
Loss functions for functional grasp prediction.

Implements:
1. ContactLoss: Multi-label BCE for finger-specific contact prediction
2. FlowMatchingLoss: MSE loss for conditional flow matching
3. compute_total_loss: Weighted combination of both losses
"""
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


class ContactLoss(nn.Module):
    """
    Multi-label Binary Cross-Entropy loss for contact prediction with soft targets.

    Semantics:
    - We train with INDEPENDENT probabilities for the 6 hand parts
      derived from distances via RBF: P = exp(-d/τ). These do NOT sum to 1.
    - Loss is applied to logits indices 1..6 (skip index 0 = no_contact).
    - Non-contact points are downweighted to address class imbalance.

    Label Encoding (from utils/contact_utils.py):
        0: no_contact, 1: palm, 2: thumb, 3: index, 4: middle, 5: ring, 6: pinky

    Args:
        no_contact_weight: Weight for non-contact points (default 0.1)
            Lower weight = less penalty for errors on non-contact points
            Typical range: 0.05-0.2
    """
    def __init__(self, no_contact_weight: float = 0.1):
        super().__init__()
        self.no_contact_weight = no_contact_weight

    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        """
        Compute BCE loss for contact prediction with soft probability targets.

        Args:
            logits: [B, N, 7] raw logits from contact head
                Index 0: no_contact logit (computed but UNUSED in loss)
                Index 1-6: palm, thumb, index, middle, ring, pinky (USED in loss)
            soft_targets: [B, N, 6] independent probability targets for 6 contact parts
                Computed from per-finger distances using RBF kernel: exp(-d/tau)
                CRITICAL: These are INDEPENDENT probabilities (do NOT sum to 1.0)
                Maps to logits[1:6] (class 0 = no_contact is implicit)

        Returns:
            Scalar loss value
        
        Note on Indexing:
            Dataset labels: [0, 1, 2, 3, 4, 5, 6] where 0=no_contact
            Model logits:   [0, 1, 2, 3, 4, 5, 6] where index 0 is unused
            Soft targets:   [0, 1, 2, 3, 4, 5] mapping to logits[1:6]
        """
        B, N, K = logits.shape
        assert K == 7, f"Expected 7 contact classes, got {K}"
        assert soft_targets.shape == (B, N, 6), f"Soft targets should be [B, N, 6], got {soft_targets.shape}"

        # Compute BCE loss on contact parts only (logits indices 1-6, skip index 0)
        # soft_targets are independent probabilities (from RBF kernel, NOT normalized)
        loss_per_point = F.binary_cross_entropy_with_logits(
            logits[..., 1:],  # [B, N, 6] - contact parts only (skip no_contact at index 0)
            soft_targets,      # [B, N, 6] - independent probability targets
            reduction='none'
        ).mean(dim=-1)  # [B, N] - average across 6 classes

        # Compute weights based on contact strength
        # Points with high contact probability (sum of probs) get higher weight
        contact_strength = soft_targets.sum(dim=-1)  # [B, N] - how "contact-y" is this point

        # Map to [no_contact_weight, 1.0] range and clamp for stability
        # High contact_strength → weight close to 1.0
        # Low contact_strength → weight close to no_contact_weight
        weights = contact_strength * (1.0 - self.no_contact_weight) + self.no_contact_weight
        weights = torch.clamp(weights, min=self.no_contact_weight, max=1.0)

        # Weighted mean: prioritizes points that are clearly in contact
        weighted_loss = (loss_per_point * weights).sum() / weights.sum()

        return weighted_loss


class FlowMatchingLoss(nn.Module):
    """
    Conditional Flow Matching loss for grasp pose generation.

    Implements the rectified flow objective where we learn a velocity field
    v(x_t, t, c) that transports noise (x_0) to data (x_1) along straight paths:

        x_t = (1-t) * x_0 + t * x_1        # Linear interpolation
        v_target = x_1 - x_0               # Target velocity (constant along path)
        loss = MSE(v_model, v_target)      # Match model prediction to target

    At inference time, we integrate the learned velocity field starting from noise:
        x_0 ~ N(0, I)                      # Sample from noise
        x_{t+dt} = x_t + dt * v(x_t, t, c) # Euler integration
        x_T → valid grasp pose             # After T steps

    Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2022)
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        model_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute MSE loss between model and target velocities.

        Args:
            model_velocity: [B, D] predicted velocity from flow network
            target_velocity: [B, D] target velocity (x1 - x0)
            t: [B] time values (not used in simple rectified flow, included for future extensions)

        Returns:
            Scalar MSE loss
        """
        return F.mse_loss(model_velocity, target_velocity)


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    contact_criterion: ContactLoss,
    flow_criterion: FlowMatchingLoss,
    lambda_contact: float = 1.0,
    lambda_flow: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Compute weighted sum of contact and flow matching losses.

    Total loss = λ_contact * L_contact + λ_flow * L_flow

    Args:
        outputs: Model outputs dict with:
            - 'logits_c': [B, N, 7] contact logits
            - 'model_velocity': [B, 61] predicted velocity (optional)
        targets: Ground truth dict with:
            - 'contact_soft_targets': [B, N, 6] soft probability distributions
            - 'target_velocity': [B, 61] target velocity (optional)
            - 't': [B] time values (optional)
        contact_criterion: ContactLoss instance
        flow_criterion: FlowMatchingLoss instance
        lambda_contact: Weight for contact loss (default 1.0)
        lambda_flow: Weight for flow matching loss (default 1.0)

    Returns:
        Dict with:
            - 'total': Total weighted loss (requires_grad=True)
            - 'contact': Contact loss component (detached)
            - 'flow': Flow loss component (detached)
    """
    # Contact loss (always computed)
    contact_loss = contact_criterion(
        outputs['logits_c'],
        targets['contact_soft_targets']  # Use soft targets from per-finger distances
    )

    # Flow matching loss (only if model_velocity is provided)
    flow_loss = torch.tensor(0.0, device=contact_loss.device)
    if 'model_velocity' in outputs and outputs['model_velocity'] is not None:
        flow_loss = flow_criterion(
            outputs['model_velocity'],
            targets['target_velocity'],
            targets.get('t', None)
        )

    # Total weighted loss
    total_loss = lambda_contact * contact_loss + lambda_flow * flow_loss

    return {
        'total': total_loss,               # Requires grad for backward
        'contact': contact_loss.detach(),  # For logging only
        'flow': flow_loss.detach()         # For logging only
    }

