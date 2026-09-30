"""Partial-label set loss for continuous PUMA-MV queries."""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class PUQuerySetLoss(nn.Module):
    """Hungarian positive matching without treating unknown queries as negatives.

    Under full annotation, every unmatched query receives ordinary no-object
    BCE. Under partial annotation, unmatched queries form the unlabeled mixture
    used by a non-negative PU risk estimator. Coordinate regression is applied
    only to observed positives.
    """

    def __init__(
        self,
        annotation_propensity=0.4,
        coordinate_weight=1.0,
        beta=0.0,
        gamma=1.0,
        max_dynamic_prior=0.95,
        separation_weight=0.1,
        minimum_separation_m=0.25,
        propensity_mode="scar",
        propensity_regularization=0.1,
        minimum_propensity=0.05,
        minimum_sigma_m=0.05,
        coordinate_student_dof=3.0,
    ):
        super().__init__()
        if not 0.0 < annotation_propensity <= 1.0:
            raise ValueError("annotation_propensity must be in (0,1]")
        if not 0.0 < max_dynamic_prior <= 1.0:
            raise ValueError("max_dynamic_prior must be in (0,1]")
        if (coordinate_weight < 0 or beta < 0 or gamma < 0
                or separation_weight < 0 or minimum_separation_m <= 0
                or propensity_regularization < 0):
            raise ValueError("loss weights and beta must be non-negative")
        if propensity_mode not in ("scar", "sar"):
            raise ValueError("propensity_mode must be 'scar' or 'sar'")
        if not 0.0 < minimum_propensity < 1.0:
            raise ValueError("minimum_propensity must be in (0,1)")
        if minimum_sigma_m <= 0:
            raise ValueError("minimum_sigma_m must be positive")
        if coordinate_student_dof <= 0:
            raise ValueError("coordinate_student_dof must be positive")
        self.annotation_propensity = float(annotation_propensity)
        self.coordinate_weight = float(coordinate_weight)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.max_dynamic_prior = float(max_dynamic_prior)
        self.separation_weight = float(separation_weight)
        self.minimum_separation_m = float(minimum_separation_m)
        self.propensity_mode = propensity_mode
        self.propensity_regularization = float(propensity_regularization)
        self.minimum_propensity = float(minimum_propensity)
        self.minimum_sigma_m = float(minimum_sigma_m)
        self.coordinate_student_dof = float(coordinate_student_dof)
        self.last_components = {}

    @staticmethod
    def _targets_for_batch(map_target, bev_xy_m):
        occupied = (map_target > 0).nonzero(as_tuple=False)
        if occupied.numel() == 0:
            return bev_xy_m.new_zeros((0, 2))
        return bev_xy_m[occupied[:, -2], occupied[:, -1]]

    def forward(self, queries, map_target, bev_xy_m):
        xy = queries["xy_m"]
        logits = queries["objectness"]
        if xy.ndim != 3 or xy.shape[-1] != 2 or logits.shape != xy.shape[:2]:
            raise ValueError("queries must contain xy_m [B,K,2] and objectness [B,K]")
        if map_target.ndim == 3:
            map_target = map_target.unsqueeze(1)
        if map_target.ndim != 4 or map_target.shape[0] != xy.shape[0]:
            raise ValueError("map_target must have shape [B,1,H,W]")
        if bev_xy_m.ndim != 3 or bev_xy_m.shape[-1] != 2:
            raise ValueError("bev_xy_m must have shape [H,W,2]")
        bev_xy_m = bev_xy_m.to(xy)
        map_target = map_target.to(xy.device, non_blocking=True)
        if self.propensity_mode == "sar" and "propensity_logit" not in queries:
            raise ValueError("SAR mode requires queries['propensity_logit']")

        losses = []
        positive_terms, negative_terms, coordinate_terms, separation_terms = [], [], [], []
        for batch_index in range(xy.shape[0]):
            target_xy = self._targets_for_batch(map_target[batch_index, 0], bev_xy_m)
            # An empty partially annotated frame is not evidence of an empty
            # scene. It therefore contributes no query classification loss.
            if target_xy.numel() == 0 and self.annotation_propensity < 1.0:
                losses.append(logits[batch_index].sum() * 0.0)
                continue

            num_queries = xy.shape[1]
            matched_query = torch.zeros(num_queries, dtype=torch.bool, device=xy.device)
            coordinate_loss = xy.new_zeros(())
            if target_xy.numel() > 0:
                cost = torch.cdist(xy[batch_index], target_xy)
                row, col = linear_sum_assignment(cost.detach().cpu().numpy())
                row = torch.as_tensor(row, device=xy.device, dtype=torch.long)
                col = torch.as_tensor(col, device=xy.device, dtype=torch.long)
                matched_query[row] = True
                coordinate_error = xy[batch_index, row] - target_xy[col]
                if "log_sigma_m" in queries:
                    minimum_log_sigma = math.log(self.minimum_sigma_m)
                    log_sigma = queries["log_sigma_m"][batch_index, row].clamp(
                        min=minimum_log_sigma, max=math.log(2.0)
                    )
                    # Robust Student-t NLL retains covariance calibration but
                    # grows logarithmically for far-away warm-up proposals.
                    dof = self.coordinate_student_dof
                    normalized_square = coordinate_error.pow(2) * torch.exp(
                        -2.0 * log_sigma
                    )
                    coordinate_loss = (
                        log_sigma - minimum_log_sigma
                        + 0.5 * (dof + 1.0) * torch.log1p(
                            normalized_square / dof
                        )
                    ).mean()
                else:
                    coordinate_loss = F.smooth_l1_loss(
                        xy[batch_index, row], target_xy[col], reduction="mean"
                    )

            batch_logits = logits[batch_index]
            positive_loss = (
                F.binary_cross_entropy_with_logits(
                    batch_logits[matched_query], torch.ones_like(batch_logits[matched_query])
                )
                if matched_query.any()
                else batch_logits.sum() * 0.0
            )
            unmatched = ~matched_query
            if self.annotation_propensity >= 1.0:
                negative_loss = (
                    F.binary_cross_entropy_with_logits(
                        batch_logits[unmatched], torch.zeros_like(batch_logits[unmatched])
                    )
                    if unmatched.any()
                    else batch_logits.sum() * 0.0
                )
                classification_loss = positive_loss + negative_loss
            elif self.propensity_mode == "sar":
                propensity = self.minimum_propensity + (1.0 - self.minimum_propensity) * (
                    queries["propensity_logit"][batch_index].sigmoid()
                )
                object_probability = batch_logits.sigmoid()
                selection_probability = (object_probability * propensity).clamp(1e-6, 1 - 1e-6)
                observed_selection = (
                    -selection_probability[matched_query].log().mean()
                    if matched_query.any()
                    else batch_logits.sum() * 0.0
                )
                unlabeled_selection = (
                    -(1.0 - selection_probability[unmatched]).log().mean()
                    if unmatched.any()
                    else batch_logits.sum() * 0.0
                )
                # Propensity is defined for positives. Weight the rate anchor
                # by stopped-gradient object probability to avoid background
                # queries dominating this identifiability constraint.
                probability_weight = object_probability.detach()
                mean_propensity = (
                    probability_weight * propensity
                ).sum() / probability_weight.sum().clamp_min(1e-6)
                rate_regularizer = (
                    mean_propensity - self.annotation_propensity
                ).pow(2)
                classification_loss = (
                    observed_selection + unlabeled_selection
                    + self.propensity_regularization * rate_regularizer
                )
                positive_loss = observed_selection
                negative_loss = unlabeled_selection
            else:
                observed_fraction = matched_query.to(xy.dtype).mean()
                prior = (observed_fraction / self.annotation_propensity).clamp(
                    min=observed_fraction.detach(), max=self.max_dynamic_prior
                )
                negative_on_positive = (
                    F.binary_cross_entropy_with_logits(
                        batch_logits[matched_query], torch.zeros_like(batch_logits[matched_query])
                    )
                    if matched_query.any()
                    else batch_logits.sum() * 0.0
                )
                # Propensity-corrected single-set PU risk. The marginal
                # negative term includes every query; using unmatched queries
                # alone conditions on S=0 and produces a biased mixture.
                negative_marginal = F.binary_cross_entropy_with_logits(
                    batch_logits, torch.zeros_like(batch_logits)
                )
                negative_risk = negative_marginal - prior * negative_on_positive
                if negative_risk.detach() < -self.beta:
                    classification_loss = prior * positive_loss - self.gamma * negative_risk
                else:
                    classification_loss = prior * positive_loss + negative_risk
                negative_loss = negative_risk

            loss = classification_loss + self.coordinate_weight * coordinate_loss
            if self.separation_weight > 0 and num_queries > 1:
                pair_distance = torch.cdist(
                    xy[batch_index].unsqueeze(0), xy[batch_index].unsqueeze(0)
                ).squeeze(0)
                pair_mask = torch.triu(
                    torch.ones_like(pair_distance, dtype=torch.bool), diagonal=1
                )
                confidence = batch_logits.sigmoid().detach()
                pair_weight = confidence[:, None] * confidence[None, :]
                overlap = (self.minimum_separation_m - pair_distance).clamp_min(0.0)
                separation_loss = (
                    overlap[pair_mask].pow(2) * pair_weight[pair_mask]
                ).sum() / pair_weight[pair_mask].sum().clamp_min(1e-6)
                loss = loss + self.separation_weight * separation_loss
            else:
                separation_loss = xy[batch_index].sum() * 0.0
            losses.append(loss)
            positive_terms.append(positive_loss.detach())
            negative_terms.append(negative_loss.detach())
            coordinate_terms.append(coordinate_loss.detach())
            separation_terms.append(separation_loss.detach())

        total = torch.stack(losses).mean()
        zero = total.detach().new_zeros(())
        self.last_components = {
            "positive": float(torch.stack(positive_terms).mean() if positive_terms else zero),
            "negative": float(torch.stack(negative_terms).mean() if negative_terms else zero),
            "coordinate": float(torch.stack(coordinate_terms).mean() if coordinate_terms else zero),
            "separation": float(torch.stack(separation_terms).mean() if separation_terms else zero),
        }
        return total
