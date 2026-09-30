"""Positive-unlabeled heatmap loss for partially annotated occupancy maps."""

import torch
from torch import nn
import torch.nn.functional as F


class PUGaussianMSE(nn.Module):
    """Non-negative PU risk using squared error on Gaussian heatmaps.

    Supplied annotations are trusted positives. Pixels outside their Gaussian
    support are treated as samples from an unlabeled positive/negative mixture,
    rather than as certain background. ``annotation_propensity`` is
    P(annotation observed | true positive). It is exactly ``1-drop_ratio`` for
    the controlled random-drop experiments used by this repository.

    This is a heatmap baseline for validating the partial-label hypothesis. It
    does not claim that spatial pixels are i.i.d.; the proposed sparse-query
    architecture will apply PU risk to object candidates instead.
    """

    def __init__(
        self,
        annotation_propensity=0.4,
        pos_thr=0.1,
        class_prior=None,
        beta=0.0,
        gamma=1.0,
        max_dynamic_prior=0.5,
    ):
        super().__init__()
        if not 0.0 < annotation_propensity <= 1.0:
            raise ValueError("annotation_propensity must be in (0, 1]")
        if not 0.0 < pos_thr <= 1.0:
            raise ValueError("pos_thr must be in (0, 1]")
        if class_prior is not None and not 0.0 < class_prior < 1.0:
            raise ValueError("class_prior must be in (0, 1)")
        if beta < 0.0 or gamma < 0.0:
            raise ValueError("beta and gamma must be non-negative")
        if not 0.0 < max_dynamic_prior < 1.0:
            raise ValueError("max_dynamic_prior must be in (0, 1)")
        self.annotation_propensity = float(annotation_propensity)
        self.pos_thr = float(pos_thr)
        self.class_prior = class_prior
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.max_dynamic_prior = float(max_dynamic_prior)
        self.last_components = {}

    def forward(self, prediction, target, kernel):
        soft_target = self._target_transform(prediction, target, kernel)

        # With exhaustive labels retain the original objective exactly. This
        # makes full-label experiments a controlled compatibility baseline.
        if self.annotation_propensity >= 1.0:
            loss = F.mse_loss(prediction, soft_target)
            self.last_components = {
                "positive_risk": float(loss.detach()),
                "negative_risk": 0.0,
                "class_prior": 1.0,
            }
            return loss

        positive_mask = soft_target >= self.pos_thr
        if not positive_mask.any():
            # With no observed positive, a PU batch cannot distinguish true
            # background from a completely unannotated positive frame.
            return prediction.sum() * 0.0

        positive_prediction = prediction[positive_mask]
        positive_target = soft_target[positive_mask]

        if self.class_prior is None:
            observed_fraction = positive_mask.to(prediction.dtype).mean()
            prior = (observed_fraction / self.annotation_propensity).clamp(
                min=observed_fraction.detach(), max=self.max_dynamic_prior
            )
        else:
            prior = prediction.new_tensor(self.class_prior)

        positive_risk = prior * F.mse_loss(
            positive_prediction, positive_target, reduction="mean"
        )
        negative_on_positive = prior * positive_prediction.pow(2).mean()
        # Single-training-set PU identity under SCAR:
        #   R = E[l_-] + E[S/c * (l_+ - l_-)]
        # The first expectation is over the complete observed marginal, not
        # only S=0 cells. Restricting it to the unmatched region changes the
        # class mixture and double-counts the selection mechanism.
        marginal_negative = prediction.pow(2).mean()
        negative_risk = marginal_negative - negative_on_positive

        # Kiryo et al.'s non-negative correction: if the empirical negative
        # risk crosses below -beta, reverse its gradient instead of letting a
        # flexible network overfit by driving the risk toward -infinity.
        if negative_risk.detach() < -self.beta:
            loss = positive_risk - self.gamma * negative_risk
        else:
            loss = positive_risk + negative_risk

        self.last_components = {
            "positive_risk": float(positive_risk.detach()),
            "negative_risk": float(negative_risk.detach()),
            "class_prior": float(prior.detach()),
        }
        return loss

    @staticmethod
    def _target_transform(prediction, target, kernel):
        target = F.adaptive_max_pool2d(target, prediction.shape[2:])
        with torch.no_grad():
            return F.conv2d(
                target,
                kernel.float().to(target.device),
                padding=int((kernel.shape[-1] - 1) / 2),
            )

    # Preserve the misspelled method used by existing visualization code.
    def _traget_transform(self, prediction, target, kernel):
        return self._target_transform(prediction, target, kernel)
