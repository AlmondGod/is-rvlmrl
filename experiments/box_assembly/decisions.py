"""Conservative decision contract using observable evidence, not simulator truth.

This is a baseline for testing decisions, not a trained perception/controller stack.
An absent measurement is unknown. Acceptance must be calibrated outside this module.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Evidence:
    alignment_error_m: float | None = None
    head_protrusion_m: float | None = None
    visible_split: bool | None = None
    nose_contact: bool = False
    driver_stroke_m: float = 0.
    clamp_history_verified: bool = False
    elapsed_since_glue_s: float = 0.
    # From an actual prescribed probe/load test, NOT the model's latent bond strength.
    proof_test_passed: bool | None = None
    measurements_credible: bool = False


    def __post_init__(self):
        for value in (self.alignment_error_m,self.head_protrusion_m,self.driver_stroke_m,self.elapsed_since_glue_s):
            if value is not None and (not math.isfinite(value) or value<0):
                raise ValueError("Measurements must be finite and nonnegative, or unknown (None)")


@dataclass(frozen=True)
class DecisionLimits:
    validated_for_material_and_tool: bool = False
    alignment_tolerance_m: float = .001
    seating_tolerance_m: float = .0005
    minimum_handling_s: float | None = None


    def __post_init__(self):
        for value in (self.alignment_tolerance_m,self.seating_tolerance_m,self.minimum_handling_s):
            if value is not None and (not math.isfinite(value) or value<=0):
                raise ValueError("Decision limits must be finite and positive")


def decide(e:Evidence,limits=DecisionLimits()):
    if e.visible_split is True or e.proof_test_passed is False:
        return {'action':'reject_or_repair_offline','reason':'Observed damage or failed proof test; do not erase failure state.'}
    if not e.measurements_credible:
        return {'action':'inspect','reason':'Obtain credible alignment, nail-head and tool observations.'}
    if e.alignment_error_m is None or e.visible_split is None:
        return {'action':'inspect','reason':'Alignment/damage status is unknown.'}
    if e.alignment_error_m>limits.alignment_tolerance_m:
        return {'action':'realign_before_fastening' if e.driver_stroke_m==0 else 'reject_or_repair_offline',
                'reason':'Alignment out of tolerance; a driven joint cannot be reset by software.'}
    if e.driver_stroke_m==0:
        return {'action':'drive' if e.nose_contact else 'establish_nose_contact','reason':'Aligned joint; tool contact required before driving.'}
    if e.head_protrusion_m is None:
        return {'action':'inspect','reason':'Driver travel alone does not establish nail seating.'}
    if e.head_protrusion_m>limits.seating_tolerance_m:
        return {'action':'inspect_for_redrive','reason':'Proud head; establish intact nail/wood and remaining tool capacity before retry.'}
    if not limits.validated_for_material_and_tool or limits.minimum_handling_s is None:
        return {'action':'hold_for_calibration','reason':'Handling time and proof load have not been validated for these materials.'}
    if not e.clamp_history_verified:
        return {'action':'reject_or_repair_offline','reason':'Required adhesive process history is not established.'}
    if e.elapsed_since_glue_s<limits.minimum_handling_s:
        return {'action':'wait','reason':'Validated handling interval has not elapsed.'}
    if e.proof_test_passed is not True:
        return {'action':'probe_with_support','reason':'Perform the validated inspection/proof-load procedure before release.'}
    return {'action':'release_support','reason':'Observed seating, verified process history and prescribed proof test passed.'}
