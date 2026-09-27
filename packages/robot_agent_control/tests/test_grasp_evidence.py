import pytest

from robot_agent_control.skills.manipulation.grasp.validators.grasp_validator import (
    GraspExecutionValidationError,
    GraspValidator,
)
from robot_agent_control.skills.manipulation.grasp.utils.error_codes import ErrorCode


def _request():
    return {
        "target": {"type": "object", "object_id": "apple"},
        "grasp": {"verification_mode": "auto"},
        "constraints": {
            "verify_grasp": True,
            "contact_required": True,
            "maximum_force": 40.0,
            "slip_check": True,
        },
    }


def _raw(*, contact=False, present=False, holding=False, sides=()):
    return {
        "contact_detected": contact,
        "object_present": present,
        "holding": holding,
        "slip_detected": False,
        "final_width": 0.04,
        "final_force": 20.0,
        "execution_time": 0.1,
        "evidence": {"gripper_state": {"contact_sides": list(sides)}},
    }


def test_one_pad_contact_is_not_valid_grasp_evidence():
    with pytest.raises(GraspExecutionValidationError) as exc_info:
        GraspValidator().validate_result(_raw(contact=False, sides=("left",)), _request(), {})
    assert exc_info.value.error_code == ErrorCode.CONTACT_NOT_DETECTED
    assert "left" in str(exc_info.value)


def test_bilateral_contact_without_stable_hold_is_still_a_failure():
    with pytest.raises(GraspExecutionValidationError) as exc_info:
        GraspValidator().validate_result(_raw(contact=True, present=True, holding=False, sides=("left", "right")), _request(), {})
    assert exc_info.value.error_code == ErrorCode.GRASP_VERIFICATION_FAILED
    assert "stable holding" in str(exc_info.value)
