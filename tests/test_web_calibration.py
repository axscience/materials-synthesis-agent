"""The calibration gate as it surfaces in the web layer: a blocking HTTP 409, not a warning banner
(CLAUDE.md guardrail). Builds a miscalibrated report directly so the test is fast and doesn't need a
scaffolded project/store -- the gate logic itself is covered in test_calibration.py."""

from materials_synthesis_agent.optimize import CalibrationReport
from materials_synthesis_agent.web.app import _calibration_blocked_page, _page


def _overconfident_report() -> CalibrationReport:
    return CalibrationReport(
        n=12,
        sufficient=True,
        passes=False,
        z_mean=0.10,
        z_std=1.80,
        coverage_80=0.45,
        coverage_90=0.55,
        mean_interval_width_80=3.2,
        rmse=5.0,
        variance_scale=3.24,  # sqrt -> 1.80x correction
        conformal_multiplier_80=2.1,
        conformal_multiplier_90=2.6,
        messages=["OVERCONFIDENT: intervals too tight (z_std=1.80)."],
    )


def test_page_status_code_passthrough():
    assert _page("t", "<p>x</p>", status_code=409).status_code == 409
    assert _page("t", "<p>x</p>").status_code == 200  # default unchanged


def test_calibration_blocked_page_is_409_with_report_and_recommended_fix():
    resp = _calibration_blocked_page(_overconfident_report())
    assert resp.status_code == 409
    body = resp.body.decode()
    assert "overconfident" in body.lower()
    assert "blocked" in body.lower()
    # The recommended correction is surfaced so the operator knows the fix, not just the failure.
    assert "variance_scale=3.24" in body
    assert "1.80" in body  # sqrt(3.24) correction factor
    # The full LOO diagnostics are shown.
    assert "coverage" in body.lower()
