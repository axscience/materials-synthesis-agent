from materials_synthesis_agent.literature.agent import build_query
from materials_synthesis_agent.schema import Target


def test_build_query_prefers_name_when_set():
    target = Target(
        name="COF-5",
        functional_groups=["imine"],
        linkage_chemistry="imine condensation",
        application="CO2 capture",
        metric_name="crystallinity",
        metric_measurement_method="PXRD",
    )
    query = build_query(target)
    assert query == "COF-5 synthesis"


def test_build_query_falls_back_to_functional_groups_when_no_name():
    target = Target(
        functional_groups=["imine"],
        linkage_chemistry="imine condensation",
        application="CO2 capture",
        metric_name="crystallinity",
        metric_measurement_method="PXRD",
    )
    query = build_query(target)
    assert query == "imine condensation imine synthesis CO2 capture"
