from unittest.mock import patch

from materials_synthesis_agent.literature.agent import build_query, estimate_generation_cost, generate_protocols
from materials_synthesis_agent.literature.retrieval import Paper
from materials_synthesis_agent.schema import Target


def make_paper(**overrides):
    base = dict(source_id="10.1234/fake", title="A fake COF synthesis paper", abstract="...", year=2024, url="http://x", source="semantic_scholar", oa_pdf_url=None)
    base.update(overrides)
    return Paper(**base)


class FakeClient:
    """Implements the provider-agnostic LLMClient interface (llm.base.LLMClient) directly --
    extract_protocol() only ever calls .call_tool(), regardless of which of the 4 real providers
    is configured, so a single fake exercises the same code path all of them go through."""

    def __init__(self, tool_input):
        self._tool_input = tool_input
        self.last_call_kwargs = None

    def call_tool(self, prompt, tool_name, tool_description, tool_schema, max_tokens=2000):
        self.last_call_kwargs = {"prompt": prompt}
        return self._tool_input


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
    assert query == "COF-5 covalent organic framework synthesis"


def test_build_query_falls_back_to_linkage_when_no_name():
    # With no name, build_query returns the same-linkage tier's query: the base linkage ("imine",
    # normalized from "imine condensation"), plus functional groups and application, deduped.
    target = Target(
        functional_groups=["imine"],
        linkage_chemistry="imine condensation",
        application="CO2 capture",
        metric_name="crystallinity",
        metric_measurement_method="PXRD",
    )
    query = build_query(target)
    assert query == "imine covalent organic framework synthesis CO2 capture"


def _target():
    return Target(
        functional_groups=["imine"], linkage_chemistry="imine condensation", application="CO2 capture",
        metric_name="crystallinity", metric_measurement_method="PXRD",
    )


def test_generate_protocols_without_full_text_never_calls_fulltext_module():
    tool_input = {"found_protocol": False}
    client = FakeClient(tool_input)
    with patch("materials_synthesis_agent.literature.agent.search", return_value=[make_paper()]):
        with patch("materials_synthesis_agent.literature.fulltext.get_full_text_excerpt") as mock_excerpt:
            generate_protocols(_target(), n=1, client=client, use_full_text=False)
    mock_excerpt.assert_not_called()


def test_generate_protocols_with_full_text_resolves_an_excerpt_per_paper():
    tool_input = {"found_protocol": False}
    client = FakeClient(tool_input)
    with patch("materials_synthesis_agent.literature.agent.search", return_value=[make_paper()]):
        with patch("materials_synthesis_agent.literature.fulltext.get_full_text_excerpt", return_value="real excerpt text") as mock_excerpt:
            generate_protocols(_target(), n=1, client=client, use_full_text=True)
    mock_excerpt.assert_called_once()
    assert "real excerpt text" in client.last_call_kwargs["prompt"]


def test_generate_protocols_falls_back_to_abstract_when_no_excerpt_resolved():
    paper = make_paper(abstract="a real abstract sentence")
    client = FakeClient({"found_protocol": False})
    with patch("materials_synthesis_agent.literature.agent.search", return_value=[paper]):
        with patch("materials_synthesis_agent.literature.fulltext.get_full_text_excerpt", return_value=None):
            generate_protocols(_target(), n=1, client=client, use_full_text=True)
    assert "a real abstract sentence" in client.last_call_kwargs["prompt"]


def test_estimate_generation_cost_with_full_text_resolves_excerpts_too():
    paper = make_paper()
    with patch("materials_synthesis_agent.literature.fulltext.get_full_text_excerpt", return_value="x" * 5000) as mock_excerpt:
        cost_with_ft = estimate_generation_cost(_target(), [paper], use_full_text=True)
    mock_excerpt.assert_called_once()
    cost_without_ft = estimate_generation_cost(_target(), [paper], use_full_text=False)
    assert cost_with_ft > cost_without_ft
