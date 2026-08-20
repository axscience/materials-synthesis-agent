from materials_synthesis_agent.literature.extraction import estimate_extraction_cost, extract_protocol
from materials_synthesis_agent.literature.retrieval import Paper
from materials_synthesis_agent.schema import Target


def make_target():
    return Target(
        functional_groups=["imine"],
        linkage_chemistry="imine condensation",
        application="CO2 capture",
        metric_name="crystallinity",
        metric_measurement_method="PXRD",
    )


def make_paper():
    return Paper(source_id="10.1234/fake", title="A fake COF synthesis paper", abstract="...", year=2024, url="http://x", source="semantic_scholar")


class FakeClient:
    """Implements the provider-agnostic LLMClient interface (llm.base.LLMClient) directly --
    extract_protocol() only ever calls .call_tool(), regardless of which of the 4 real providers
    is configured, so a single fake exercises the same code path all of them go through."""

    def __init__(self, tool_input):
        self._tool_input = tool_input
        self.last_call_kwargs = None

    def call_tool(self, prompt, tool_name, tool_description, tool_schema, max_tokens=2000):
        self.last_call_kwargs = {
            "prompt": prompt, "tool_name": tool_name, "tool_description": tool_description,
            "tool_schema": tool_schema, "max_tokens": max_tokens,
        }
        return self._tool_input


def test_dry_run_returns_cost_without_calling_anything():
    target, paper = make_target(), make_paper()
    cost = extract_protocol(target, paper, dry_run=True)
    assert isinstance(cost, float)
    assert cost > 0


def test_estimate_extraction_cost_scales_with_text_length():
    target = make_target()
    short = make_paper()
    long_paper = Paper(**{**short.__dict__, "abstract": short.abstract * 200})
    assert estimate_extraction_cost(target, long_paper) > estimate_extraction_cost(target, short)


def test_extract_protocol_grounds_cited_fields_and_flags_inferred():
    target, paper = make_target(), make_paper()
    tool_input = {
        "found_protocol": True,
        "building_blocks": {
            "TAPB": {"value": "SMILES1", "excerpt": "TAPB was used as the amine node", "inferred": False},
            "PDA": {"value": "SMILES2", "excerpt": None, "inferred": True},
        },
        "monomer_roles": {
            "TAPB": {"value": "node", "excerpt": "TAPB was used as the amine node", "inferred": False},
            "PDA": {"value": "linker", "excerpt": None, "inferred": True},
        },
        "stoichiometry": {},
        "synthesis_method": {"value": "solvothermal", "excerpt": "sealed Pyrex tube at 120 C for 72 h", "inferred": False},
        "solvent": {"value": "dioxane/mesitylene 1:1", "excerpt": "dioxane and mesitylene (1:1 v/v)", "inferred": False},
        "catalyst": None,
        "modulator": None,
        "temperature_c": {"value": "120", "excerpt": "heated at 120 C", "inferred": False},
        "time_hours": None,
        "concentration_molar": None,
        "atmosphere": {"value": "N2", "excerpt": "under nitrogen atmosphere", "inferred": False},
        "activation_method": {"value": "Soxhlet extraction with THF, then vacuum drying at 120 C", "excerpt": "Soxhlet-extracted with THF for 24 h, then dried under vacuum at 120 C", "inferred": False},
        "purification": None,
        "yield_percent": {"value": "85", "excerpt": "isolated in 85% yield", "inferred": False},
        "characterization_notes": None,
    }
    candidate = extract_protocol(target, paper, client=FakeClient(tool_input))

    assert candidate is not None
    assert candidate.building_blocks["TAPB"].citation is not None
    assert candidate.building_blocks["TAPB"].inferred is False
    assert candidate.building_blocks["PDA"].citation is None
    assert candidate.building_blocks["PDA"].inferred is True
    assert candidate.monomer_roles["TAPB"].value == "node"
    assert candidate.monomer_roles["PDA"].inferred is True
    assert candidate.synthesis_method.value == "solvothermal"
    assert candidate.solvent.citation.excerpt == "dioxane and mesitylene (1:1 v/v)"
    assert candidate.atmosphere.value == "N2"
    assert candidate.activation_method.citation is not None
    assert candidate.yield_percent.value == "85"
    assert candidate.catalyst is None
    assert candidate.modulator is None
    assert 0 < candidate.citation_coverage() <= 1


def test_extract_protocol_returns_none_when_paper_has_no_protocol():
    target, paper = make_target(), make_paper()
    tool_input = {"found_protocol": False}
    result = extract_protocol(target, paper, client=FakeClient(tool_input))
    assert result is None


def test_extract_protocol_calls_the_record_protocol_tool():
    target, paper = make_target(), make_paper()
    client = FakeClient({"found_protocol": False})
    extract_protocol(target, paper, client=client)
    assert client.last_call_kwargs["tool_name"] == "record_protocol"
    assert "properties" in client.last_call_kwargs["tool_schema"]  # a real JSON Schema, not a stub


def test_full_text_excerpt_replaces_the_abstract_in_the_prompt():
    target, paper = make_target(), make_paper()
    client = FakeClient({"found_protocol": False})
    extract_protocol(target, paper, client=client, full_text_excerpt="Synthesized using 5 mmol reagent A at 120 C.")
    prompt = client.last_call_kwargs["prompt"]
    assert "FULL-TEXT EXCERPT" in prompt
    assert "Supplementary Information" in prompt  # the prompt now tells the model SI is where numbers live
    assert "Synthesized using 5 mmol reagent A" in prompt
    assert paper.abstract not in prompt


def test_no_full_text_excerpt_still_uses_the_abstract():
    target, paper = make_target(), make_paper()
    client = FakeClient({"found_protocol": False})
    extract_protocol(target, paper, client=client)
    prompt = client.last_call_kwargs["prompt"]
    assert "ABSTRACT" in prompt
    assert paper.abstract in prompt


def test_estimate_extraction_cost_with_full_text_reflects_the_larger_prompt():
    target, paper = make_target(), make_paper()
    abstract_only = estimate_extraction_cost(target, paper)
    with_full_text = estimate_extraction_cost(target, paper, full_text_excerpt="x" * 5000)
    assert with_full_text > abstract_only
