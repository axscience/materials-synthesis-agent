from materials_synthesis_agent.nl.parser import estimate_parse_cost, parse_request


class FakeClient:
    """Implements the provider-agnostic LLMClient interface (llm.base.LLMClient) directly --
    parse_request() only ever calls .call_tool(), regardless of which of the 4 real providers is
    configured, so a single fake exercises the same code path all of them go through."""

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
    cost = parse_request("Synthesize an imine COF for CO2 capture, maximizing crystallinity.", dry_run=True)
    assert isinstance(cost, float)
    assert cost > 0


def test_estimate_parse_cost_scales_with_text_length():
    short = "Maximize crystallinity."
    long_text = short * 200
    assert estimate_parse_cost(long_text) > estimate_parse_cost(short)


def test_parse_request_reads_name_and_objective():
    tool_input = {
        "cof_name": "COF-5",
        "cif_path": None,
        "functional_groups": [],
        "linkage_chemistry": None,
        "application": None,
        "objectives": [
            {"name": "crystallinity", "measurement_method": "PXRD peak area ratio", "direction": "maximize"},
        ],
        "inferred_fields": [],
        "clarifications_needed": [],
    }
    parsed = parse_request("Tell me how to make COF-5, maximizing crystallinity via PXRD peak ratio.", client=FakeClient(tool_input))

    assert parsed.cof_name == "COF-5"
    assert parsed.cif_path is None
    assert len(parsed.objectives) == 1
    assert parsed.objectives[0].name == "crystallinity"
    assert parsed.objectives[0].maximize is True
    assert parsed.clarifications_needed == []


def test_parse_request_flags_inferred_fields_distinct_from_stated():
    tool_input = {
        "cof_name": None,
        "cif_path": None,
        "functional_groups": ["imine"],
        "linkage_chemistry": "imine condensation",
        "application": None,
        "objectives": [{"name": "crystallinity", "measurement_method": "PXRD", "direction": "maximize"}],
        "inferred_fields": ["objectives[0].direction"],
        "clarifications_needed": [],
    }
    parsed = parse_request("An imine COF, optimize crystallinity via PXRD.", client=FakeClient(tool_input))

    assert parsed.inferred_fields == ["objectives[0].direction"]


def test_parse_request_surfaces_clarifications_instead_of_guessing():
    tool_input = {
        "cof_name": None,
        "cif_path": None,
        "functional_groups": [],
        "linkage_chemistry": None,
        "application": None,
        "objectives": [],
        "inferred_fields": [],
        "clarifications_needed": ["What should I optimize for, and how would you measure it?"],
    }
    parsed = parse_request("I have a COF I want to make better.", client=FakeClient(tool_input))

    assert parsed.objectives == []
    assert parsed.clarifications_needed == ["What should I optimize for, and how would you measure it?"]


def test_parse_request_drops_and_flags_a_nonexistent_cif_path():
    tool_input = {
        "cof_name": None,
        "cif_path": "/definitely/not/a/real/path.cif",
        "functional_groups": [],
        "linkage_chemistry": None,
        "application": None,
        "objectives": [],
        "inferred_fields": [],
        "clarifications_needed": [],
    }
    parsed = parse_request("Here's a CIF at /definitely/not/a/real/path.cif", client=FakeClient(tool_input))

    assert parsed.cif_path is None
    assert any("couldn't find a file" in q for q in parsed.clarifications_needed)


def test_parse_request_keeps_a_real_cif_path(tmp_path):
    cif_file = tmp_path / "my_cof.cif"
    cif_file.write_text("data_my_cof\n")
    tool_input = {
        "cof_name": None,
        "cif_path": str(cif_file),
        "functional_groups": [],
        "linkage_chemistry": None,
        "application": None,
        "objectives": [],
        "inferred_fields": [],
        "clarifications_needed": [],
    }
    parsed = parse_request(f"Here's a CIF at {cif_file}", client=FakeClient(tool_input))

    assert parsed.cif_path == str(cif_file)
    assert parsed.clarifications_needed == []


def test_parse_request_calls_the_record_parsed_request_tool():
    client = FakeClient({
        "cof_name": None, "cif_path": None, "functional_groups": [], "linkage_chemistry": None,
        "application": None, "objectives": [], "inferred_fields": [], "clarifications_needed": [],
    })
    parse_request("some request", client=client)
    assert client.last_call_kwargs["tool_name"] == "record_parsed_request"
    assert "properties" in client.last_call_kwargs["tool_schema"]  # a real JSON Schema, not a stub
