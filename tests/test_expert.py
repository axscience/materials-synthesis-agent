from materials_synthesis_agent.expert import ExpertAgent, ExpertPlan, PlanStep
from materials_synthesis_agent.expert.prompts import SYSTEM_PROMPT, build_context_block
from materials_synthesis_agent.optimize.single_objective import Observation
from materials_synthesis_agent.schema import Target, ProtocolCandidate, ProtocolSource, FieldValue, Citation, MeasuredOutcome


class FakeChatClient:
    """Implements both call_tool and chat for testing the expert agent."""

    def __init__(self, responses=None):
        self._responses = list(responses or ["I'm the expert."])
        self._call_count = 0
        self.last_messages = None
        self.last_system = None

    def call_tool(self, prompt, tool_name, tool_description, tool_schema, max_tokens=2000):
        return {}

    def chat(self, messages, system=None, max_tokens=4000):
        self.last_messages = messages
        self.last_system = system
        response = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return response


def _target():
    return Target(
        name="COF-LZU1",
        functional_groups=["amine", "aldehyde"],
        linkage_chemistry="imine",
        application="gas separation",
        metric_name="crystallinity",
        metric_measurement_method="PXRD peak area ratio",
    )


def _protocol():
    return ProtocolCandidate(
        target_id="test-id",
        source=ProtocolSource.LITERATURE,
        building_blocks={"TFB": FieldValue(value="O=Cc1cc(C=O)cc(C=O)c1", citation=Citation(source_id="10.1234/test", title="Test", excerpt="TFB"), inferred=False)},
        solvent=FieldValue(value="dioxane/mesitylene 1:1", citation=Citation(source_id="10.1234/test", title="Test", excerpt="dioxane"), inferred=False),
        temperature_c=FieldValue(value="120", citation=Citation(source_id="10.1234/test", title="Test", excerpt="120 C"), inferred=False),
        measured_outcomes=[
            MeasuredOutcome(metric_name="BET_surface_area", value=410.0, unit="m²/g", measurement_method="N₂ adsorption at 77 K", citation=Citation(source_id="10.1234/test", title="Test", excerpt="410 m²/g"), inferred=False),
            MeasuredOutcome(metric_name="crystallinity", value=0.72, unit="ratio", measurement_method="PXRD peak area ratio", inferred=True),
        ],
    )


def test_expert_agent_sends_system_prompt():
    client = FakeChatClient(["Hello, I can help with COF synthesis."])
    agent = ExpertAgent(client)
    agent.respond("How do I synthesize COF-5?")
    assert client.last_system == SYSTEM_PROMPT


def test_expert_agent_maintains_conversation_history():
    client = FakeChatClient(["First response", "Second response"])
    agent = ExpertAgent(client)
    agent.respond("Question 1")
    agent.respond("Question 2")
    assert len(agent.history) == 4  # 2 user + 2 assistant
    assert agent.history[0]["role"] == "user"
    assert agent.history[1]["role"] == "assistant"
    assert agent.history[1]["content"] == "First response"


def test_expert_agent_includes_target_context():
    client = FakeChatClient(["Got the context."])
    agent = ExpertAgent(client)
    agent.set_target(_target())
    agent.respond("What should I do next?")
    user_msg = client.last_messages[0]["content"]
    assert "COF-LZU1" in user_msg
    assert "imine" in user_msg
    assert "crystallinity" in user_msg


def test_expert_agent_includes_protocol_context():
    client = FakeChatClient(["I see the protocols."])
    agent = ExpertAgent(client)
    agent.set_target(_target())
    agent.add_protocols([_protocol()])
    agent.respond("Evaluate these protocols.")
    user_msg = client.last_messages[0]["content"]
    assert "EXTRACTED PROTOCOLS" in user_msg
    assert "TFB" in user_msg
    assert "BET_surface_area=410.0m²/g" in user_msg


def test_expert_agent_includes_plan_context():
    client = FakeChatClient(["Plan updated."])
    agent = ExpertAgent(client)
    plan = ExpertPlan(
        goal="Optimize COF-LZU1 crystallinity",
        steps=[
            PlanStep(description="Search literature", action="search_literature", completed=True, result="Found 20 papers"),
            PlanStep(description="Extract protocols", action="extract_protocols"),
        ],
        current_step=1,
    )
    agent.set_plan(plan)
    agent.respond("What's the status?")
    user_msg = client.last_messages[0]["content"]
    assert "CURRENT PLAN" in user_msg
    assert "Search literature" in user_msg


def test_expert_agent_create_plan():
    client = FakeChatClient(["Here's a 5-step plan for COF-LZU1..."])
    agent = ExpertAgent(client)
    response = agent.create_plan(
        goal="Optimize COF-LZU1 crystallinity",
        user_message="I want to maximize crystallinity of COF-LZU1",
    )
    assert "5-step plan" in response
    user_msg = client.last_messages[0]["content"]
    assert "step-by-step plan" in user_msg
    assert "search_literature" in user_msg


def test_expert_agent_interpret_results():
    client = FakeChatClient(["The BET surface area of 410 m²/g is below the theoretical..."])
    agent = ExpertAgent(client)
    response = agent.interpret_results("BET surface area: 410 m²/g, PXRD shows broad peaks")
    assert "410" in response
    user_msg = client.last_messages[0]["content"]
    assert "interpret" in user_msg.lower() or "results" in user_msg.lower()


def test_expert_agent_evaluate_feasibility():
    client = FakeChatClient(["This protocol looks feasible but I'd recommend..."])
    agent = ExpertAgent(client)
    response = agent.evaluate_feasibility("TFB + PDA, dioxane/mesitylene, 120°C, 72h")
    assert "feasible" in response
    user_msg = client.last_messages[0]["content"]
    assert "TFB + PDA" in user_msg


def test_expert_agent_reset_clears_state():
    client = FakeChatClient(["Response"])
    agent = ExpertAgent(client)
    agent.set_target(_target())
    agent.add_protocols([_protocol()])
    agent.respond("test")
    agent.reset()
    assert agent.target is None
    assert len(agent.protocols) == 0
    assert len(agent.history) == 0
    assert agent.plan is None


def test_plan_step_advance():
    plan = ExpertPlan(
        goal="Test",
        steps=[
            PlanStep(description="Step 1", action="advise"),
            PlanStep(description="Step 2", action="advise"),
        ],
    )
    assert not plan.is_complete
    assert plan.next_step.description == "Step 1"
    plan.advance("Done step 1")
    assert plan.steps[0].completed
    assert plan.steps[0].result == "Done step 1"
    assert plan.next_step.description == "Step 2"
    plan.advance("Done step 2")
    assert plan.is_complete
    assert plan.next_step is None


def test_system_prompt_covers_key_topics():
    assert "imine" in SYSTEM_PROMPT
    assert "boronate ester" in SYSTEM_PROMPT
    assert "solvothermal" in SYSTEM_PROMPT
    assert "BET" in SYSTEM_PROMPT
    assert "PXRD" in SYSTEM_PROMPT
    assert "Bayesian" in SYSTEM_PROMPT
    assert "modulator" in SYSTEM_PROMPT
    assert "confirmation" in SYSTEM_PROMPT.lower() or "confirm" in SYSTEM_PROMPT.lower()


def test_build_context_block_empty_when_no_state():
    assert build_context_block({}) == ""


def test_measured_outcome_model():
    outcome = MeasuredOutcome(
        metric_name="BET_surface_area",
        value=410.0,
        unit="m²/g",
        measurement_method="N₂ adsorption at 77 K",
        inferred=False,
    )
    assert outcome.value == 410.0
    assert outcome.uncertainty is None
