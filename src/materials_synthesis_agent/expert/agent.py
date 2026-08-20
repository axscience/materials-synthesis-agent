"""Materials Science Expert agent — an LLM-powered orchestrator that reasons about synthesis
strategy, coordinates the literature agent and optimizer, and guides the user through the
full COF synthesis optimization workflow.

Unlike the literature extraction pipeline (forced tool-use, no free-form reasoning) and the
NL parser (forced tool-use, single turn), the expert agent uses conversational `chat()` for
multi-turn reasoning and planning, with explicit user confirmation before executing actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from materials_synthesis_agent.expert.prompts import SYSTEM_PROMPT, build_context_block
from materials_synthesis_agent.llm import LLMClient
from materials_synthesis_agent.schema import ProtocolCandidate, Target


@dataclass
class PlanStep:
    description: str
    action: str
    completed: bool = False
    result: Optional[str] = None
    requires_confirmation: bool = True

    VALID_ACTIONS = (
        "search_literature",
        "extract_protocols",
        "check_feasibility",
        "resolve_structure",
        "setup_parameter_space",
        "suggest_next_experiment",
        "interpret_results",
        "advise",
        "ask_user",
    )


@dataclass
class ExpertPlan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    current_step: int = 0

    @property
    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)

    @property
    def next_step(self) -> Optional[PlanStep]:
        if self.is_complete:
            return None
        return self.steps[self.current_step]

    def advance(self, result: str) -> None:
        if not self.is_complete:
            self.steps[self.current_step].completed = True
            self.steps[self.current_step].result = result
            self.current_step += 1


class ExpertAgent:
    """Orchestrates synthesis optimization with user-in-the-loop confirmation.

    The agent maintains conversation history and session state (target, protocols, plan,
    observations). Every action that modifies state or costs money is presented as a plan
    step and requires explicit user confirmation before execution.

    Usage:
        agent = ExpertAgent(client)
        response = agent.respond("I want to synthesize COF-LZU1 with maximum crystallinity")
        # agent.plan is now populated — inspect it, confirm steps, execute
    """

    def __init__(self, client: LLMClient):
        self.client = client
        self.history: list[dict] = []
        self.target: Optional[Target] = None
        self.protocols: list[ProtocolCandidate] = []
        self.plan: Optional[ExpertPlan] = None
        self.observations: list = []

    def _context(self) -> dict:
        return {
            "target": self.target,
            "protocols": self.protocols,
            "observations": self.observations,
            "plan": self.plan,
        }

    def respond(self, user_message: str) -> str:
        """Send a message to the expert and get a response. The expert sees the full
        conversation history and current session state."""
        context_block = build_context_block(self._context())
        enriched_message = user_message
        if context_block:
            enriched_message = f"{user_message}\n\n[Current session state]\n{context_block}"

        self.history.append({"role": "user", "content": enriched_message})

        response = self.client.chat(
            messages=self.history,
            system=SYSTEM_PROMPT,
            max_tokens=4000,
        )

        self.history.append({"role": "assistant", "content": response})
        return response

    def create_plan(self, goal: str, user_message: str) -> str:
        """Ask the expert to create a plan for a given goal. Returns the expert's response
        describing the plan. The plan itself is extracted from the response and stored in
        self.plan for step-by-step execution."""
        prompt = (
            f"{user_message}\n\n"
            f"Create a step-by-step plan for: {goal}\n\n"
            "For each step, specify:\n"
            "1. What to do (description)\n"
            "2. What action type it maps to (one of: search_literature, extract_protocols, "
            "check_feasibility, resolve_structure, setup_parameter_space, "
            "suggest_next_experiment, interpret_results, advise, ask_user)\n"
            "3. Whether it requires user confirmation before executing\n\n"
            "Present the plan clearly so the user can review and approve it."
        )

        response = self.respond(prompt)

        # The plan is parsed from the expert's response by the CLI layer (or caller),
        # which has access to the user for confirmation. The expert's response is the
        # human-readable plan description.
        return response

    def update_plan(self, feedback: str) -> str:
        """Ask the expert to update the current plan based on user feedback or new results."""
        if self.plan is None:
            return self.respond(f"No plan exists yet. {feedback}")

        prompt = (
            f"The user has feedback on the current plan: {feedback}\n\n"
            f"Current plan goal: {self.plan.goal}\n"
            f"Steps completed: {self.plan.current_step} of {len(self.plan.steps)}\n\n"
            "Update the plan accordingly. Present the revised plan for approval."
        )
        return self.respond(prompt)

    def interpret_results(self, results_description: str) -> str:
        """Ask the expert to interpret experimental results and suggest next steps."""
        prompt = (
            f"New experimental results:\n{results_description}\n\n"
            "As a materials scientist, interpret these results:\n"
            "1. Are they consistent with expectations? If not, what might explain the discrepancy?\n"
            "2. What do they tell us about the parameter space?\n"
            "3. Should we modify the optimization strategy?\n"
            "4. What characterization would confirm or clarify these results?\n"
            "5. What should we try next?"
        )
        return self.respond(prompt)

    def evaluate_feasibility(self, protocol_description: str) -> str:
        """Ask the expert to evaluate whether a proposed protocol is feasible."""
        prompt = (
            f"Evaluate the feasibility of this synthesis protocol:\n{protocol_description}\n\n"
            "Consider:\n"
            "1. Are the monomers commercially available? If not, are they synthetically accessible?\n"
            "2. Are the reaction conditions reasonable (temperature, time, solvent compatibility)?\n"
            "3. Is the stoichiometry correct for the target topology?\n"
            "4. Are there known issues with this combination (e.g., solubility problems, "
            "competing reactions)?\n"
            "5. What is the expected difficulty level (routine, moderate, challenging)?\n"
            "6. Any safety concerns with the proposed conditions?"
        )
        return self.respond(prompt)

    def advise(self, question: str) -> str:
        """Ask the expert a domain question — no action, just reasoning."""
        return self.respond(question)

    def set_target(self, target: Target) -> None:
        self.target = target

    def add_protocols(self, protocols: list[ProtocolCandidate]) -> None:
        self.protocols.extend(protocols)

    def set_plan(self, plan: ExpertPlan) -> None:
        self.plan = plan

    def reset(self) -> None:
        self.history.clear()
        self.target = None
        self.protocols.clear()
        self.plan = None
        self.observations.clear()
