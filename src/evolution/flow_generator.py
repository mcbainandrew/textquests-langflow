"""
Langflow Flow Generator

Programmatically creates Langflow flow JSON files for society-of-mind orchestrations.

Design Choices:
- Build flows bottom-up: nodes first, then edges
- Use builder pattern for fluent API
- Generate UUIDs for node IDs
- Support loading prompt templates from files
- Output valid Langflow v1.0+ JSON

Note: This is simplified - real Langflow schema is complex.
May need adjustments based on actual Langflow version.
"""

import json
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class LangflowNode:
    """Represents a Langflow node"""
    id: str
    type: str  # "ChatInput", "ChatOutput", "Prompt", "ChatOpenAI", "Code", etc.
    position: Tuple[int, int] = (0, 0)  # (x, y) coordinates
    data: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate node structure"""
        if not self.id:
            raise ValueError("Node ID cannot be empty")
        if not self.type:
            raise ValueError("Node type cannot be empty")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Langflow JSON format"""
        self.validate()
        return {
            "id": self.id,
            "type": self.type,
            "position": {"x": self.position[0], "y": self.position[1]},
            "data": self.data
        }


@dataclass
class LangflowEdge:
    """Represents a connection between nodes"""
    source: str  # Source node ID
    target: str  # Target node ID
    sourceHandle: str = "output"
    targetHandle: str = "input"

    def validate(self) -> bool:
        """Validate edge structure"""
        if not self.source:
            raise ValueError("Edge source cannot be empty")
        if not self.target:
            raise ValueError("Edge target cannot be empty")
        return True

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "source": self.source,
            "target": self.target,
            "sourceHandle": self.sourceHandle,
            "targetHandle": self.targetHandle,
            "id": f"{self.source}-{self.target}"
        }

@dataclass
class FlowConfig:
    """Configuration for flow generation"""
    default_model: str = "claude-3-5-sonnet-20241022"
    default_temperature: float = 0.3
    prompts_dir: str = "prompts"
    flows_dir: str = "flows"

class FlowGenerator:
    """
    Builder for Langflow flow JSON files.

    Usage:
        generator = FlowGenerator()

        # Add input/output
        input_node = generator.add_chat_input()
        output_node = generator.add_chat_output()

        # Add agent nodes
        navigator = generator.add_prompt_agent("Navigator", "prompts/navigator_agent.txt")

        # Connect nodes
        generator.add_edge(input_node, navigator)

        # Generate JSON
        flow_json = generator.generate()
        generator.save("flows/my_flow.json")
    """

    def __init__(self, name: str = "Society of Mind Flow"):
        self.name = name
        self.nodes: List[LangflowNode] = []
        self.edges: List[LangflowEdge] = []
        self.node_counter = 0  # For positioning

    def _generate_id(self, prefix: str = "node") -> str:
        """Generate unique node ID"""
        return f"{prefix}-{uuid.uuid4().hex[:8]}"

    def _get_position(self) -> Tuple[int, int]:
        """Get next node position (simple grid layout)"""
        x = (self.node_counter % 4) * 300
        y = (self.node_counter // 4) * 200
        self.node_counter += 1
        return (x, y)

    def add_chat_input(self, name: str = "Input") -> str:
        """Add ChatInput node (entry point)"""
        node_id = self._generate_id("input")
        node = LangflowNode(
            id=node_id,
            type="ChatInput",
            position=self._get_position(),
            data={
                "node": {
                    "template": {
                        "input_value": {"value": ""},
                        "sender": {"value": "User"},
                        "sender_name": {"value": "User"},
                        "session_id": {"value": ""},
                        "should_store_message": {"value": True}
                    },
                    "display_name": name
                }
            }
        )
        self.nodes.append(node)
        return node_id

    def add_chat_output(self, name: str = "Output") -> str:
        """Add ChatOutput node (exit point)"""
        node_id = self._generate_id("output")
        node = LangflowNode(
            id=node_id,
            type="ChatOutput",
            position=self._get_position(),
            data={
                "node": {
                    "template": {
                        "input_value": {"value": ""},
                        "sender": {"value": "Machine"},
                        "sender_name": {"value": "AI"},
                        "session_id": {"value": ""},
                        "data_template": {"value": "{text}"},
                        "should_store_message": {"value": True}
                    },
                    "display_name": name
                }
            }
        )
        self.nodes.append(node)
        return node_id

    def add_prompt(
        self,
        template: str,
        name: str = "Prompt",
        variables: Optional[List[str]] = None
    ) -> str:
        """
        Add Prompt node.

        Args:
            template: Prompt text with {variable} placeholders
            name: Display name
            variables: List of variable names to expose as inputs

        Returns:
            Node ID
        """
        node_id = self._generate_id("prompt")

        # Build template structure
        template_data = {
            "template": {"value": template},
        }

        # Add variable inputs
        if variables:
            for var in variables:
                template_data[var] = {"value": ""}

        node = LangflowNode(
            id=node_id,
            type="Prompt",
            position=self._get_position(),
            data={
                "node": {
                    "template": template_data,
                    "display_name": name
                }
            }
        )
        self.nodes.append(node)
        return node_id

    def add_llm(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.3,
        name: str = "LLM"
    ) -> str:
        """
        Add LLM node (Claude via Anthropic).

        Args:
            model: Model identifier
            temperature: Sampling temperature
            name: Display name

        Returns:
            Node ID
        """
        node_id = self._generate_id("llm")
        node = LangflowNode(
            id=node_id,
            type="ChatAnthropic",
            position=self._get_position(),
            data={
                "node": {
                    "template": {
                        "model_name": {"value": model},
                        "temperature": {"value": temperature},
                        "max_tokens": {"value": 4096},
                        "anthropic_api_key": {"value": ""},  # From env
                        "input_value": {"value": ""}
                    },
                    "display_name": name
                }
            }
        )
        self.nodes.append(node)
        return node_id

    def add_code(
        self,
        code: str,
        name: str = "Code",
        input_vars: Optional[List[str]] = None
    ) -> str:
        """
        Add Python Code node.

        Args:
            code: Python code to execute
            name: Display name
            input_vars: Input variable names

        Returns:
            Node ID
        """
        node_id = self._generate_id("code")

        template_data = {
            "code": {"value": code}
        }

        if input_vars:
            for var in input_vars:
                template_data[var] = {"value": ""}

        node = LangflowNode(
            id=node_id,
            type="Code",
            position=self._get_position(),
            data={
                "node": {
                    "template": template_data,
                    "display_name": name
                }
            }
        )
        self.nodes.append(node)
        return node_id

    def add_agent(
        self,
        agent_name: str,
        prompt_template_path: str,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.3
    ) -> Tuple[str, str, str]:
        """
        Add a complete agent (Prompt + LLM + Parser).

        Args:
            agent_name: Agent name (e.g., "Navigator")
            prompt_template_path: Path to prompt template file
            model: LLM model
            temperature: Sampling temperature

        Returns:
            Tuple of (prompt_id, llm_id, parser_id)
        """
        # Load prompt template
        prompt_path = Path(prompt_template_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")

        with open(prompt_path, 'r') as f:
            template = f.read()

        # Extract variables from template
        import re
        variables = list(set(re.findall(r'\{(\w+)\}', template)))

        # Create nodes
        prompt_id = self.add_prompt(
            template=template,
            name=f"{agent_name} Prompt",
            variables=variables
        )

        llm_id = self.add_llm(
            model=model,
            temperature=temperature,
            name=f"{agent_name} LLM"
        )

        # Parser code
        parser_code = f"""
import json

# Parse {agent_name} response
try:
    response = json.loads(llm_output)
    return response
except json.JSONDecodeError:
    # Fallback
    return {{"agent": "{agent_name.lower()}", "error": "parse_failed", "confidence": 0.0}}
"""

        parser_id = self.add_code(
            code=parser_code,
            name=f"Parse {agent_name}",
            input_vars=["llm_output"]
        )

        # Connect: Prompt → LLM → Parser
        self.add_edge(prompt_id, llm_id)
        self.add_edge(llm_id, parser_id)

        return (prompt_id, llm_id, parser_id)

    def add_edge(
        self,
        source: str,
        target: str,
        source_handle: str = "output",
        target_handle: str = "input"
    ):
        """Add an edge between two nodes"""
        edge = LangflowEdge(
            source=source,
            target=target,
            sourceHandle=source_handle,
            targetHandle=target_handle
        )
        self.edges.append(edge)

    def generate(self) -> Dict[str, Any]:
        """Generate complete Langflow JSON"""
        return {
            "name": self.name,
            "description": "Generated by FlowGenerator",
            "data": {
                "nodes": [node.to_dict() for node in self.nodes],
                "edges": [edge.to_dict() for edge in self.edges]
            }
        }

    def save(self, output_path: str, pretty: bool = True):
        """Save flow to JSON file"""
        flow_json = self.generate()

        output_dir = Path(output_path).parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            if pretty:
                json.dump(flow_json, f, indent=2)
            else:
                json.dump(flow_json, f)

        print(f"Flow saved to {output_path}")

    @classmethod
    def load(cls, flow_path: str) -> 'FlowGenerator':
        """Load existing flow for modification"""
        with open(flow_path, 'r') as f:
            flow_data = json.load(f)

        generator = cls(name=flow_data.get("name", "Loaded Flow"))

        # Load nodes
        for node_data in flow_data.get("data", {}).get("nodes", []):
            node = LangflowNode(
                id=node_data["id"],
                type=node_data["type"],
                position=(node_data["position"]["x"], node_data["position"]["y"]),
                data=node_data.get("data", {})
            )
            generator.nodes.append(node)

        # Load edges
        for edge_data in flow_data.get("data", {}).get("edges", []):
            edge = LangflowEdge(
                source=edge_data["source"],
                target=edge_data["target"],
                sourceHandle=edge_data.get("sourceHandle", "output"),
                targetHandle=edge_data.get("targetHandle", "input")
            )
            generator.edges.append(edge)

        generator.node_counter = len(generator.nodes)

        return generator


def create_society_v1_flow(output_path: str = "flows/society_v1_base.json"):
    """
    Create the v1_base society-of-mind flow.

    This is a helper function that generates the complete flow structure
    as designed in flows/FLOW_DESIGN_v1.md
    """
    print("Generating Society of Mind v1 flow...")

    generator = FlowGenerator(name="Society of Mind v1")

    # 1. Input/Output
    print("  Adding I/O nodes...")
    input_node = generator.add_chat_input("Game Observation")
    output_node = generator.add_chat_output("Action Response")

    # 2. State parser
    print("  Adding state management...")
    state_parser_code = """
# Parse game state from session
map_state = session.get("map_state", {})
inventory = session.get("inventory", [])
action_history = session.get("action_history", [])
current_step = session.get("step", 0)
progress = session.get("progress", 0)
score = session.get("score", 0)

# Return structured state
return {
    "observation": observation,
    "map_state": json.dumps(map_state),
    "inventory": json.dumps(inventory),
    "action_history": json.dumps(action_history[-20:]),
    "current_step": current_step,
    "progress": progress,
    "score": score,
    "steps_since_progress": session.get("steps_since_progress", 0)
}
"""
    state_parser = generator.add_code(
        code=state_parser_code,
        name="State Parser",
        input_vars=["observation", "session"]
    )

    generator.add_edge(input_node, state_parser)

    # 3. Specialist agents
    print("  Adding specialist agents...")
    navigator_prompt, navigator_llm, navigator_parser = generator.add_agent(
        "Navigator",
        "prompts/navigator_agent.txt",
        temperature=0.3
    )

    puzzle_prompt, puzzle_llm, puzzle_parser = generator.add_agent(
        "Puzzle Solver",
        "prompts/puzzle_solver_agent.txt",
        temperature=0.4
    )

    memory_prompt, memory_llm, memory_parser = generator.add_agent(
        "Memory Tracker",
        "prompts/memory_tracker_agent.txt",
        temperature=0.2
    )

    # Connect state parser to all specialist prompts
    generator.add_edge(state_parser, navigator_prompt)
    generator.add_edge(state_parser, puzzle_prompt)
    generator.add_edge(state_parser, memory_prompt)

    # 4. Strategy coordinator
    print("  Adding strategy coordinator...")
    strategy_prompt, strategy_llm, strategy_parser = generator.add_agent(
        "Strategy Coordinator",
        "prompts/strategy_coordinator_agent.txt",
        temperature=0.3
    )

    # Connect all parsers to strategy
    generator.add_edge(navigator_parser, strategy_prompt)
    generator.add_edge(puzzle_parser, strategy_prompt)
    generator.add_edge(memory_parser, strategy_prompt)
    generator.add_edge(state_parser, strategy_prompt)

    # 5. Final output formatting
    print("  Adding output formatter...")
    format_code = """
import json

# Format for TextQuests
return {
    "reasoning": strategy_output.get("reasoning", ""),
    "action": strategy_output.get("final_action", "look")
}
"""
    formatter = generator.add_code(
        code=format_code,
        name="Format Output",
        input_vars=["strategy_output"]
    )

    generator.add_edge(strategy_parser, formatter)
    generator.add_edge(formatter, output_node)

    # Generate and save
    print(f"  Saving to {output_path}...")
    generator.save(output_path)

    print("✅ Society v1 flow generated successfully!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = "flows/society_v1_base.json"

    create_society_v1_flow(output_path)
