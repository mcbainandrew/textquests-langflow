# Agent Prompt Templates

This directory contains prompt templates for the specialized agents in our society-of-mind system.

## Prompt Files

### navigator_agent.txt
**Purpose**: Spatial reasoning and map management
**Inputs**: observation, map_state, action_history
**Output**: Location identification, navigation suggestions, map updates
**Use in Langflow**: Prompt node connected to LLM node (Claude/GPT)

### puzzle_solver_agent.txt
**Purpose**: Object interactions and puzzle-solving
**Inputs**: observation, inventory, known_objects, interaction_history
**Output**: Puzzle analysis, suggested actions, item priorities
**Use in Langflow**: Prompt node connected to LLM node

### memory_tracker_agent.txt
**Purpose**: Action history and repetition prevention
**Inputs**: proposed_action, action_history, progress, score
**Output**: VETO/ALLOW recommendation, loop detection, alternatives
**Use in Langflow**: Prompt node connected to LLM node

### strategy_coordinator_agent.txt
**Purpose**: High-level coordination and final decisions
**Inputs**: observation, navigator_response, puzzle_response, memory_response, game_state
**Output**: Final action decision with reasoning
**Use in Langflow**: Prompt node connected to LLM node (should be most powerful model)

## Variable Substitution

Prompts use `{variable_name}` placeholders that should be substituted with actual game state in the Langflow flow:

- `{observation}` - Current game observation text
- `{map_state}` - JSON of explored locations and connections
- `{inventory}` - List of items player is carrying
- `{action_history}` - Recent actions and their results
- `{current_step}` - Step number in current game
- `{progress}` - Game progress percentage (0-100)
- `{score}` - Current game score
- `{proposed_action}` - Action being considered
- `{navigator_response}` - Navigator agent's JSON output
- `{puzzle_response}` - Puzzle solver's JSON output
- `{memory_response}` - Memory tracker's JSON output

## JSON Output Schemas

All agents output JSON. See each prompt file for exact schema.

Key fields common to all:
- `agent`: Agent identifier string
- `confidence`: Float 0.0-1.0 indicating confidence in response
- `reasoning`: String explaining the recommendation

## Usage in Langflow

1. **Create Prompt nodes** - One per agent, load template from file
2. **Add variable substitution** - Connect game state nodes to prompt variables
3. **Connect to LLM nodes** - Route prompts to Claude/GPT API calls
4. **Parse JSON outputs** - Use JSON parser nodes to extract structured responses
5. **Feed to Coordinator** - All specialist outputs feed into Strategy Coordinator

## Evolution Notes

These prompts can evolve over time based on reflection feedback:
- Add new puzzle patterns to puzzle_solver
- Refine loop detection criteria in memory_tracker
- Adjust decision heuristics in strategy_coordinator
- Add new responsibilities as discovered through gameplay

Prompt versions should be tracked alongside flow versions (see FLOWS.md).
