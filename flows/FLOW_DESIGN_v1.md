# Langflow Society-of-Mind Flow Design (v1_base)

## Overview

This document describes the structure of the v1_base Langflow flow for the 4-agent society-of-mind system.

## High-Level Architecture

```
Input (Game Observation)
    ↓
State Management (Update Blackboard)
    ↓
Parallel Specialist Queries
    ├→ Navigator Agent
    ├→ Puzzle Solver Agent
    └→ Memory Tracker Agent
    ↓
Synthesizer (Strategy Coordinator)
    ↓
Output (Final Action JSON)
```

## Detailed Node Structure

### 1. Input Layer

#### Node: ChatInput
- **Type**: ChatInput
- **Purpose**: Receives game observation text
- **Output**: Raw observation string
- **Example**: "You are in a dark forest. Exits: north, east."

### 2. State Management Layer

#### Node: StateParser
- **Type**: Code (Python)
- **Purpose**: Parse current game state from session
- **Inputs**:
  - observation (from ChatInput)
  - session_state (from Memory component)
- **Logic**:
  ```python
  # Extract previous state
  map_state = session.get("map_state", {})
  inventory = session.get("inventory", [])
  action_history = session.get("action_history", [])
  current_step = session.get("step", 0)
  progress = session.get("progress", 0)
  score = session.get("score", 0)

  # Parse observation for updates
  # (Basic parsing - agents will do deeper analysis)

  # Return structured state
  return {
    "observation": observation,
    "map_state": map_state,
    "inventory": inventory,
    "action_history": action_history,
    "current_step": current_step,
    "progress": progress,
    "score": score
  }
  ```
- **Outputs**: Structured game state dict

#### Node: Memory (Session State)
- **Type**: Memory/SessionMemory component
- **Purpose**: Persistent state across steps within single game
- **Stored**: map_state, inventory, action_history, step, progress, score

### 3. Specialist Agent Layer (Parallel)

#### Node: NavigatorPrompt
- **Type**: Prompt
- **Template**: Load from `prompts/navigator_agent.txt`
- **Variables**:
  - `{observation}` ← StateParser.observation
  - `{map_state}` ← StateParser.map_state (JSON string)
  - `{action_history}` ← StateParser.action_history (last 10 actions)
- **Output**: Formatted prompt string

#### Node: NavigatorLLM
- **Type**: ChatOpenAI / ChatAnthropic
- **Model**: claude-3-5-sonnet (or specified model)
- **Input**: NavigatorPrompt output
- **Temperature**: 0.3 (deterministic)
- **Output**: Navigator's JSON response

#### Node: PuzzlePrompt
- **Type**: Prompt
- **Template**: Load from `prompts/puzzle_solver_agent.txt`
- **Variables**:
  - `{observation}` ← StateParser.observation
  - `{inventory}` ← StateParser.inventory (JSON)
  - `{known_objects}` ← StateParser.map_state.objects (extracted)
  - `{interaction_history}` ← StateParser.action_history (filtered for object interactions)
- **Output**: Formatted prompt string

#### Node: PuzzleLLM
- **Type**: ChatOpenAI / ChatAnthropic
- **Model**: claude-3-5-sonnet
- **Input**: PuzzlePrompt output
- **Temperature**: 0.4 (slightly creative)
- **Output**: Puzzle solver's JSON response

#### Node: MemoryPrompt
- **Type**: Prompt
- **Template**: Load from `prompts/memory_tracker_agent.txt`
- **Variables**:
  - `{proposed_action}` ← "TBD" (initially empty, updated later)
  - `{action_history}` ← StateParser.action_history (last 20 actions)
  - `{current_step}` ← StateParser.current_step
  - `{progress}` ← StateParser.progress
  - `{score}` ← StateParser.score
  - `{steps_since_progress}` ← calculated from history
- **Output**: Formatted prompt string

#### Node: MemoryLLM
- **Type**: ChatOpenAI / ChatAnthropic
- **Model**: claude-3-5-sonnet
- **Input**: MemoryPrompt output
- **Temperature**: 0.2 (very deterministic)
- **Output**: Memory tracker's JSON response

### 4. Response Parsing Layer

#### Node: ParseNavigator
- **Type**: Code (Python)
- **Purpose**: Parse Navigator's JSON response
- **Input**: NavigatorLLM output
- **Logic**:
  ```python
  import json
  try:
      response = json.loads(navigator_output)
      return response
  except:
      # Fallback if JSON parsing fails
      return {"agent": "navigator", "confidence": 0.0, "error": "parse_failed"}
  ```
- **Output**: Navigator dict

#### Node: ParsePuzzle
- **Type**: Code (Python)
- **Purpose**: Parse Puzzle solver's JSON response
- **Logic**: Same as ParseNavigator
- **Output**: Puzzle dict

#### Node: ParseMemory
- **Type**: Code (Python)
- **Purpose**: Parse Memory tracker's JSON response
- **Logic**: Same as ParseNavigator
- **Output**: Memory dict

### 5. Coordinator Layer

#### Node: StrategyPrompt
- **Type**: Prompt
- **Template**: Load from `prompts/strategy_coordinator_agent.txt`
- **Variables**:
  - `{observation}` ← StateParser.observation
  - `{current_step}` ← StateParser.current_step
  - `{progress}` ← StateParser.progress
  - `{score}` ← StateParser.score
  - `{current_goal}` ← session.get("current_goal", "explore")
  - `{steps_since_progress}` ← calculated
  - `{navigator_response}` ← ParseNavigator output (JSON string)
  - `{puzzle_response}` ← ParsePuzzle output (JSON string)
  - `{memory_response}` ← ParseMemory output (JSON string)
- **Output**: Formatted prompt string

#### Node: StrategyLLM
- **Type**: ChatOpenAI / ChatAnthropic
- **Model**: claude-3-5-sonnet (most powerful model)
- **Input**: StrategyPrompt output
- **Temperature**: 0.3
- **Output**: Strategy coordinator's JSON response

#### Node: ParseStrategy
- **Type**: Code (Python)
- **Purpose**: Parse final decision and extract action
- **Input**: StrategyLLM output
- **Logic**:
  ```python
  import json
  try:
      response = json.loads(strategy_output)
      final_action = response.get("final_action", "look")
      reasoning = response.get("reasoning", "")
      confidence = response.get("confidence", 0.5)

      # Log decision for analysis
      decision_log.append({
          "step": current_step,
          "action": final_action,
          "reasoning": reasoning,
          "confidence": confidence,
          "agent_votes": response.get("specialist_votes", {})
      })

      return {
          "action": final_action,
          "reasoning": reasoning,
          "full_response": response
      }
  except:
      # Fallback
      return {"action": "look", "reasoning": "parse error", "full_response": {}}
  ```
- **Output**: Final action dict

### 6. State Update Layer

#### Node: UpdateState
- **Type**: Code (Python)
- **Purpose**: Update session state with new information
- **Inputs**:
  - ParseStrategy output
  - ParseNavigator output
  - StateParser (previous state)
- **Logic**:
  ```python
  # Update map state from Navigator
  if "map_update" in navigator_response:
      map_state.update(navigator_response["map_update"])

  # Add action to history
  action_history.append({
      "step": current_step,
      "action": final_action,
      "observation": observation,
      "result": "pending"  # Will be updated next step
  })

  # Update session
  session["map_state"] = map_state
  session["action_history"] = action_history[-50:]  # Keep last 50
  session["step"] = current_step + 1
  session["current_goal"] = strategy_response.get("current_goal", "explore")

  return {"updated": True}
  ```
- **Output**: Confirmation

### 7. Output Layer

#### Node: FormatOutput
- **Type**: Code (Python)
- **Purpose**: Format final action as expected JSON
- **Input**: ParseStrategy output
- **Logic**:
  ```python
  # TextQuests expects: {"reasoning": "...", "action": "..."}
  return {
      "reasoning": strategy_output["reasoning"],
      "action": strategy_output["action"]
  }
  ```
- **Output**: TextQuests-compatible JSON

#### Node: ChatOutput
- **Type**: ChatOutput
- **Input**: FormatOutput output
- **Purpose**: Return action to game engine

## Data Flow Summary

```
ChatInput (observation)
    ↓
StateParser (extract state from session)
    ↓
    ├→ NavigatorPrompt → NavigatorLLM → ParseNavigator
    ├→ PuzzlePrompt → PuzzleLLM → ParsePuzzle
    └→ MemoryPrompt → MemoryLLM → ParseMemory
    ↓
StrategyPrompt (receives all 3 parsed responses)
    ↓
StrategyLLM
    ↓
ParseStrategy (extract final action)
    ↓
UpdateState (save to session)
    ↓
FormatOutput (TextQuests JSON format)
    ↓
ChatOutput (return to game)
```

## Key Connections (Edges)

1. ChatInput → StateParser
2. StateParser → NavigatorPrompt (observation, map_state, action_history)
3. StateParser → PuzzlePrompt (observation, inventory, known_objects)
4. StateParser → MemoryPrompt (action_history, step, progress, score)
5. NavigatorPrompt → NavigatorLLM
6. PuzzlePrompt → PuzzleLLM
7. MemoryPrompt → MemoryLLM
8. NavigatorLLM → ParseNavigator
9. PuzzleLLM → ParsePuzzle
10. MemoryLLM → ParseMemory
11. ParseNavigator + ParsePuzzle + ParseMemory + StateParser → StrategyPrompt
12. StrategyPrompt → StrategyLLM
13. StrategyLLM → ParseStrategy
14. ParseStrategy + ParseNavigator → UpdateState
15. ParseStrategy → FormatOutput
16. FormatOutput → ChatOutput

## Session State Schema

```json
{
  "map_state": {
    "locations": {
      "west_of_house": {
        "description": "...",
        "exits": ["north", "south", "west"],
        "objects": ["mailbox"],
        "visit_count": 3
      }
    },
    "current_location": "west_of_house",
    "connections": {
      "west_of_house": {"north": "north_of_house"}
    }
  },
  "inventory": ["lamp", "sword"],
  "action_history": [
    {
      "step": 1,
      "action": "go north",
      "observation": "You are north of the house.",
      "score_change": 0,
      "progress_change": 0
    }
  ],
  "step": 15,
  "progress": 5,
  "score": 10,
  "current_goal": "explore",
  "decision_log": [...]
}
```

## Model Recommendations

| Agent | Recommended Model | Temperature | Rationale |
|-------|-------------------|-------------|-----------|
| Navigator | claude-3-5-sonnet | 0.3 | Needs spatial reasoning |
| Puzzle | claude-3-5-sonnet | 0.4 | Benefits from creativity |
| Memory | claude-3-5-haiku | 0.2 | Fast, deterministic checks |
| Strategy | claude-3-5-sonnet | 0.3 | Most critical decisions |

Alternative for cost optimization:
- Use claude-3-5-haiku for Navigator and Puzzle
- Use claude-3-5-sonnet only for Strategy

## Implementation Options

### Option A: Build in Langflow UI
1. Start Langflow server
2. Create new flow
3. Add nodes as specified above
4. Connect edges as listed
5. Test with sample observation
6. Export JSON to `flows/society_v1_base.json`

### Option B: Programmatic Generation
1. Write Python script to generate Langflow JSON
2. Use flow_generator.py (to be built)
3. Output to `flows/society_v1_base.json`
4. Import into Langflow to verify

## Testing Checklist

- [ ] Flow accepts observation text as input
- [ ] All 3 specialist agents return valid JSON
- [ ] Strategy coordinator receives all specialist responses
- [ ] Final action is in correct format: `{"reasoning": "...", "action": "..."}`
- [ ] Session state persists across multiple steps
- [ ] Map state accumulates location information
- [ ] Action history tracks all previous actions
- [ ] Memory agent detects repeated actions
- [ ] Navigator suggests sensible directions
- [ ] Puzzle solver identifies objects in observation

## Integration with TextQuests

The LangFlowAgent in `src/llm_agents.py` needs to:
1. Convert OpenAI message format to single observation string
2. POST to `/api/v1/responses` with flow ID
3. Parse response to extract action
4. Format as TextQuests expects: `{"reasoning": "...", "action": "..."}`

Current LangFlowAgent already does this, but may need adjustment for multi-step sessions.

## Next Steps

1. Decide implementation approach (UI vs. programmatic)
2. Build the flow
3. Test with sample observations
4. Integrate with TextQuests play_textquests.py
5. Run on Zork1 for initial testing
6. Collect logs and analyze performance
