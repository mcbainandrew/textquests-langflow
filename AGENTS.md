# Agent Specifications

## Overview

This document defines the specialized agents in our society-of-mind architecture. Each agent has specific responsibilities, inputs, outputs, and internal state.

## Design Philosophy

### Minsky's Society of Mind
- Intelligence emerges from interaction of simple specialists
- No single agent understands the whole problem
- Agents communicate through structured messages
- Coordinator synthesizes specialist advice into actions

### Agent Communication Protocol
```
Query → Agent → Response

Query = {
  "type": "question" | "update" | "reflect",
  "context": {...},
  "question": "...",
  "observation": "..."
}

Response = {
  "agent": "navigator" | "puzzle" | "memory" | "strategy",
  "answer": "...",
  "confidence": 0.0-1.0,
  "metadata": {...}
}
```

---

## Agent 1: Navigator

### Responsibility
Spatial reasoning and map management. Answers questions about locations, directions, and navigation.

### Core Questions
- "Where am I right now?"
- "What locations have I discovered?"
- "How do I get from here to X?"
- "What objects/features are available in location Y?"
- "Have I fully explored this area?"

### Internal State (Blackboard)
```json
{
  "locations": {
    "west_of_house": {
      "description": "You are standing in an open field...",
      "exits": ["north", "south", "west"],
      "objects_seen": ["mailbox", "door"],
      "visit_count": 3,
      "last_visited": "step_12"
    }
  },
  "current_location": "west_of_house",
  "connection_graph": {
    "west_of_house": {"north": "north_of_house", "south": "forest"}
  },
  "unexplored_exits": ["east_from_kitchen"],
  "map_completeness": 0.35
}
```

### Inputs
- Current observation text
- Previous location
- Action taken to reach current location
- Full history of movements

### Outputs
- Current location name (inferred from observation)
- Suggested navigation actions
- Map state (what's been explored)
- Confidence in location identification

### Reasoning Process
1. Parse observation for location clues (room descriptions, exits)
2. Match against known locations (fuzzy matching)
3. Update connection graph when new exit discovered
4. Identify unexplored exits
5. Suggest navigation priorities (unexplored > backtrack > known)

### Failure Modes Navigator Should Catch
- Looping between same 2-3 locations
- Not systematically exploring exits
- Forgetting locations visited earlier
- Not recognizing when back in known location

### Example Interaction
```
Coordinator: "Where should we go next?"
Navigator: {
  "agent": "navigator",
  "answer": "We're in the Living Room. Unexplored exit: 'up' (likely leads to upstairs). Unexplored locations: 4 more rooms to discover based on exit count.",
  "confidence": 0.85,
  "metadata": {
    "suggested_action": "go up",
    "unexplored_count": 4,
    "current_location": "living_room"
  }
}
```

---

## Agent 2: Puzzle Solver

### Responsibility
Analyzes object interactions, inventory management, and puzzle-solving patterns. Identifies what actions to try with objects.

### Core Questions
- "What objects do I have?"
- "What can I do with object X?"
- "How do I solve puzzle Y?"
- "What objects might be useful later?"
- "What combination of items might work here?"

### Internal State
```json
{
  "inventory": ["lamp", "sword", "bottle of water"],
  "objects_seen": {
    "trophy_case": {
      "location": "living_room",
      "interactions_tried": ["examine", "open"],
      "properties": ["container", "locked"],
      "last_seen": "step_15"
    }
  },
  "puzzle_patterns": {
    "locked_door": ["find key", "try items", "examine surroundings"],
    "dark_room": ["find light source", "turn on lamp"]
  },
  "item_combinations_tried": [
    ["lamp", "examine"],
    ["sword", "attack grue"]
  ],
  "successful_solutions": {
    "open_mailbox": "open mailbox"
  }
}
```

### Inputs
- Current observation (what's visible)
- Current inventory
- Objects mentioned in observation
- Previous action results
- Puzzle patterns learned from experience

### Outputs
- Suggested actions with objects
- Item prioritization (what to pick up)
- Puzzle solution hypotheses
- What hasn't been tried yet

### Reasoning Process
1. Extract objects from observation text
2. Cross-reference with inventory and known objects
3. Identify puzzles (locked doors, obstacles, requirements)
4. Match against learned puzzle patterns
5. Suggest untried interactions
6. Prioritize by likelihood of progress

### Failure Modes Puzzle Solver Should Catch
- Trying same failed action repeatedly
- Not picking up important items
- Missing obvious object interactions
- Not combining items creatively
- Forgetting about items picked up earlier

### Example Interaction
```
Coordinator: "There's a locked grate here. What should we do?"
Puzzle Solver: {
  "agent": "puzzle",
  "answer": "This is a locked barrier puzzle. I see we have keys in inventory. Suggest: 1) 'unlock grate with keys', 2) 'examine grate' for more clues. Pattern match: 72% similar to mailbox puzzle we solved earlier.",
  "confidence": 0.78,
  "metadata": {
    "puzzle_type": "locked_barrier",
    "suggested_actions": ["unlock grate with keys", "examine grate"],
    "relevant_inventory": ["keys"],
    "similar_puzzles": ["mailbox"]
  }
}
```

---

## Agent 3: Memory Tracker

### Responsibility
Maintains game state knowledge, action history, and what has/hasn't worked. Prevents repetition and reminds other agents of past experiences.

### Core Questions
- "Have we tried this before?"
- "What happened last time we did X?"
- "What actions led to progress?"
- "What actions led to failure/danger?"
- "Are we repeating ourselves?"

### Internal State
```json
{
  "action_history": [
    {
      "step": 5,
      "action": "go north",
      "result": "You are in the Kitchen",
      "progress_change": 0,
      "score_change": 0,
      "outcome": "neutral"
    },
    {
      "step": 8,
      "action": "attack troll with sword",
      "result": "The troll laughs at your puny weapon",
      "progress_change": 0,
      "score_change": 0,
      "outcome": "failure"
    }
  ],
  "successful_patterns": [
    {"pattern": "examine X → clue revealed", "count": 5}
  ],
  "failed_patterns": [
    {"pattern": "attack troll → no effect", "count": 3}
  ],
  "harmful_actions": [
    "drink acid"
  ],
  "loop_detection": {
    "last_5_actions": ["north", "south", "north", "south", "north"],
    "is_looping": true,
    "loop_pattern": "alternating_north_south"
  },
  "game_state": {
    "score": 10,
    "progress": 5,
    "steps_since_progress": 15,
    "current_run_length": 30
  }
}
```

### Inputs
- Current action being considered
- Full action history
- Results of previous actions
- Progress/score changes over time

### Outputs
- "Have we tried this?" flag
- Action recommendations based on history
- Loop detection warnings
- Successful pattern identification

### Reasoning Process
1. Check if proposed action tried recently
2. Look up past results of similar actions
3. Detect loops (repeated action sequences)
4. Identify stagnation (no progress for N steps)
5. Recommend novel actions or proven winners

### Failure Modes Memory Should Catch
- Repeating failed actions
- Looping between states
- Forgetting successful strategies
- Not recognizing when stuck
- Trying harmful actions again

### Example Interaction
```
Coordinator: "Should we try 'attack troll with sword'?"
Memory: {
  "agent": "memory",
  "answer": "NO - We tried this 3 times at steps 8, 12, 19. Result was always 'troll laughs, no damage'. Also, we're in a loop: last 5 actions alternate north/south. Suggest trying something completely different.",
  "confidence": 0.95,
  "metadata": {
    "tried_before": true,
    "times_tried": 3,
    "success_rate": 0.0,
    "loop_detected": true,
    "steps_since_progress": 15
  }
}
```

---

## Agent 4: Strategy Agent (Coordinator)

### Responsibility
High-level planning, goal-setting, and synthesizing advice from other agents into final decisions. Makes the ultimate choice of what action to take.

### Core Questions
- "What's our current goal?"
- "What should we prioritize: exploration, puzzle-solving, or backtracking?"
- "How do we synthesize conflicting agent advice?"
- "Are we making progress or spinning wheels?"
- "Should we try something risky or play it safe?"

### Internal State
```json
{
  "current_goal": "explore_house",
  "goal_stack": [
    {"goal": "explore_house", "priority": "high"},
    {"goal": "solve_mailbox", "priority": "medium"},
    {"goal": "find_treasure", "priority": "low"}
  ],
  "strategy": "systematic_exploration",
  "decision_log": [
    {
      "step": 10,
      "options": ["go north", "examine lamp", "take sword"],
      "agent_votes": {
        "navigator": "go north (0.8)",
        "puzzle": "take sword (0.7)",
        "memory": "any novel action (0.6)"
      },
      "final_decision": "go north",
      "reasoning": "Navigator confidence highest, exploration priority"
    }
  ],
  "progress_tracker": {
    "last_progress_step": 8,
    "current_step": 25,
    "steps_since_progress": 17,
    "strategy_effectiveness": 0.4
  }
}
```

### Inputs
- Responses from all specialist agents
- Current game observation
- Progress metrics (score, checkpoint percentage)
- Time constraints (max steps)

### Outputs
- Final action decision
- Reasoning for choice
- Goal updates
- Strategy adjustments

### Reasoning Process
1. Query relevant specialists based on observation
2. Collect their recommendations
3. Weight by confidence + relevance
4. Resolve conflicts (voting, priority rules)
5. Choose final action
6. Reflect: "Is this working?"
7. Adjust strategy if stuck

### Decision-Making Heuristics
- **When exploring**: Trust Navigator's suggestions (high weight)
- **When stuck on puzzle**: Trust Puzzle Solver (high weight)
- **When looping**: Trust Memory's warnings (veto power)
- **When no progress for 10+ steps**: Switch strategy (exploration → puzzle-solving or vice versa)
- **When conflicting advice**: Combine or choose highest confidence

### Failure Modes Strategy Should Catch
- Over-committing to failing strategy
- Not switching approaches when stuck
- Ignoring high-confidence warnings from Memory
- No clear goal (wandering aimlessly)
- Not adjusting to game progress feedback

### Example Interaction
```
[Strategy Agent queries all specialists]

Navigator: "Go east (unexplored exit) - confidence 0.85"
Puzzle: "Examine lamp (might be useful) - confidence 0.65"
Memory: "Both are novel, no prior attempts - confidence 0.5"

Strategy Decision: {
  "agent": "strategy",
  "action": "go east",
  "reasoning": "Navigator has highest confidence and we're in exploration mode. Goal: map out the house. Examining lamp is lower priority until we identify a puzzle requiring light.",
  "confidence": 0.82,
  "metadata": {
    "current_goal": "exploration",
    "votes": {"navigator": 0.85, "puzzle": 0.65, "memory": 0.5},
    "decision_basis": "highest_confidence + goal_alignment"
  }
}
```

---

## Inter-Agent Communication Flow

### Typical Decision Cycle

1. **Observation Arrives** (from game)
   ```
   "You are in a dark room. Exits: north, south."
   ```

2. **Strategy Agent broadcasts query** to specialists:
   ```json
   {
     "type": "question",
     "observation": "You are in a dark room...",
     "context": {
       "current_goal": "explore_house",
       "steps_since_progress": 5
     },
     "question": "What should we do?"
   }
   ```

3. **Specialists respond**:
   - **Navigator**: "Dark room + exits = navigation choice. But 'dark' = danger? Confidence: 0.6"
   - **Puzzle**: "DARK ROOM! This is a classic puzzle. We need light source. Check inventory for lamp. Confidence: 0.9"
   - **Memory**: "We have a lamp in inventory (picked up at step 3). Never tried 'turn on lamp'. Confidence: 0.8"

4. **Strategy synthesizes**:
   ```
   Puzzle detected critical pattern (dark room puzzle).
   Memory confirms we have lamp.
   Decision: "turn on lamp"
   Confidence: 0.95
   ```

5. **Action executed**, result fed back to all agents for state update

### Conflict Resolution Examples

#### Conflict: Navigator wants to explore, Puzzle wants to solve current puzzle
**Resolution**: Check goal priority. If stuck on puzzle for 5+ steps, prioritize Puzzle. If making progress with exploration, prioritize Navigator.

#### Conflict: Multiple agents suggest different actions with similar confidence
**Resolution**:
- If Memory says "tried this before and failed", veto
- Otherwise, try highest confidence first
- If tied, use current goal as tiebreaker

#### Conflict: Memory says "we're looping" but no agent has better suggestion
**Resolution**: Strategy Agent generates novel action:
- Random unexplored exit
- Backtrack to earlier promising location
- Try "help" or "hint" command
- Examine something mentioned but not yet examined

---

## Agent Evolution & Self-Improvement

### How Agents Get Better

1. **Pattern Learning**
   - Puzzle Solver: "Last 3 games, 'examine X' always revealed clues"
   - Memory: "Action loops always led to stagnation"
   - Navigator: "Second floor usually accessible via 'up' command"

2. **Cross-Game Transfer**
   - "Dark room + lamp" pattern works across multiple games
   - "Locked door → find key" is universal
   - Navigation strategies transfer (systematic exploration works)

3. **Reflection-Driven Specialization**
   - Post-game analysis: "We failed navigation puzzles"
   - System suggests: "Navigator needs better loop detection"
   - Flow mutation: Add "loop detector" sub-component to Navigator

### How New Agents Are Born

**Reflection identifies gap:**
```
Analysis: "Game 3 failed because we didn't track NPC locations and dialogue state.
Failure pattern: Lost track of which NPCs we talked to, repeated conversations."

Suggestion: "Need NPC Tracker agent to manage dialogue trees and NPC state."
```

**Flow evolution creates new agent:**
- New node in Langflow: "NPC Tracker"
- Responsibilities: Track NPCs, dialogue history, quest state
- Integrated into coordinator flow
- Tested on next game

---

## Agent Metrics & Performance Tracking

### Per-Agent Metrics

| Metric | Navigator | Puzzle Solver | Memory | Strategy |
|--------|-----------|---------------|--------|----------|
| Query count | # times queried | # times queried | # times queried | Always active |
| Confidence (avg) | 0.0-1.0 | 0.0-1.0 | 0.0-1.0 | 0.0-1.0 |
| Accuracy | Correct locations identified | Successful puzzle solutions | Loop detections correct | Progress per decision |
| Response time | ms per query | ms per query | ms per query | ms per decision |
| Contribution | Actions taken from Navigator advice | Actions from Puzzle advice | Vetoes accepted | Final decisions made |

### Success Criteria

**Navigator Success:**
- Map completeness > 80% by end of game
- Location identification accuracy > 90%
- No multi-step loops (same 2-3 locations)

**Puzzle Solver Success:**
- Puzzle solve rate > 50%
- Novel interactions suggested > 20 per game
- Successful inventory management (pick up useful items)

**Memory Success:**
- Loop detection rate: 100% (catch all loops)
- Action repetition prevention: > 90%
- No harmful action repetitions

**Strategy Success:**
- Progress per 10 steps > baseline
- Goal achievement rate > 60%
- Decision confidence correlates with success

---

## Future Agent Types (Discovered Through Reflection)

These agents don't exist yet but might be created based on gameplay analysis:

### Potential Future Agents

1. **NPC Tracker**
   - Manages dialogue trees, NPC states, quest tracking
   - Born from: Failing games with complex NPC interactions

2. **Resource Manager**
   - Tracks consumable items (food, water, fuel)
   - Born from: Dying due to resource depletion

3. **Risk Assessor**
   - Evaluates danger of actions (combat, harmful items)
   - Born from: Too many game-over states from risky actions

4. **Hint Interpreter**
   - Parses game hints and "feelies" (manuals, invisiclues)
   - Born from: Not utilizing available hint resources

5. **Meta-Learner**
   - Identifies cross-game patterns
   - Transfers strategies between similar games
   - Born from: Repeatedly solving same puzzle types

---

## Implementation Status

- [ ] Navigator: Specification complete, implementation pending
- [ ] Puzzle Solver: Specification complete, implementation pending
- [ ] Memory Tracker: Specification complete, implementation pending
- [ ] Strategy Agent: Specification complete, implementation pending
- [ ] Communication protocol: Defined, implementation pending
- [ ] Shared blackboard: Designed, implementation pending

---

## Next Steps

1. Build message bus and shared memory infrastructure
2. Implement agents as Langflow components
3. Create coordinator flow that connects all agents
4. Test on single game (Zork1) with logging
5. Analyze agent contributions and refine
