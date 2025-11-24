# Self-Evolving Society-of-Mind Architecture for TextQuests

## Vision

Build a self-improving agent orchestration system that plays text-based adventure games by:
1. Organizing specialized agents into a Minsky-style "society of mind"
2. Collecting gameplay traces and analyzing performance
3. Reflecting on failures to identify architectural improvements
4. Automatically evolving the orchestration structure based on feedback

This is a **data-driven bitter-lesson approach**: let the system grow its own structure through experience rather than hand-coding solutions.

## Core Principles

### 1. Society of Mind (Minsky)
- Complex intelligence emerges from interaction of simple specialized agents
- No single "smart" agent, but coordinated specialists
- Agents communicate through message passing and shared memory

### 2. Reflective Self-Improvement
- System analyzes its own gameplay logs
- Identifies patterns of failure (getting stuck, looping, low progress)
- Suggests new agent types or structural changes
- Example: "We keep getting lost → need a Navigator agent with map tracking"

### 3. Bitter Lesson (Data-Driven)
- Don't hand-code domain knowledge
- Let the system discover what works through trial and error
- Leverage compute: run many variants, keep what works
- Programmatic generation of orchestration structures

### 4. Langflow Nested Orchestrations
- Orchestrations can orchestrate other orchestrations
- Enables hierarchical agent societies
- Flow files can be generated and mutated programmatically

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TextQuests Game                          │
│              (25 Infocom Text Adventures)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │ observation
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          Society Orchestration (Langflow)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Navigator │  │  Puzzle  │  │  Memory  │  │Strategy  │   │
│  │  Agent   │  │  Solver  │  │ Tracker  │  │  Agent   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       └─────────────┴──────────────┴─────────────┘         │
│                 Message Bus / Coordinator                   │
└──────────────────────┬──────────────────────────────────────┘
                       │ action
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Log Collection                            │
│  - Action/observation pairs                                 │
│  - Agent contributions per decision                         │
│  - Performance metrics                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Reflection System (Hybrid)                     │
│  ┌─────────────────────┐  ┌──────────────────────┐        │
│  │  Rule-Based         │  │   LLM Analyzer       │        │
│  │  Failure Detector   │  │   (Suggests changes) │        │
│  └──────────┬──────────┘  └──────────┬───────────┘        │
│             └────────────┬────────────┘                    │
│                   Improvement Suggestions                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Flow Evolution Engine                          │
│  - Programmatic Langflow JSON generation                    │
│  - LLM proposes changes, code validates/implements          │
│  - Flow versioning and A/B testing                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            New Flow Variants Created                        │
│         → Test → Compare → Keep Best → Iterate             │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Approach: Langflow-Native

**KEY INSIGHT**: All agent communication, state management, and message passing happens WITHIN Langflow orchestrations, not as separate Python infrastructure.

### What Langflow Handles Natively:
- **Message passing**: Node connections and data flow
- **Shared state**: Memory components and session state
- **Agent coordination**: Flow structure defines how agents interact
- **Parallel queries**: Multiple LLM nodes executing in parallel

### What We Build in Python:
- **Reflection system**: Analyzes game logs, identifies failures
- **Flow evolution**: Generates/mutates Langflow JSON files
- **Performance tracking**: Compares flow variants
- **Prompt templates**: Agent instructions and schemas

## MVP Scope (Minimal Viable Prototype)

### Phase 1: Foundation (CURRENT)
- [x] Install mgrep for code searching
- [x] Create documentation structure (DESIGN.md, AGENTS.md, FLOWS.md)
- [ ] Update docs to clarify Langflow-native approach

### Phase 2: Society of Mind (Langflow Implementation)
- [ ] Design Langflow flow structure:
  - 4 specialized agent nodes (Navigator, Puzzle, Memory, Strategy)
  - Memory/state components for blackboard
  - Coordinator logic for synthesizing responses
- [ ] Create agent prompt templates
- [ ] Build v1_base flow (via Langflow UI or programmatic JSON)
- [ ] Test basic flow execution

### Phase 3: Reflection System
- [ ] Rule-based failure detector:
  - Action loops → memory problem
  - Low progress → navigation problem
  - Stuck on puzzle → need better puzzle-solving
- [ ] LLM analyzer:
  - Reads logs + failure reports
  - Suggests architectural improvements
  - "We need a maps department" style insights
- [ ] Post-game analyzer that runs automatically

### Phase 4: Flow Evolution
- [ ] Langflow JSON schema understanding
- [ ] Programmatic flow generation API
- [ ] Flow mutation engine (LLM suggests, code implements)
- [ ] Flow versioning system
- [ ] A/B testing framework (run variants in parallel)

### Phase 5: Integration
- [ ] Connect all components into feedback loop
- [ ] Enhanced logging (agent contributions, performance)
- [ ] Test on 2-3 games
- [ ] Verify system improves over iterations

## Key Design Decisions

### 1. Langflow-Native Communication (REVISED)
**Decision**: Use Langflow's built-in state management and node connections, not separate Python infrastructure
**Rationale**: Langflow already provides message passing, memory, and orchestration. Don't reinvent the wheel.
**Trade-off**: Must work within Langflow's paradigms, but much simpler and leverages existing tooling

### 2. Minimal Viable Prototype First
**Decision**: Build simplest version that demonstrates self-evolution
**Rationale**: Learn what works before building complex system
**Trade-off**: May need refactoring later, but faster initial learning

### 3. Hybrid Reflection (Rule-Based + LLM)
**Decision**: Combine pattern detection with AI analysis
**Rationale**: Rules catch obvious failures, LLM provides creative insights
**Trade-off**: More complex than pure LLM, but more reliable

### 4. Programmatic Flow Evolution (with LLM assist)
**Decision**: Code generates Langflow JSON, LLM proposes changes
**Rationale**: Full control + AI creativity, code validates correctness
**Trade-off**: Requires understanding Langflow schema, but flexible

### 5. Four Initial Agents
**Decision**: Navigator, Puzzle Solver, Memory, Strategy
**Rationale**: Cover main failure modes in text adventures
**Trade-off**: More complex than 2 agents, but necessary for real gameplay

## Technologies

- **Game Engine**: Jericho (Z-machine interpreter)
- **Orchestration**: Langflow (visual + programmatic)
- **LLMs**: Claude/Anthropic (via API)
- **Language**: Python 3.13
- **Search**: mgrep (fast multi-line grep)

## Success Metrics

1. **Self-Evolution Works**: System generates improved flows based on feedback
2. **Performance Improves**: Later iterations achieve higher game progress than early ones
3. **Architectural Discovery**: System identifies need for new agent types autonomously
4. **Collaboration Emerges**: Agents effectively coordinate to solve problems

## Evolution Log

This section tracks how the architecture evolves over time:

### 2025-11-24: Initial Design
- Starting with MVP plan
- 4-agent society-of-mind architecture
- Hybrid reflection system
- Programmatic flow evolution

---

## Next Steps

1. ~~Create documentation (DESIGN.md, AGENTS.md, FLOWS.md)~~ ✅ Done
2. Create agent prompt templates (prompts/ directory)
3. Design Langflow flow structure for 4-agent society
4. Build v1_base flow in Langflow (UI or JSON)
5. Implement reflection system (src/reflection/)
6. Implement flow evolution system (src/evolution/)
7. Test and iterate

## References

- Marvin Minsky: "The Society of Mind" (1986)
- Rich Sutton: "The Bitter Lesson" (2019)
- TextQuests Benchmark: https://github.com/microsoft/TextQuests
- Langflow Documentation: https://docs.langflow.org/
