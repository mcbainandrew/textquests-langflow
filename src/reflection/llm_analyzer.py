"""
LLM-Based Reflection Analyzer

Takes failure reports from rule-based detector and uses LLM to:
- Generate creative insights about architectural improvements
- Suggest specific changes to prompts, flows, or agents
- Identify patterns that rules might miss
- Propose new agent types or structural changes

Design Choice: Use LLM for creative insights, not just pattern matching.
This is where "we need a maps department" style suggestions come from.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import os
from anthropic import Anthropic


@dataclass
class ImprovementSuggestion:
    """Structured suggestion for system improvement"""
    category: str  # "prompt_change", "new_agent", "flow_mutation", "parameter_tuning"
    priority: str  # "critical", "high", "medium", "low"
    description: str
    rationale: str  # Why this improvement is needed
    implementation: str  # How to implement it
    affected_components: List[str]  # Which agents/flows need changes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "priority": self.priority,
            "description": self.description,
            "rationale": self.rationale,
            "implementation": self.implementation,
            "affected_components": self.affected_components
        }


class LLMAnalyzer:
    """
    Uses LLM to analyze failure patterns and suggest improvements.

    Design choices:
    - Uses Claude (Anthropic) for analysis - good at reasoning
    - Structured output via JSON for parse-ability
    - Provides both failure reports AND raw game logs for context
    - Temperature = 0.7 for creative but sensible suggestions
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022"
    ):
        """
        Initialize LLM analyzer.

        Args:
            api_key: Anthropic API key (or from env ANTHROPIC_API_KEY)
            model: Claude model to use for analysis
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.client = Anthropic(api_key=self.api_key)
        self.model = model

    def analyze_failures(
        self,
        failure_reports: List[Dict[str, Any]],
        game_logs: Optional[List[Dict[str, Any]]] = None,
        current_architecture: Optional[str] = None
    ) -> List[ImprovementSuggestion]:
        """
        Analyze failures and generate improvement suggestions.

        Args:
            failure_reports: From FailureDetector.analyze_multiple_games()
            game_logs: Optional full game logs for deeper context
            current_architecture: Description of current system (from DESIGN.md)

        Returns:
            List of prioritized improvement suggestions
        """

        # Build analysis prompt
        prompt = self._build_analysis_prompt(
            failure_reports,
            game_logs,
            current_architecture
        )

        # Call Claude
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            temperature=0.7,
            system=self._get_system_prompt(),
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Parse response
        suggestions = self._parse_suggestions(response.content[0].text)

        return suggestions

    def _get_system_prompt(self) -> str:
        """System prompt defining LLM's role"""
        return """You are an AI systems architect analyzing the performance of a self-evolving agent orchestration system for playing text-based adventure games.

Your system uses a "society of mind" architecture with 4 specialized agents:
1. Navigator: Spatial reasoning and map tracking
2. Puzzle Solver: Object interactions and puzzle-solving
3. Memory Tracker: Action history and repetition prevention
4. Strategy Coordinator: High-level planning and final decisions

Your task is to analyze failure patterns from gameplay logs and suggest specific, actionable improvements. Focus on:
- Identifying root causes, not just symptoms
- Suggesting concrete changes (prompt modifications, new agents, flow changes)
- Prioritizing by impact and feasibility
- Being creative - suggest new agent types if needed (e.g., "need an NPC tracker")

Output your analysis as structured JSON matching the ImprovementSuggestion schema."""

    def _build_analysis_prompt(
        self,
        failure_reports: List[Dict[str, Any]],
        game_logs: Optional[List[Dict[str, Any]]],
        current_architecture: Optional[str]
    ) -> str:
        """Build the analysis prompt with all context"""

        prompt_parts = [
            "# Failure Analysis Task",
            "",
            "## Current Architecture",
            current_architecture or "4-agent society: Navigator, Puzzle Solver, Memory, Strategy Coordinator",
            "",
            "## Detected Failures",
            json.dumps(failure_reports, indent=2),
            ""
        ]

        if game_logs:
            # Include sample of game logs (first 10 steps of each game)
            prompt_parts.extend([
                "## Game Log Samples (first 10 steps per game)",
                ""
            ])

            for game_log in game_logs[:3]:  # Max 3 games to avoid token limit
                game_name = game_log.get("game_name", "unknown")
                logs = game_log.get("logs", [])[:10]

                prompt_parts.append(f"### {game_name}")
                for log in logs:
                    step = log.get("step", 0)
                    action = log.get("action", "")
                    obs = log.get("observation", "")[:100]
                    prompt_parts.append(f"Step {step}: {action} → {obs}...")

                prompt_parts.append("")

        prompt_parts.extend([
            "## Your Analysis Task",
            "",
            "Based on the failures detected, provide improvement suggestions in this JSON format:",
            "",
            "```json",
            "[",
            "  {",
            '    "category": "prompt_change" | "new_agent" | "flow_mutation" | "parameter_tuning",',
            '    "priority": "critical" | "high" | "medium" | "low",',
            '    "description": "Brief description of the improvement",',
            '    "rationale": "Why this is needed based on failure patterns",',
            '    "implementation": "Specific steps to implement this",',
            '    "affected_components": ["Navigator", "Memory", ...]',
            "  }",
            "]",
            "```",
            "",
            "Focus on:",
            "1. Most impactful changes first (critical/high priority)",
            "2. Specific, actionable improvements",
            "3. Root cause fixes, not band-aids",
            "4. Creative solutions - suggest new agents if needed",
            "",
            "Provide your analysis:"
        ])

        return "\n".join(prompt_parts)

    def _parse_suggestions(self, response_text: str) -> List[ImprovementSuggestion]:
        """Parse LLM response into structured suggestions"""

        # Extract JSON from response (handle markdown code blocks)
        json_text = response_text
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()

        try:
            suggestions_data = json.loads(json_text)

            suggestions = []
            for item in suggestions_data:
                suggestions.append(ImprovementSuggestion(
                    category=item.get("category", "unknown"),
                    priority=item.get("priority", "medium"),
                    description=item.get("description", ""),
                    rationale=item.get("rationale", ""),
                    implementation=item.get("implementation", ""),
                    affected_components=item.get("affected_components", [])
                ))

            return suggestions

        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse LLM suggestions: {e}")
            print(f"Response: {response_text[:500]}...")

            # Return empty list on parse failure
            return []

    def generate_reflection_report(
        self,
        failure_reports: List[Dict[str, Any]],
        suggestions: List[ImprovementSuggestion],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive reflection report.

        Args:
            failure_reports: From FailureDetector
            suggestions: From this analyzer
            output_path: Optional path to save report JSON

        Returns:
            Complete report dict
        """

        report = {
            "summary": {
                "total_failures": len(failure_reports),
                "failure_types": self._count_failure_types(failure_reports),
                "total_suggestions": len(suggestions),
                "critical_suggestions": len([s for s in suggestions if s.priority == "critical"]),
                "high_priority_suggestions": len([s for s in suggestions if s.priority == "high"])
            },
            "failures": failure_reports,
            "suggestions": [s.to_dict() for s in suggestions],
            "next_steps": self._generate_next_steps(suggestions)
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Reflection report saved to {output_path}")

        return report

    def _count_failure_types(self, failure_reports: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count failures by type"""
        from collections import Counter
        types = [f.get("failure_type", "unknown") for f in failure_reports]
        return dict(Counter(types))

    def _generate_next_steps(self, suggestions: List[ImprovementSuggestion]) -> List[str]:
        """Generate prioritized action items"""

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_suggestions = sorted(
            suggestions,
            key=lambda s: priority_order.get(s.priority, 4)
        )

        next_steps = []
        for i, suggestion in enumerate(sorted_suggestions[:5], 1):  # Top 5
            next_steps.append(
                f"[{suggestion.priority.upper()}] {suggestion.description}: {suggestion.implementation}"
            )

        return next_steps


if __name__ == "__main__":
    # Example usage
    import sys
    from .failure_detector import FailureDetector, load_run_results

    if len(sys.argv) < 2:
        print("Usage: python llm_analyzer.py <path_to_run_dir>")
        sys.exit(1)

    run_dir = sys.argv[1]

    # Load and analyze failures
    print("Loading game logs...")
    game_logs = load_run_results(run_dir)

    print("Running rule-based failure detection...")
    detector = FailureDetector()
    failure_report = detector.analyze_multiple_games(game_logs)

    # Analyze with LLM
    print("Analyzing failures with LLM...")
    analyzer = LLMAnalyzer()

    suggestions = analyzer.analyze_failures(
        failure_reports=failure_report["all_failures"],
        game_logs=game_logs
    )

    # Generate report
    report = analyzer.generate_reflection_report(
        failure_reports=failure_report["all_failures"],
        suggestions=suggestions,
        output_path=f"{run_dir}/reflection_report.json"
    )

    print(f"\n=== Reflection Report ===")
    print(f"Total failures: {report['summary']['total_failures']}")
    print(f"Total suggestions: {report['summary']['total_suggestions']}")
    print(f"\n=== Top Suggestions ===")
    for i, step in enumerate(report['next_steps'], 1):
        print(f"{i}. {step}")
