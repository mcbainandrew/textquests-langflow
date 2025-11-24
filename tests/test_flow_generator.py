import unittest
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evolution.flow_generator import FlowGenerator, LangflowNode, LangflowEdge

class TestFlowGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = FlowGenerator("Test Flow")

    def test_node_creation(self):
        node_id = self.generator.add_chat_input("Test Input")
        self.assertTrue(node_id.startswith("input-"))
        self.assertEqual(len(self.generator.nodes), 1)
        self.assertEqual(self.generator.nodes[0].type, "ChatInput")

    def test_edge_creation(self):
        node1 = self.generator.add_chat_input("Input")
        node2 = self.generator.add_chat_output("Output")
        self.generator.add_edge(node1, node2)
        
        self.assertEqual(len(self.generator.edges), 1)
        self.assertEqual(self.generator.edges[0].source, node1)
        self.assertEqual(self.generator.edges[0].target, node2)

    def test_json_generation(self):
        self.generator.add_chat_input("Input")
        flow_json = self.generator.generate()
        
        self.assertEqual(flow_json["name"], "Test Flow")
        self.assertIn("nodes", flow_json["data"])
        self.assertIn("edges", flow_json["data"])
        self.assertEqual(len(flow_json["data"]["nodes"]), 1)

    def test_add_agent(self):
        # Create a dummy prompt file
        os.makedirs("prompts", exist_ok=True)
        with open("prompts/test_agent.txt", "w") as f:
            f.write("You are a {role}.")
            
        try:
            prompt_id, llm_id, parser_id = self.generator.add_agent(
                "TestAgent", 
                "prompts/test_agent.txt"
            )
            
            self.assertEqual(len(self.generator.nodes), 3) # Prompt, LLM, Code(Parser)
            self.assertEqual(len(self.generator.edges), 2) # Prompt->LLM, LLM->Parser
            
        finally:
            # Cleanup
            if os.path.exists("prompts/test_agent.txt"):
                os.remove("prompts/test_agent.txt")

if __name__ == '__main__':
    unittest.main()
