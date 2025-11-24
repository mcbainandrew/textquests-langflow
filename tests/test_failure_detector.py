import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.reflection.failure_detector import FailureDetector, FailureReport

class TestFailureDetector(unittest.TestCase):
    def setUp(self):
        self.detector = FailureDetector(
            loop_threshold=3,
            stagnation_threshold=5, # Lower for testing
            max_tries_threshold=2
        )

    def test_detect_action_loop(self):
        actions = ["look", "north", "look", "north", "look", "north", "look", "north"]
        # Should detect loop of "look", "north" or just "look" depending on implementation
        # The current implementation checks for exact repetition of single actions or alternating
        
        # Test exact loop
        actions_exact = ["wait", "wait", "wait", "wait"]
        failures = self.detector._detect_action_loops(actions_exact)
        self.assertTrue(len(failures) > 0)
        self.assertEqual(failures[0].failure_type, "exact_action_loop")

    def test_detect_stagnation(self):
        logs = []
        for i in range(10):
            logs.append({"step": i, "progress": 0, "score": 0})
            
        failures = self.detector._detect_stagnation(logs)
        self.assertTrue(len(failures) > 0)
        self.assertEqual(failures[0].failure_type, "stagnation")

    def test_detect_repeated_failure(self):
        logs = [
            {"step": 1, "action": "jump", "observation": "You can't jump here."},
            {"step": 2, "action": "jump", "observation": "You can't jump here."},
            {"step": 3, "action": "jump", "observation": "You can't jump here."}
        ]
        
        failures = self.detector._detect_repeated_failures(logs)
        self.assertTrue(len(failures) > 0)
        self.assertEqual(failures[0].failure_type, "repeated_failure")
        self.assertEqual(failures[0].evidence["action"], "jump")

if __name__ == '__main__':
    unittest.main()
