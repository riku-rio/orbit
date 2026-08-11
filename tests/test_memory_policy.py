from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbit.mcp_tools.add_to_memory import register as register_add_to_memory
from orbit.mcp_tools.retrieve_from_memory import register as register_retrieve_from_memory
from orbit.memory.policy import MEMORY_SYSTEM_PROMPT


class FakeMCP:
    def __init__(self) -> None:
        self.tools = {}

    def tool(self):
        def decorator(function):
            self.tools[function.__name__] = function
            return function

        return decorator


class MemoryPolicyTests(unittest.TestCase):
    def test_policy_requires_retrieval_before_unknown_personal_fact(self):
        prompt = MEMORY_SYSTEM_PROMPT.lower()
        self.assertIn("what's my name?", prompt)
        self.assertIn("before saying that you do not know", prompt)
        self.assertIn("retrieve_from_memory", prompt)

    def test_policy_keeps_generic_questions_memory_free(self):
        prompt = MEMORY_SYSTEM_PROMPT.lower()
        self.assertIn("how do i center a div?", prompt)
        self.assertIn("do not retrieve memory", prompt)

    def test_policy_proactively_stores_durable_user_information(self):
        prompt = MEMORY_SYSTEM_PROMPT.lower()
        self.assertIn("do not wait for the user", prompt)
        self.assertIn("my name is yosef", prompt)
        self.assertIn("add_to_memory", prompt)

    def test_tool_descriptions_repeat_selection_rules(self):
        mcp = FakeMCP()
        register_add_to_memory(mcp)
        register_retrieve_from_memory(mcp)

        add_doc = (mcp.tools["add_to_memory"].__doc__ or "").lower()
        retrieve_doc = (mcp.tools["retrieve_from_memory"].__doc__ or "").lower()

        self.assertIn("even if they did not explicitly ask", add_doc)
        self.assertIn("my name is", add_doc)
        self.assertIn("before claiming that you do not know", retrieve_doc)
        self.assertIn("what's my name?", retrieve_doc)
        self.assertIn("how do i center a div?", retrieve_doc)


if __name__ == "__main__":
    unittest.main()
