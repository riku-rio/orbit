from __future__ import annotations

import os
import sys
import unittest
from collections import UserDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbit.memory.models import MemoryModels


class BatchEncodingStyleTokenizer:
    def __call__(self, text: str, **kwargs):
        del text, kwargs
        return UserDict({"input_ids": [101, 2001, 102]})


class FakeModel:
    tokenizer = BatchEncodingStyleTokenizer()


class MemoryTokenizerTests(unittest.TestCase):
    def test_token_count_accepts_batch_encoding_style_mapping(self):
        self.assertEqual(MemoryModels._token_count(FakeModel(), "hello"), 3)


if __name__ == "__main__":
    unittest.main()
