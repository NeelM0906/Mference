import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "swift_qwen_task_screen.py"
spec = importlib.util.spec_from_file_location("screen", path)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def response(content, finish="stop", calls=None):
    return {"choices": [{"finish_reason": finish, "message": {"content": content, "tool_calls": calls or []}}]}


class ScreenRubricTests(unittest.TestCase):
    def test_exact_content_and_incomplete_answers(self):
        self.assertTrue(screen.score({"expected": "4"}, response("\n4\n"))[0])
        self.assertFalse(screen.score({"expected": "4"}, response("4", "length"))[0])
        self.assertFalse(screen.score({"expected": "4"}, response("Answer: 4"))[0])
        self.assertFalse(screen.score({"expected": "4"}, {"error": "failure"})[0])

    def test_json_types_extra_fields_and_fences(self):
        case = {"expected_json": {"ok": True, "count": 3}}
        self.assertTrue(screen.score(case, response('{"count":3,"ok":true}'))[0])
        for text in ('{"count":3,"ok":1}', '{"count":3,"ok":true,"extra":0}', '```json\n{"count":3,"ok":true}\n```'):
            self.assertFalse(screen.score(case, response(text))[0])

    def test_tool_call_shape_name_and_arguments(self):
        case = {"expected_call": {"name": "add", "arguments": {"a": 2}}}
        call = {"function": {"name": "add", "arguments": '{"a":2}'}}
        self.assertTrue(screen.score(case, response(None, "tool_calls", [call]))[0])
        self.assertFalse(screen.score(case, response(None, "tool_calls", [call, call]))[0])
        self.assertFalse(screen.score(case, response(None, "stop", [call]))[0])


if __name__ == "__main__":
    unittest.main()
