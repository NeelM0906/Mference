"""Print deterministic reference renders; no model or tokenizer download.

uv run --no-project --with jinja2==3.1.6 Scripts/swift_qwen_template_oracle.py
"""
import json
from pathlib import Path
from jinja2.sandbox import ImmutableSandboxedEnvironment

root = Path(__file__).resolve().parents[1]
template = root / 'Tests/Mference/Core/Tokenization/Fixtures/SwiftQwenTemplate/chat_template.jinja'
environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
def fail(message):
    raise ValueError(message)
environment.globals['raise_exception'] = fail
# Transformers' tojson differs from Jinja's HTML-escaping default.
environment.filters['tojson'] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
render = environment.from_string(template.read_text()).render
histories = {
    'single': [{'role': 'user', 'content': ' Hi '}],
    'history': [{'role': 'system', 'content': ' Be terse. '},
                {'role': 'user', 'content': 'A'},
                {'role': 'assistant', 'content': 'B', 'reasoning_content': ' Check A. '},
                {'role': 'user', 'content': 'C'}],
    'tool_result': [{'role': 'user', 'content': 'Lookup A'},
                    {'role': 'assistant', 'content': None, 'reasoning_content': 'Need lookup.',
                     'tool_calls': [{'function': {'name': 'lookup', 'arguments': {'query': 'A'}}}]},
                    {'role': 'tool', 'content': 'Found A'}],
}
result = []
for name, messages in histories.items():
    for effort in ['xhigh', 'medium', 'low', 'none']:
        result.append({'name': name, 'effort': effort,
                       'render': render(messages=messages, tools=[], add_generation_prompt=True,
                                        enable_thinking=effort != 'none', reasoning_effort=effort,
                                        preserve_thinking=True)})
print(json.dumps(result, ensure_ascii=False, indent=2))
