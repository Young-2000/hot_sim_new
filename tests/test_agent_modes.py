import pytest

from agent import _cli_response_text, _json_from_text, _response_text, codex_plan_request
from schemas import AgentRequest


def test_agent_request_mode_defaults_to_local():
    assert AgentRequest(model_id='model_1', prompt='热源').mode == 'local'
    assert AgentRequest(model_id='model_1', prompt='热源', mode='codex').mode == 'codex'
    with pytest.raises(ValueError):
        AgentRequest(model_id='model_1', prompt='热源', mode='remote')


def test_codex_mode_reports_missing_key_without_local_fallback(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('THERMAL_CODEX_PROVIDER', 'api')
    result = codex_plan_request('missing-model', '铝件 10 W', {})
    assert result['ok'] is False
    assert result['mode'] == 'codex'
    assert 'OPENAI_API_KEY' in result['questions'][0]


def test_extract_structured_response_json():
    payload = {'output': [{'content': [{'type': 'output_text', 'text': '{"model_id":"x"}'}]}]}
    assert _json_from_text(_response_text(payload)) == {'model_id': 'x'}
    assert _json_from_text('```json\n{"ok": true}\n```') == {'ok': True}


def test_extract_cli_jsonl_agent_message():
    output = '\n'.join([
        '{"type":"thread.started","thread_id":"t"}',
        '{"type":"item.completed","item":{"type":"agent_message","text":"{\\"model_id\\":\\"x\\"}"}}',
    ])
    assert _cli_response_text(output) == '{"model_id":"x"}'
