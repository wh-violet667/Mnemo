import config


def test_opencode_go_headers_and_reasoning(monkeypatch):
    monkeypatch.setenv("BASE_URL", "https://opencode.ai/inference/go/openai/v1")
    monkeypatch.setenv("OPENCODE_SESSION_ID", "ses_test_000000000000000000")
    monkeypatch.setenv("OPENCODE_REASONING_EFFORT", "none")

    headers = config.llm_default_headers()
    assert headers["x-opencode-session"] == "ses_test_000000000000000000"
    assert config.chat_model_extra_body("deepseek-v4.1-flash") == {
        "reasoning_effort": "none"
    }


def test_dashscope_deepseek_legacy_switch(monkeypatch):
    monkeypatch.setenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("OPENCODE_SESSION_ID", "ses_test")
    assert config.llm_default_headers() == {}
    assert config.chat_model_extra_body("deepseek-v4.1-flash") == {
        "enable_thinking": False
    }
