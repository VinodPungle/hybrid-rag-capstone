from llm.generator import generate_answer


def _make_mock_response(content="Mocked response", prompt_tokens=100, completion_tokens=50):
    """Create a mock OpenAI chat completion response."""
    class Usage:
        def __init__(self):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = prompt_tokens + completion_tokens

    class Message:
        def __init__(self):
            self.content = content

    class Choice:
        def __init__(self):
            self.message = Message()

    class Response:
        def __init__(self):
            self.choices = [Choice()]
            self.usage = Usage()

    return Response()


def test_generate_answer(monkeypatch):
    monkeypatch.setattr(
        "llm.generator.client.chat.completions.create",
        lambda *args, **kwargs: _make_mock_response(),
    )

    result = generate_answer("What is X?", "X is a concept.")
    assert result == "Mocked response"


def test_generate_answer_uses_context(monkeypatch):
    """Verify the prompt includes both query and context."""
    captured = {}

    def capture_create(*args, **kwargs):
        captured["messages"] = kwargs.get("messages", args[1] if len(args) > 1 else [])
        return _make_mock_response()

    monkeypatch.setattr(
        "llm.generator.client.chat.completions.create",
        capture_create,
    )

    generate_answer("my question", "my context")
    user_msg = captured["messages"][-1]["content"]
    assert "my question" in user_msg
    assert "my context" in user_msg
