from llm.generator import generate_answer

def test_generate_answer(monkeypatch):

    def fake_create(*args, **kwargs):
        class FakeResp:
            output_text = "Mocked response"
        return FakeResp()

    monkeypatch.setattr(
        "llm.generator.client.responses.create",
        fake_create
    )

    result = generate_answer("Q", "context")
    assert result == "Mocked response"
