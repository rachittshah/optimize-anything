from optimize_anything.proposer.prompts import build_reflection_prompt, extract_candidate


def test_build_reflection_prompt():
    prompt = build_reflection_prompt(
        current_text="You are a helpful assistant",
        component_name="system_prompt",
        objective="Optimize for helpfulness",
        asi_entries=[
            {"score": 0.5, "feedback": {"error": "Too vague"}},
            {"score": 0.8, "feedback": {"note": "Good but verbose"}},
        ],
        background=None,
    )
    assert "system_prompt" in prompt
    assert "Too vague" in prompt
    assert "Good but verbose" in prompt
    assert "Optimize for helpfulness" in prompt


def test_extract_candidate_from_fenced():
    text = '''Here is my improved version:

```
You are a precise, helpful assistant that always provides sources.
```

This improves clarity.'''
    result = extract_candidate(text)
    assert result == "You are a precise, helpful assistant that always provides sources."


def test_extract_candidate_no_fence():
    text = "You are a precise, helpful assistant."
    result = extract_candidate(text)
    assert result == "You are a precise, helpful assistant."
