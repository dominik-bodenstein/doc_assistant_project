from unittest.mock import MagicMock, patch
from src.agent import (
    classify_intent,
    qa_agent,
    summarization_agent,
    calculation_agent,
    update_memory,
    should_continue,
)

# Import from 'schemas' (bare) so the same class objects are used as in agent.py,
# which also does `from schemas import ...` with src/ on sys.path (via conftest).
from schemas import (
    UserIntent,
    AnswerResponse,
    SummarizationResponse,
    CalculationResponse,
)


def _make_base_state():
    return {
        "user_input": "test input",
        "messages": [],
        "conversation_summary": "",
        "tools_used": [],
        "actions_taken": [],
        "next_step": "",
    }


def _make_config(mock_llm, tools=None):
    return {"configurable": {"llm": mock_llm, "tools": tools or []}}


def _make_mock_llm():
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    return mock_llm


# --- classify_intent tests ---


def test_classify_intent_routes_to_qa():
    mock_llm = _make_mock_llm()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = UserIntent(
        intent_type="qa", confidence=0.9, reasoning="test"
    )

    state = _make_base_state()
    config = _make_config(mock_llm)

    with patch("src.agent.ChatOpenAI", type(mock_llm)):
        result = classify_intent(state, config)

    assert result["next_step"] == "qa_agent"


def test_classify_intent_routes_to_summarization():
    mock_llm = _make_mock_llm()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = UserIntent(
        intent_type="summarization", confidence=0.9, reasoning="test"
    )

    state = _make_base_state()
    config = _make_config(mock_llm)

    with patch("src.agent.ChatOpenAI", type(mock_llm)):
        result = classify_intent(state, config)

    assert result["next_step"] == "summarization_agent"


def test_classify_intent_routes_to_calculation():
    mock_llm = _make_mock_llm()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = UserIntent(
        intent_type="calculation", confidence=0.9, reasoning="test"
    )

    state = _make_base_state()
    config = _make_config(mock_llm)

    with patch("src.agent.ChatOpenAI", type(mock_llm)):
        result = classify_intent(state, config)

    assert result["next_step"] == "calculation_agent"


def test_classify_intent_routes_unknown_to_qa():
    mock_llm = _make_mock_llm()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = UserIntent(
        intent_type="unknown", confidence=0.4, reasoning="unclear"
    )

    state = _make_base_state()
    config = _make_config(mock_llm)

    with patch("src.agent.ChatOpenAI", type(mock_llm)):
        result = classify_intent(state, config)

    assert result["next_step"] == "qa_agent"


# --- agent node tests ---


def test_qa_agent_populates_response():
    mock_llm = _make_mock_llm()
    state = _make_base_state()
    config = _make_config(mock_llm)

    fake_response = {
        "messages": [],
        "structured_response": AnswerResponse(question="q", answer="a", confidence=0.9),
    }
    fake_tools_used = ["calculator"]

    with (
        patch("src.agent.ChatOpenAI", type(mock_llm)),
        patch(
            "src.agent.invoke_react_agent",
            return_value=(fake_response, fake_tools_used),
        ),
    ):
        result = qa_agent(state, config)

    assert result["next_step"] == "update_memory"
    assert result["tools_used"] == ["calculator"]


def test_summarization_agent_populates_response():
    mock_llm = _make_mock_llm()
    state = _make_base_state()
    config = _make_config(mock_llm)

    fake_response = {
        "messages": [],
        "structured_response": SummarizationResponse(
            original_length=100,
            summary="A summary",
            key_points=["p1"],
            document_ids=["d1"],
        ),
    }
    fake_tools_used = []

    with (
        patch("src.agent.ChatOpenAI", type(mock_llm)),
        patch(
            "src.agent.invoke_react_agent",
            return_value=(fake_response, fake_tools_used),
        ),
    ):
        result = summarization_agent(state, config)

    assert result["next_step"] == "update_memory"


def test_calculation_agent_populates_response():
    mock_llm = _make_mock_llm()
    state = _make_base_state()
    config = _make_config(mock_llm)

    fake_response = {
        "messages": [],
        "structured_response": CalculationResponse(
            expression="2+2",
            result=4.0,
            explanation="Basic addition",
        ),
    }
    fake_tools_used = ["calculator"]

    with (
        patch("src.agent.ChatOpenAI", type(mock_llm)),
        patch(
            "src.agent.invoke_react_agent",
            return_value=(fake_response, fake_tools_used),
        ),
    ):
        result = calculation_agent(state, config)

    assert result["next_step"] == "update_memory"


# --- update_memory test ---


def test_update_memory_updates_summary():
    mock_llm = _make_mock_llm()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = SummarizationResponse(
        original_length=100,
        summary="New summary",
        key_points=["p1"],
        document_ids=["d1"],
    )

    state = _make_base_state()
    config = _make_config(mock_llm)

    with patch("src.agent.ChatOpenAI", type(mock_llm)):
        result = update_memory(state, config)

    assert result["conversation_summary"] == "New summary"


# --- should_continue test ---


def test_should_continue_routes_correctly():
    assert should_continue({"next_step": "qa_agent"}) == "qa_agent"
    assert should_continue({}) == "end"
    assert should_continue({"next_step": "update_memory"}) == "update_memory"


# --- Phase 3: Prompt polish & debug cleanup tests ---


def test_calculation_prompt_no_typos():
    from src.prompts import CALCULATION_SYSTEM_PROMPT

    assert "Wllays" not in CALCULATION_SYSTEM_PROMPT
    assert "necassary" not in CALCULATION_SYSTEM_PROMPT
    assert "caluculated" not in CALCULATION_SYSTEM_PROMPT


def test_create_workflow_compiles(capsys):
    from src.agent import create_workflow

    mock_llm = _make_mock_llm()
    with patch("src.agent.ChatOpenAI", type(mock_llm)):
        result = create_workflow(mock_llm, [])
    assert result is not None
    captured = capsys.readouterr()
    assert captured.out == ""
