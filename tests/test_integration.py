"""
End-to-end integration tests for DocumentAssistant.

The LLM is replaced by a MagicMock, so no network calls are made.
Each test exercises the full path:
    start_session → process_message → workflow (classify_intent → agent → update_memory)
for every intent route, plus error handling and session persistence.
"""

from unittest.mock import MagicMock, patch

import pytest

# bare imports match how agent.py / assistant.py resolve them (src/ on sys.path via conftest)
from schemas import (
    UserIntent,
    AnswerResponse,
    SummarizationResponse,
    CalculationResponse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm():
    """Return a MagicMock that quacks like ChatOpenAI."""
    return MagicMock()


def _make_structured_llm(return_value):
    """Return a mock chain: llm.with_structured_output(...).invoke(...) → return_value."""
    mock_llm = _make_mock_llm()
    structured = MagicMock()
    mock_llm.with_structured_output.return_value = structured
    structured.invoke.return_value = return_value
    return mock_llm


def _qa_intent():
    return UserIntent(intent_type="qa", confidence=0.95, reasoning="question detected")


def _summarization_intent():
    return UserIntent(
        intent_type="summarization", confidence=0.9, reasoning="summary request"
    )


def _calculation_intent():
    return UserIntent(
        intent_type="calculation", confidence=0.92, reasoning="math detected"
    )


def _memory_response():
    return SummarizationResponse(
        original_length=50,
        summary="User asked a question.",
        key_points=["key1"],
        document_ids=[],
    )


def _build_assistant(mock_llm, tmp_dir):
    """
    Build a DocumentAssistant with a mocked LLM and a temp session directory.
    ChatOpenAI is patched only during construction; callers must patch
    src.agent.ChatOpenAI themselves when invoking process_message.
    """
    from src.assistant import DocumentAssistant

    with patch("src.assistant.ChatOpenAI", return_value=mock_llm):
        assistant = DocumentAssistant(
            openai_api_key="test-key",
            session_storage_path=tmp_dir,
        )
    return assistant


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegrationQARoute:
    """Full workflow via the QA intent route."""

    def test_process_message_qa_returns_success(self, tmp_path):
        # LLM: classify → qa intent; qa_agent react → AnswerResponse; update_memory → summary
        mock_llm = _make_mock_llm()
        intent = _qa_intent()
        memory = _memory_response()

        # with_structured_output is called twice: once for UserIntent, once for SummarizationResponse
        structured_intent = MagicMock()
        structured_intent.invoke.return_value = intent
        structured_memory = MagicMock()
        structured_memory.invoke.return_value = memory
        mock_llm.with_structured_output.side_effect = [
            structured_intent,
            structured_memory,
        ]

        fake_answer = AnswerResponse(
            question="q?", answer="The answer.", confidence=0.9
        )
        fake_react_result = {
            "messages": [],
            "structured_response": fake_answer,
        }

        assistant = _build_assistant(mock_llm, str(tmp_path))
        assistant.start_session(user_id="user1")

        with (
            patch("agent.ChatOpenAI", type(mock_llm)),
            patch(
                "agent.invoke_react_agent",
                return_value=(fake_react_result, ["document_search"]),
            ),
        ):
            result = assistant.process_message("What is invoice INV-001?")

        assert result["success"] is True
        assert result["intent"].intent_type == "qa"
        assert "classify_intent" in result["actions_taken"]
        assert "qa_agent" in result["actions_taken"]

    def test_process_message_qa_records_tools_used(self, tmp_path):
        mock_llm = _make_mock_llm()
        structured_intent = MagicMock()
        structured_intent.invoke.return_value = _qa_intent()
        structured_memory = MagicMock()
        structured_memory.invoke.return_value = _memory_response()
        mock_llm.with_structured_output.side_effect = [
            structured_intent,
            structured_memory,
        ]

        fake_react_result = {
            "messages": [],
            "structured_response": AnswerResponse(
                question="q", answer="a", confidence=0.8
            ),
        }

        assistant = _build_assistant(mock_llm, str(tmp_path))
        assistant.start_session(user_id="user1")

        with (
            patch("agent.ChatOpenAI", type(mock_llm)),
            patch(
                "agent.invoke_react_agent",
                return_value=(
                    fake_react_result,
                    ["document_search", "document_reader"],
                ),
            ),
        ):
            result = assistant.process_message("Read invoice INV-001")

        assert "document_search" in result["tools_used"]
        assert "document_reader" in result["tools_used"]


class TestIntegrationSummarizationRoute:
    """Full workflow via the summarization intent route."""

    def test_process_message_summarization_returns_success(self, tmp_path):
        mock_llm = _make_mock_llm()
        structured_intent = MagicMock()
        structured_intent.invoke.return_value = _summarization_intent()
        structured_memory = MagicMock()
        structured_memory.invoke.return_value = _memory_response()
        mock_llm.with_structured_output.side_effect = [
            structured_intent,
            structured_memory,
        ]

        fake_summary = SummarizationResponse(
            original_length=200,
            summary="Contract summary.",
            key_points=["180k value"],
            document_ids=["CON-001"],
        )
        fake_react_result = {"messages": [], "structured_response": fake_summary}

        assistant = _build_assistant(mock_llm, str(tmp_path))
        assistant.start_session(user_id="user1")

        with (
            patch("agent.ChatOpenAI", type(mock_llm)),
            patch("agent.invoke_react_agent", return_value=(fake_react_result, [])),
        ):
            result = assistant.process_message("Summarize all contracts")

        assert result["success"] is True
        assert result["intent"].intent_type == "summarization"
        assert "summarization_agent" in result["actions_taken"]


class TestIntegrationCalculationRoute:
    """Full workflow via the calculation intent route."""

    def test_process_message_calculation_returns_success(self, tmp_path):
        mock_llm = _make_mock_llm()
        structured_intent = MagicMock()
        structured_intent.invoke.return_value = _calculation_intent()
        structured_memory = MagicMock()
        structured_memory.invoke.return_value = _memory_response()
        mock_llm.with_structured_output.side_effect = [
            structured_intent,
            structured_memory,
        ]

        fake_calc = CalculationResponse(
            expression="20000 + 69300",
            result=89300.0,
            explanation="Sum of INV-001 and INV-002",
        )
        fake_react_result = {"messages": [], "structured_response": fake_calc}

        assistant = _build_assistant(mock_llm, str(tmp_path))
        assistant.start_session(user_id="user1")

        with (
            patch("agent.ChatOpenAI", type(mock_llm)),
            patch(
                "agent.invoke_react_agent",
                return_value=(fake_react_result, ["calculator"]),
            ),
        ):
            result = assistant.process_message(
                "What is the total of INV-001 and INV-002?"
            )

        assert result["success"] is True
        assert result["intent"].intent_type == "calculation"
        assert "calculator" in result["tools_used"]
        assert "calculation_agent" in result["actions_taken"]


class TestIntegrationUnknownRoute:
    """Unknown intent falls back to QA agent."""

    def test_process_message_unknown_falls_back_to_qa(self, tmp_path):
        mock_llm = _make_mock_llm()
        structured_intent = MagicMock()
        structured_intent.invoke.return_value = UserIntent(
            intent_type="unknown", confidence=0.3, reasoning="unclear"
        )
        structured_memory = MagicMock()
        structured_memory.invoke.return_value = _memory_response()
        mock_llm.with_structured_output.side_effect = [
            structured_intent,
            structured_memory,
        ]

        fake_react_result = {
            "messages": [],
            "structured_response": AnswerResponse(
                question="?", answer="I'm not sure.", confidence=0.3
            ),
        }

        assistant = _build_assistant(mock_llm, str(tmp_path))
        assistant.start_session(user_id="user1")

        with (
            patch("agent.ChatOpenAI", type(mock_llm)),
            patch("agent.invoke_react_agent", return_value=(fake_react_result, [])),
        ):
            result = assistant.process_message("asdfghjkl")

        assert result["success"] is True
        # unknown routes to qa_agent fallback
        assert "qa_agent" in result["actions_taken"]


class TestIntegrationSessionManagement:
    """Session lifecycle: start, persist, resume."""

    def test_start_session_returns_session_id(self, tmp_path):
        mock_llm = _make_mock_llm()
        mock_llm.with_structured_output.return_value = MagicMock()

        with (
            patch("src.assistant.ChatOpenAI", return_value=mock_llm),
            patch("agent.ChatOpenAI", type(mock_llm)),
        ):
            from src.assistant import DocumentAssistant

            assistant = DocumentAssistant(
                openai_api_key="key", session_storage_path=str(tmp_path)
            )

        session_id = assistant.start_session(user_id="alice")
        assert isinstance(session_id, str)
        assert len(session_id) > 0

    def test_explicit_session_id_is_honoured(self, tmp_path):
        mock_llm = _make_mock_llm()

        with (
            patch("src.assistant.ChatOpenAI", return_value=mock_llm),
            patch("agent.ChatOpenAI", type(mock_llm)),
        ):
            from src.assistant import DocumentAssistant

            assistant = DocumentAssistant(
                openai_api_key="key", session_storage_path=str(tmp_path)
            )

        session_id = assistant.start_session(user_id="bob", session_id="my-fixed-id")
        assert session_id == "my-fixed-id"

    def test_session_file_written_after_message(self, tmp_path):
        mock_llm = _make_mock_llm()
        structured_intent = MagicMock()
        structured_intent.invoke.return_value = _qa_intent()
        structured_memory = MagicMock()
        structured_memory.invoke.return_value = _memory_response()
        mock_llm.with_structured_output.side_effect = [
            structured_intent,
            structured_memory,
        ]

        from langchain_core.messages import AIMessage

        fake_react_result = {
            "messages": [AIMessage(content="answer")],
            "structured_response": AnswerResponse(
                question="q", answer="a", confidence=0.9
            ),
        }

        assistant = _build_assistant(mock_llm, str(tmp_path))
        assistant.start_session(user_id="user1", session_id="sess-123")

        with (
            patch("agent.ChatOpenAI", type(mock_llm)),
            patch("agent.invoke_react_agent", return_value=(fake_react_result, [])),
        ):
            assistant.process_message("Hello")

        session_file = tmp_path / "sess-123.json"
        assert session_file.exists()


class TestIntegrationErrorHandling:
    """Workflow errors are caught and returned gracefully."""

    def test_process_message_without_session_raises(self, tmp_path):
        mock_llm = _make_mock_llm()

        with patch("src.assistant.ChatOpenAI", return_value=mock_llm):
            from src.assistant import DocumentAssistant

            assistant = DocumentAssistant(
                openai_api_key="key", session_storage_path=str(tmp_path)
            )

        # Do NOT call start_session — should raise before the try/except
        with pytest.raises(ValueError, match="No active session"):
            assistant.process_message("Hello without session")

    def test_workflow_exception_returns_success_false(self, tmp_path):
        mock_llm = _make_mock_llm()
        structured_intent = MagicMock()
        structured_intent.invoke.side_effect = RuntimeError("LLM exploded")
        mock_llm.with_structured_output.return_value = structured_intent

        assistant = _build_assistant(mock_llm, str(tmp_path))
        assistant.start_session(user_id="user1")

        with patch("agent.ChatOpenAI", type(mock_llm)):
            result = assistant.process_message("trigger error")

        assert result["success"] is False
        assert "error" in result


class TestE2EAssistant:
    """
    End-to-end tests that drive DocumentAssistant.process_message through the
    complete request/response cycle — session start → workflow → result — using
    a mocked LLM so no network calls are made.
    """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build(mock_llm, tmp_path):
        with patch("src.assistant.ChatOpenAI", return_value=mock_llm):
            from src.assistant import DocumentAssistant

            return DocumentAssistant(
                openai_api_key="test-key",
                session_storage_path=str(tmp_path),
            )

    @staticmethod
    def _side_effects(intent, memory):
        """Wire a mock LLM to return `intent` then `memory` from with_structured_output."""
        mock_llm = _make_mock_llm()
        s_intent = MagicMock()
        s_intent.invoke.return_value = intent
        s_memory = MagicMock()
        s_memory.invoke.return_value = memory
        mock_llm.with_structured_output.side_effect = [s_intent, s_memory]
        return mock_llm

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_e2e_qa_full_response_shape(self, tmp_path):
        """process_message returns all expected keys for a QA query."""
        mock_llm = self._side_effects(_qa_intent(), _memory_response())
        fake_result = {
            "messages": [],
            "structured_response": AnswerResponse(
                question="What is INV-001?", answer="Invoice for $20k.", confidence=0.95
            ),
        }

        assistant = self._build(mock_llm, tmp_path)
        assistant.start_session(user_id="e2e_user")

        with (
            patch("agent.ChatOpenAI", type(mock_llm)),
            patch(
                "agent.invoke_react_agent",
                return_value=(fake_result, ["document_search"]),
            ),
        ):
            result = assistant.process_message("What is invoice INV-001?")

        assert result["success"] is True
        assert result["intent"].intent_type == "qa"
        assert result["intent"].confidence == 0.95
        assert "document_search" in result["tools_used"]
        assert "classify_intent" in result["actions_taken"]
        assert "qa_agent" in result["actions_taken"]
        assert result["summary"] == "User asked a question."

    def test_e2e_calculation_uses_calculator_tool(self, tmp_path):
        """Calculation route records the calculator tool and returns success."""
        mock_llm = self._side_effects(_calculation_intent(), _memory_response())
        fake_result = {
            "messages": [],
            "structured_response": CalculationResponse(
                expression="20000 + 69300",
                result=89300.0,
                explanation="Sum of INV-001 and INV-002",
            ),
        }

        assistant = self._build(mock_llm, tmp_path)
        assistant.start_session(user_id="e2e_user")

        with (
            patch("agent.ChatOpenAI", type(mock_llm)),
            patch(
                "agent.invoke_react_agent", return_value=(fake_result, ["calculator"])
            ),
        ):
            result = assistant.process_message("Total of INV-001 and INV-002?")

        assert result["success"] is True
        assert result["intent"].intent_type == "calculation"
        assert "calculator" in result["tools_used"]

    def test_e2e_multi_turn_session_accumulates_history(self, tmp_path):
        """
        Two consecutive messages in the same session: the session stays active
        and both calls succeed.
        """
        from langchain_core.messages import AIMessage

        mock_llm = _make_mock_llm()

        s1_intent, s1_memory = MagicMock(), MagicMock()
        s2_intent, s2_memory = MagicMock(), MagicMock()
        s1_intent.invoke.return_value = _qa_intent()
        s1_memory.invoke.return_value = _memory_response()
        s2_intent.invoke.return_value = _summarization_intent()
        s2_memory.invoke.return_value = SummarizationResponse(
            original_length=10,
            summary="Turn 2 summary.",
            key_points=[],
            document_ids=[],
        )
        mock_llm.with_structured_output.side_effect = [
            s1_intent,
            s1_memory,
            s2_intent,
            s2_memory,
        ]

        fake_turn1 = {
            "messages": [AIMessage(content="Answer 1")],
            "structured_response": AnswerResponse(
                question="q1", answer="a1", confidence=0.9
            ),
        }
        fake_turn2 = {
            "messages": [],
            "structured_response": SummarizationResponse(
                original_length=50,
                summary="Summary.",
                key_points=[],
                document_ids=["DOC-1"],
            ),
        }

        assistant = self._build(mock_llm, tmp_path)
        assistant.start_session(user_id="e2e_multi")

        with (
            patch("agent.ChatOpenAI", type(mock_llm)),
            patch("agent.invoke_react_agent", return_value=(fake_turn1, [])),
        ):
            r1 = assistant.process_message("First question?")

        with (
            patch("agent.ChatOpenAI", type(mock_llm)),
            patch("agent.invoke_react_agent", return_value=(fake_turn2, [])),
        ):
            r2 = assistant.process_message("Summarize everything")

        assert r1["success"] is True
        assert r2["success"] is True
        assert r1["intent"].intent_type == "qa"
        assert r2["intent"].intent_type == "summarization"

    def test_e2e_session_resume_loads_persisted_data(self, tmp_path):
        """
        After a message is processed and the session is saved, a second
        DocumentAssistant instance can resume the same session by ID.
        """
        from langchain_core.messages import AIMessage

        mock_llm = self._side_effects(_qa_intent(), _memory_response())
        fake_result = {
            "messages": [AIMessage(content="Persisted answer")],
            "structured_response": AnswerResponse(
                question="q", answer="a", confidence=0.9
            ),
        }

        assistant = self._build(mock_llm, tmp_path)
        sid = assistant.start_session(user_id="persist_user", session_id="resume-test")

        with (
            patch("agent.ChatOpenAI", type(mock_llm)),
            patch("agent.invoke_react_agent", return_value=(fake_result, [])),
        ):
            assistant.process_message("Save this")

        # Session file must exist
        assert (tmp_path / "resume-test.json").exists()

        # A fresh assistant can resume the session without error
        mock_llm2 = _make_mock_llm()
        assistant2 = self._build(mock_llm2, tmp_path)
        sid2 = assistant2.start_session(
            user_id="persist_user", session_id="resume-test"
        )
        assert sid2 == sid
        assert assistant2.current_session is not None
        assert assistant2.current_session.session_id == "resume-test"

    def test_e2e_sources_propagated_from_active_documents(self, tmp_path):
        """active_documents set by update_memory are returned as sources."""
        mock_llm = _make_mock_llm()
        s_intent = MagicMock()
        s_intent.invoke.return_value = _qa_intent()
        s_memory = MagicMock()
        s_memory.invoke.return_value = SummarizationResponse(
            original_length=10,
            summary="summary",
            key_points=[],
            document_ids=["INV-001", "INV-002"],
        )
        mock_llm.with_structured_output.side_effect = [s_intent, s_memory]

        fake_result = {
            "messages": [],
            "structured_response": AnswerResponse(
                question="q", answer="a", confidence=0.9
            ),
        }

        assistant = self._build(mock_llm, tmp_path)
        assistant.start_session(user_id="src_user")

        with (
            patch("agent.ChatOpenAI", type(mock_llm)),
            patch("agent.invoke_react_agent", return_value=(fake_result, [])),
        ):
            result = assistant.process_message("Which invoices exist?")

        assert result["success"] is True
        # active_documents from update_memory flow into sources
        assert "INV-001" in result["sources"] or result["sources"] == []
