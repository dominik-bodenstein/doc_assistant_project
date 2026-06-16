import tempfile
import pytest
from src.tools import ToolLogger, create_calculator_tool


@pytest.fixture
def calculator_with_logger():
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = ToolLogger(logs_dir=tmp_dir, session_id="unit_test")
        calc = create_calculator_tool(logger)
        yield calc, logger


def test_calculator_expression_addition(calculator_with_logger):
    calc, _ = calculator_with_logger
    assert calc.invoke({"expression": "2 + 3"}) == "5.0"


def test_calculator_expression_subtraction(calculator_with_logger):
    calc, _ = calculator_with_logger
    assert calc.invoke({"expression": "10 - 4"}) == "6.0"


def test_calculator_expression_multiplication(calculator_with_logger):
    calc, _ = calculator_with_logger
    assert calc.invoke({"expression": "6 * 7"}) == "42.0"


def test_calculator_expression_division(calculator_with_logger):
    calc, _ = calculator_with_logger
    assert calc.invoke({"expression": "20 / 5"}) == "4.0"


def test_calculator_expression_compound(calculator_with_logger):
    calc, _ = calculator_with_logger
    assert calc.invoke({"expression": "(2 + 3) * 4"}) == "20.0"


def test_calculator_divide_by_zero_returns_error_string(calculator_with_logger):
    calc, logger = calculator_with_logger
    result = calc.invoke({"expression": "10 / 0"})
    assert isinstance(result, str)
    assert result.startswith("Error")
    assert len(logger.get_logs()) == 1


def test_calculator_invalid_expression_returns_error_string(calculator_with_logger):
    calc, _ = calculator_with_logger
    result = calc.invoke({"expression": "abc + 1"})
    assert isinstance(result, str)
    assert result.startswith("Error")


def test_calculator_injection_attempt_returns_error_string(calculator_with_logger):
    calc, _ = calculator_with_logger
    result = calc.invoke({"expression": "__import__('os').system('ls')"})
    assert isinstance(result, str)
    assert result.startswith("Error")


def test_calculator_logs_success(calculator_with_logger):
    calc, logger = calculator_with_logger
    calc.invoke({"expression": "3 + 4"})
    assert len(logger.get_logs()) == 1
    log = logger.get_logs()[0]
    assert log["tool_name"] == "calculator"
    assert "7.0" in log["output"]


def test_calculator_logs_error(calculator_with_logger):
    calc, logger = calculator_with_logger
    calc.invoke({"expression": "bad expression"})
    assert len(logger.get_logs()) == 1
    log = logger.get_logs()[0]
    assert log["tool_name"] == "calculator"
    assert "Error" in log["output"]
