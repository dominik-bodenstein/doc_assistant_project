import math
import tempfile

import pytest

from src.tools import ToolLogger, create_calculator_tool


@pytest.fixture
def calculator_with_logger():
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = ToolLogger(logs_dir=tmp_dir, session_id="unit_test")
        calculator = create_calculator_tool(logger)
        yield calculator, logger


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"a": 5.0, "b": 3.0, "operation": "add"}, 8.0),
        ({"a": 10.0, "b": 4.0, "operation": "subtract"}, 6.0),
        ({"a": 6.0, "b": 7.0, "operation": "multiply"}, 42.0),
        ({"a": 20.0, "b": 5.0, "operation": "divide"}, 4.0),
    ],
)
def test_calculator_basic_operations(calculator_with_logger, payload, expected):
    calculator, _ = calculator_with_logger
    result = calculator.invoke(payload)
    assert result == expected


def test_calculator_divide_by_zero_returns_nan_and_logs_error(calculator_with_logger):
    calculator, logger = calculator_with_logger
    result = calculator.invoke({"a": 20.0, "b": 0.0, "operation": "divide"})

    assert math.isnan(result)
    assert len(logger.get_logs()) == 1

    log_entry = logger.get_logs()[0]
    assert log_entry["tool_name"] == "calculator"
    assert log_entry["input"]["operation"] == "divide"
    assert "Cannot divide by zero" in log_entry["output"]


def test_calculator_logs_successful_operation(calculator_with_logger):
    calculator, logger = calculator_with_logger
    result = calculator.invoke({"a": 2.0, "b": 3.0, "operation": "add"})

    assert result == 5.0
    assert len(logger.get_logs()) == 1

    log_entry = logger.get_logs()[0]
    assert log_entry["tool_name"] == "calculator"
    assert log_entry["input"]["operation"] == "add"
    assert "5.0" in log_entry["output"]


def test_calculator_unsupported_operation_fails_validation(calculator_with_logger):
    calculator, logger = calculator_with_logger
    with pytest.raises(Exception) as context:
        calculator.invoke({"a": 2.0, "b": 3.0, "operation": "power"})

    assert "Input should be 'add', 'subtract', 'multiply' or 'divide'" in str(
        context.value
    )
    assert len(logger.get_logs()) == 0
