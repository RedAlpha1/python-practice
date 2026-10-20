import time

import pytest

from ai_lab.utils import Timer, timer


def test_timer_records_elapsed():
    with Timer("sleep") as t:
        time.sleep(0.1)
    assert t.elapsed >= 0.1


def test_timer_does_not_swallow_errors():
    with pytest.raises(ValueError), Timer("boom"):
        raise ValueError("boom")


def test_timer_logs_even_on_error(caplog):
    caplog.set_level("INFO")
    with pytest.raises(ValueError), Timer("boom"):
        raise ValueError("boom")
    assert "boom took" in caplog.text


def test_generator_timer_logs(caplog):
    caplog.set_level("INFO")
    with timer("sleep"):
        time.sleep(0.01)
    assert "sleep took" in caplog.text


def test_generator_timer_does_not_swallow_errors():
    with pytest.raises(ValueError), timer("boom"):
        raise ValueError("boom")


def test_generator_timer_logs_even_on_error(caplog):
    caplog.set_level("INFO")
    with pytest.raises(ValueError), timer("boom"):
        raise ValueError("boom")
    assert "boom took" in caplog.text
