"""Transport failures must not strand the single generation slot."""

from __future__ import annotations

import logging
from http import HTTPStatus

import pytest

from mlx_moe_stream.logging import configure_logging
from mlx_moe_stream.server.app import DISCONNECT_ERRORS, _GenerationStream


class _Slot:
    """Stand-in for the BoundedSemaphore plus its settled-once bookkeeping."""

    def __init__(self) -> None:
        self.held = True
        self.releases = 0

    def release(self) -> None:
        if not self.held:
            raise AssertionError("the generation slot was released twice")
        self.held = False
        self.releases += 1


def _events(slot: _Slot, *, items: tuple[str, ...] = ("a", "b")):
    """A generator shaped like _chat_events: it settles the slot on any exit."""

    try:
        for item in items:
            yield {"chunk": item}
    except BaseException:
        slot.release()
        raise
    else:
        slot.release()


def test_slot_is_released_when_the_stream_is_never_started():
    # The disconnect-before-first-event case that used to wedge the server:
    # the generator body never runs, so it can never release the slot itself.
    slot = _Slot()
    stream = _GenerationStream(_events(slot), slot.release)

    stream.close()

    assert not slot.held, "abandoning an unstarted stream must free the slot"
    assert slot.releases == 1


def test_slot_is_released_once_when_the_stream_is_abandoned_mid_flight():
    slot = _Slot()
    stream = _GenerationStream(_events(slot), slot.release)

    assert next(stream) == {"chunk": "a"}
    stream.close()

    assert not slot.held
    assert slot.releases == 1, "close() must not double-release after GeneratorExit"


def test_slot_is_released_once_when_the_stream_runs_to_completion():
    slot = _Slot()
    stream = _GenerationStream(_events(slot), slot.release)

    assert list(stream) == [{"chunk": "a"}, {"chunk": "b"}]
    stream.close()

    assert not slot.held
    assert slot.releases == 1


def test_close_is_idempotent():
    slot = _Slot()
    stream = _GenerationStream(_events(slot), slot.release)

    stream.close()
    stream.close()

    assert slot.releases == 1


def test_disconnect_errors_cover_the_socket_failures_we_tolerate():
    for error in (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
        assert issubclass(error, DISCONNECT_ERRORS)
    # A genuine server fault must still surface rather than be mistaken for a
    # client hanging up.
    assert not issubclass(ValueError, DISCONNECT_ERRORS)


def test_configure_logging_adds_a_rotating_file_handler(tmp_path):
    logger = logging.getLogger("mlx_moe_stream")
    saved = list(logger.handlers)
    logger.handlers.clear()
    try:
        target = tmp_path / "nested" / "serve.log"
        configure_logging(log_file=target, max_bytes=1024, backup_count=2)
        logger.info("hello")
        for handler in logger.handlers:
            handler.flush()

        assert target.exists(), "the parent directory should be created"
        assert "hello" in target.read_text(encoding="utf-8")

        rotating = [h for h in logger.handlers if hasattr(h, "maxBytes")]
        assert len(rotating) == 1
        assert rotating[0].maxBytes == 1024
        assert rotating[0].backupCount == 2

        # Reconfiguring must not stack a second file handler on the same logger.
        configure_logging(log_file=target)
        assert len([h for h in logger.handlers if hasattr(h, "maxBytes")]) == 1
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers[:] = saved


def test_serve_parser_exposes_the_stability_flags():
    from mlx_moe_stream.cli import build_parser

    args = build_parser().parse_args(
        [
            "serve",
            "--manifest",
            "m.json",
            "--log-file",
            "/tmp/x.log",
            "--pid-file",
            "/tmp/x.pid",
            "--connection-timeout",
            "5",
        ]
    )
    assert str(args.log_file) == "/tmp/x.log"
    assert str(args.pid_file) == "/tmp/x.pid"
    assert args.connection_timeout == pytest.approx(5.0)
    assert HTTPStatus.OK  # sanity: module imported cleanly
