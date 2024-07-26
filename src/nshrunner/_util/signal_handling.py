import contextlib
import logging
import signal
import types
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

log = logging.getLogger(__name__)

Signal: TypeAlias = int | signal.Signals
SignalHandler: TypeAlias = (
    Callable[[int, types.FrameType | None], Any] | int | signal.Handlers | None
)
SignalHandlers: TypeAlias = Mapping[Signal, Sequence[SignalHandler]]


def _with_signal_handlers(handlers: Mapping[Signal, Sequence[SignalHandler]]):
    @contextlib.contextmanager
    def signal_handler_context():
        original_handlers = {}
        combined_handlers = {}

        for sig, handler_list in handlers.items():
            # Combine multiple handlers for the same signal
            def combined_handler(signum, frame):
                for handler in handler_list:
                    if callable(handler):
                        handler(signum, frame)
                    elif isinstance(handler, int):
                        signal.default_int_handler(signum, frame)
                    # Handle other cases (like signal.SIG_IGN or signal.SIG_DFL) as needed

            combined_handlers[sig] = combined_handler
            original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, combined_handler)

            log.info(f"Registered {len(handler_list)} handlers for signal {sig}")
        try:
            yield
        finally:
            # Restore original signal handlers
            for sig, original_handler in original_handlers.items():
                signal.signal(sig, original_handler)

    return signal_handler_context()
