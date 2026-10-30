"""
Per-Pipeline profile policy stack.

Middleware that nests around a block of Pipeline usage can temporarily tighten
how that frame behaves along five independent dimensions:

* ``exec_mode`` - whether ``execute()`` packs a MULTI/EXEC transactional block,
  a bare pipeline, or is forbidden entirely.
* ``watch_dispatch`` - whether WATCH/pre-MULTI reading commands are sent
  immediately or buffered on the command stack, and whether WATCH is allowed.
* ``script_preload`` - whether queued scripts are preloaded with immediate
  SCRIPT EXISTS/SCRIPT LOAD calls, whether preloading is skipped, and whether
  buffering SCRIPT EXISTS/SCRIPT LOAD on the command stack is forbidden.
* ``watch_fault`` - whether a broken watch (EXEC returning nil / a connection
  dying while watching) surfaces as ``WatchError`` or keeps the original
  connection error type.
* ``stack_empty`` - whether an empty ``execute()`` short-circuits or still
  performs a no-op reset that releases a held connection.

The stack is owned by a single Pipeline instance (sync ``redis.client.Pipeline``
or ``redis.asyncio.client.Pipeline``). It is deliberately independent of
connection-pool leases, Cluster MOVED/ASK topology, Sentinel discovery and
PubSub reconnect handling: frames never move between Pipeline instances and the
empty stack reproduces the Pipeline's historical behavior exactly.
"""

from enum import Enum
from typing import Any, Iterable, Optional, Tuple

from redis.exceptions import PipelinePolicyError

__all__ = [
    "ExecMode",
    "WatchDispatch",
    "ScriptPreload",
    "WatchFault",
    "StackEmptyMode",
    "PipelinePolicyFrame",
    "PipelinePolicyStack",
    "EffectivePipelinePolicy",
]


class ExecMode(str, Enum):
    """How ``execute()`` packs the queued commands."""

    #: Follow the Pipeline's ``transaction`` flag / explicit ``multi()``.
    PIN = "pin"
    #: Always execute through ``_execute_transaction`` (MULTI ... EXEC).
    MULTI_EXEC = "multi_exec"
    #: Always execute through ``_execute_pipeline`` (no MULTI/EXEC on the wire).
    BARE_PIPELINE = "bare_pipeline"
    #: ``execute()`` is forbidden for the lifetime of the frame.
    FORBID_EXECUTE = "forbid_execute"


class WatchDispatch(str, Enum):
    """How WATCH and pre-MULTI commands leave the Pipeline."""

    #: Immediate WATCH/pre-MULTI dispatch (the historical behavior).
    PIN = "pin"
    #: WATCH / commands issued while watching before MULTI are sent now.
    IMMEDIATE_PRE_MULTI = "immediate_pre_multi"
    #: The pre-MULTI immediate shortcut is disabled; commands are buffered.
    STACK_AFTER_MULTI = "stack_after_multi"
    #: Issuing WATCH is forbidden for the lifetime of the frame.
    FORBID_WATCH = "forbid_watch"


class ScriptPreload(str, Enum):
    """How queued Lua scripts are ensured before execute()."""

    #: EXISTS-then-LOAD preloading through immediate commands (historical).
    PIN = "pin"
    #: Run SCRIPT EXISTS immediately, SCRIPT LOAD for anything missing.
    EXISTS_THEN_LOAD = "exists_then_load"
    #: Skip script preloading before execute() for this frame.
    SKIP_PRELOAD = "skip_preload"
    #: Buffering SCRIPT EXISTS/SCRIPT LOAD on the command stack is forbidden.
    FORBID_BUFFER_SCRIPT = "forbid_buffer_script"


class WatchFault(str, Enum):
    """How a broken watch is reported."""

    #: Report watch breaks as WatchError (the historical behavior).
    PIN = "pin"
    #: EXEC nil / connection death while watching raises WatchError.
    WATCH_ERROR_ON_BREAK = "watch_error_on_break"
    #: Connection errors keep their original type; never rewrapped as WatchError.
    RERAISE_CONNECTION = "reraise_connection"


class StackEmptyMode(str, Enum):
    """What execute() does with an empty command stack."""

    #: Historical behavior: return [] immediately when not watching.
    PIN = "pin"
    #: Empty stack while not watching short-circuits to an empty result.
    SHORT_CIRCUIT = "short_circuit"
    #: Empty run still performs reset() so a held connection is released.
    REQUIRE_NOOP_RESET = "require_noop_reset"


_DIMENSION_ENUMS = (
    ("exec_mode", ExecMode),
    ("watch_dispatch", WatchDispatch),
    ("script_preload", ScriptPreload),
    ("watch_fault", WatchFault),
    ("stack_empty", StackEmptyMode),
)


def _validate(name: str, value: Any, enum_type: type) -> Optional[Enum]:
    if value is None:
        return None
    if not isinstance(value, enum_type):
        raise TypeError(
            f"{name} must be a {enum_type.__name__} member, got {value!r}"
        )
    return value


class PipelinePolicyFrame:
    """
    One stack entry. Every dimension is optional; unset dimensions are filled
    from the frame below when the merged profile is read.

    A frame is pushed through ``PipelinePolicyStack.push()`` and removed either
    with :meth:`pop` or by using the frame as a (sync or async) context
    manager. Frames always pop cleanly, including when the wrapped block raises.
    """

    __slots__ = (
        "exec_mode",
        "watch_dispatch",
        "script_preload",
        "watch_fault",
        "stack_empty",
        "_stack",
    )

    def __init__(
        self,
        *,
        exec_mode: Optional[ExecMode] = None,
        watch_dispatch: Optional[WatchDispatch] = None,
        script_preload: Optional[ScriptPreload] = None,
        watch_fault: Optional[WatchFault] = None,
        stack_empty: Optional[StackEmptyMode] = None,
    ) -> None:
        self.exec_mode = _validate("exec_mode", exec_mode, ExecMode)
        self.watch_dispatch = _validate(
            "watch_dispatch", watch_dispatch, WatchDispatch
        )
        self.script_preload = _validate(
            "script_preload", script_preload, ScriptPreload
        )
        self.watch_fault = _validate("watch_fault", watch_fault, WatchFault)
        self.stack_empty = _validate(
            "stack_empty", stack_empty, StackEmptyMode
        )
        self._stack: Optional["PipelinePolicyStack"] = None

    def pop(self) -> None:
        """Remove this frame from its stack. Popping twice is a no-op."""
        stack = self._stack
        if stack is None:
            return
        stack._remove_frame(self)
        self._stack = None

    def __enter__(self) -> "PipelinePolicyFrame":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.pop()
        return False

    async def __aenter__(self) -> "PipelinePolicyFrame":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> bool:
        self.pop()
        return False


class EffectivePipelinePolicy:
    """
    Read-only merged view of a stack: each dimension holds the value from the
    topmost frame that set it, or the PIN sentinel when no frame did. Instances
    are cached on the stack and rebuilt only on push/pop.
    """

    __slots__ = (
        "exec_mode",
        "watch_dispatch",
        "script_preload",
        "watch_fault",
        "stack_empty",
    )

    def __init__(
        self,
        *,
        exec_mode: ExecMode,
        watch_dispatch: WatchDispatch,
        script_preload: ScriptPreload,
        watch_fault: WatchFault,
        stack_empty: StackEmptyMode,
    ) -> None:
        self.exec_mode = exec_mode
        self.watch_dispatch = watch_dispatch
        self.script_preload = script_preload
        self.watch_fault = watch_fault
        self.stack_empty = stack_empty

    @classmethod
    def merge(cls, frames: Iterable[PipelinePolicyFrame]) -> "EffectivePipelinePolicy":
        values = {
            "exec_mode": ExecMode.PIN,
            "watch_dispatch": WatchDispatch.PIN,
            "script_preload": ScriptPreload.PIN,
            "watch_fault": WatchFault.PIN,
            "stack_empty": StackEmptyMode.PIN,
        }
        # Bottom to top: a frame only overrides dimensions it set explicitly,
        # and the topmost setter wins.
        for frame in frames:
            for name, _enum_type in _DIMENSION_ENUMS:
                value = getattr(frame, name)
                if value is not None:
                    values[name] = value
        return cls(**values)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, EffectivePipelinePolicy):
            return NotImplemented
        return all(
            getattr(self, name) is getattr(other, name)
            for name, _enum_type in _DIMENSION_ENUMS
        )

    def __repr__(self) -> str:
        return (
            "EffectivePipelinePolicy("
            f"exec_mode={self.exec_mode.value!r}, "
            f"watch_dispatch={self.watch_dispatch.value!r}, "
            f"script_preload={self.script_preload.value!r}, "
            f"watch_fault={self.watch_fault.value!r}, "
            f"stack_empty={self.stack_empty.value!r})"
        )

    # -- exec_mode -------------------------------------------------------

    @property
    def execute_forbidden(self) -> bool:
        return self.exec_mode is ExecMode.FORBID_EXECUTE

    def route_multi_exec(self, transaction: bool, explicit_transaction: bool) -> bool:
        """Resolve which executor ``execute()`` must dispatch to."""
        if self.exec_mode is ExecMode.MULTI_EXEC:
            return True
        if self.exec_mode in (ExecMode.BARE_PIPELINE, ExecMode.FORBID_EXECUTE):
            return False
        return bool(transaction or explicit_transaction)

    def ensure_executor(
        self, transactional: bool, commands: Iterable[Tuple[tuple, dict]] = ()
    ) -> None:
        """
        Guard invoked from ``_execute_transaction`` / ``_execute_pipeline`` so
        the packed wire shape cannot disagree with the active exec_mode, even
        if those methods are reached directly.
        """
        if self.exec_mode is ExecMode.FORBID_EXECUTE:
            raise PipelinePolicyError(
                "execute() is forbidden by the active pipeline policy "
                f"({ExecMode.FORBID_EXECUTE.value})"
            )
        if transactional:
            if self.exec_mode is ExecMode.BARE_PIPELINE:
                raise PipelinePolicyError(
                    "active pipeline policy bare_pipeline forbids packing a "
                    "MULTI/EXEC transactional block"
                )
            return
        if self.exec_mode is ExecMode.MULTI_EXEC:
            raise PipelinePolicyError(
                "active pipeline policy multi_exec requires a MULTI/EXEC "
                "transactional block; bare pipeline execution is forbidden"
            )
        if self.exec_mode is ExecMode.BARE_PIPELINE:
            for args, _options in commands:
                if args and args[0] in ("MULTI", "EXEC"):
                    raise PipelinePolicyError(
                        "active pipeline policy bare_pipeline forbids MULTI/EXEC "
                        "in the packed command stack"
                    )

    # -- watch_dispatch --------------------------------------------------

    @property
    def watch_forbidden(self) -> bool:
        return self.watch_dispatch is WatchDispatch.FORBID_WATCH

    def dispatch_immediately(
        self, watching: bool, command_name: Any, explicit_transaction: bool
    ) -> bool:
        """Resolve execute_command's immediate-vs-buffer fork."""
        if explicit_transaction:
            return False
        if not (watching or command_name == "WATCH"):
            return False
        return self.watch_dispatch is not WatchDispatch.STACK_AFTER_MULTI

    # -- script_preload --------------------------------------------------

    @property
    def preload_scripts(self) -> bool:
        return self.script_preload is not ScriptPreload.SKIP_PRELOAD

    @property
    def script_buffering_forbidden(self) -> bool:
        return self.script_preload is ScriptPreload.FORBID_BUFFER_SCRIPT

    # -- watch_fault -----------------------------------------------------

    @property
    def raise_watch_error_on_break(self) -> bool:
        return self.watch_fault is not WatchFault.RERAISE_CONNECTION

    @property
    def reraise_connection_errors(self) -> bool:
        return self.watch_fault is WatchFault.RERAISE_CONNECTION

    # -- stack_empty -----------------------------------------------------

    @property
    def release_connection_on_empty(self) -> bool:
        return self.stack_empty is StackEmptyMode.REQUIRE_NOOP_RESET


_PIN_PROFILE = EffectivePipelinePolicy(
    exec_mode=ExecMode.PIN,
    watch_dispatch=WatchDispatch.PIN,
    script_preload=ScriptPreload.PIN,
    watch_fault=WatchFault.PIN,
    stack_empty=StackEmptyMode.PIN,
)


class PipelinePolicyStack:
    """
    Instance-local LIFO stack of :class:`PipelinePolicyFrame` objects plus the
    cached merged :class:`EffectivePipelinePolicy`. A Pipeline owns one stack
    for its whole lifetime; ``reset()`` never pops middleware frames.
    """

    __slots__ = ("_frames", "_profile")

    def __init__(self) -> None:
        self._frames: list = []
        self._profile: EffectivePipelinePolicy = _PIN_PROFILE

    def push(
        self,
        *,
        exec_mode: Optional[ExecMode] = None,
        watch_dispatch: Optional[WatchDispatch] = None,
        script_preload: Optional[ScriptPreload] = None,
        watch_fault: Optional[WatchFault] = None,
        stack_empty: Optional[StackEmptyMode] = None,
    ) -> PipelinePolicyFrame:
        frame = PipelinePolicyFrame(
            exec_mode=exec_mode,
            watch_dispatch=watch_dispatch,
            script_preload=script_preload,
            watch_fault=watch_fault,
            stack_empty=stack_empty,
        )
        frame._stack = self
        self._frames.append(frame)
        self._profile = None
        return frame

    def pop(self) -> PipelinePolicyFrame:
        """Pop and return the top frame."""
        if not self._frames:
            raise RuntimeError("no pipeline policy frame to pop")
        frame = self._frames[-1]
        self._remove_frame(frame)
        frame._stack = None
        return frame

    def _remove_frame(self, frame: PipelinePolicyFrame) -> None:
        if not self._frames or self._frames[-1] is not frame:
            raise RuntimeError(
                "pipeline policy frames must be removed in LIFO (push) order"
            )
        self._frames.pop()
        self._profile = None

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def profile(self) -> EffectivePipelinePolicy:
        if self._profile is None:
            self._profile = EffectivePipelinePolicy.merge(self._frames)
        return self._profile
