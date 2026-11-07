"""agent-trace-replay: Deterministic trace capture and replay for tool-calling agents."""

from agent_trace_replay.rng import draw, draw_int, draw_choice, draw_bool
from agent_trace_replay.world import OrderDeskWorld, TransientToolError, EffectEntry
from agent_trace_replay.tools import EffectClass, ToolSpec, ToolRegistry, build_default_registry
from agent_trace_replay.agent import (
    AgentConfig,
    MutationFlags,
    RunOutcome,
    StepSnapshot,
    run_agent_live,
    compute_state_fingerprint,
)
from agent_trace_replay.trace import (
    CaptureLevel,
    outcome_to_full_trace,
    TraceBundle,
    TraceCorruptionError,
    serialize_trace,
    parse_trace,
    project_trace,
)
from agent_trace_replay.replay import (
    ReplayStatus,
    DivergenceCause,
    ReplayResult,
    replay_trace,
    rerun_live,
    resume_from_checkpoint,
)

__all__ = [
    "draw",
    "draw_int",
    "draw_choice",
    "draw_bool",
    "OrderDeskWorld",
    "TransientToolError",
    "EffectEntry",
    "EffectClass",
    "ToolSpec",
    "ToolRegistry",
    "build_default_registry",
    "AgentConfig",
    "MutationFlags",
    "RunOutcome",
    "StepSnapshot",
    "outcome_to_full_trace",
    "run_agent_live",
    "compute_state_fingerprint",
    "CaptureLevel",
    "TraceBundle",
    "TraceCorruptionError",
    "serialize_trace",
    "parse_trace",
    "project_trace",
    "ReplayStatus",
    "DivergenceCause",
    "ReplayResult",
    "replay_trace",
    "rerun_live",
    "resume_from_checkpoint",
]
