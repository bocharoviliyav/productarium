"""Expert stream event types.

Defines ``ExpertStreamEvent`` — a typed envelope used across the expert
streaming pipeline so the SSE router can emit typed frames and the frontend
can distinguish phases (status / reasoning / content / error).

Pipeline flow:
    _ExpertLLM.stream()  -> yields ExpertStreamEvent
    _stream_answer()     -> yields ExpertStreamEvent  (RLM wraps chunks as content)
    _run_expert_chat_stream() -> yields ExpertStreamEvent  (adds status events)
    expert router       -> maps ExpertStreamEvent to SSE JSON frames
"""

from __future__ import annotations

from dataclasses import dataclass

#: The set of valid event types.
EVENT_STATUS = "status"
EVENT_REASONING = "reasoning"
EVENT_CONTENT = "content"
EVENT_ERROR = "error"

#: Status phase values (the ``content`` of an ``EVENT_STATUS`` event).
EVENT_RETRIEVING = "retrieving"
EVENT_THINKING = "thinking"
EVENT_ANSWERING = "answering"

ValidEventType = str  # one of the EVENT_* constants above


@dataclass(frozen=True)
class ExpertStreamEvent:
    """A single typed event in the expert streaming pipeline.

    Attributes:
        type: One of ``"status"``, ``"reasoning"``, ``"content"``, ``"error"``.
            - ``status``: phase indicator (``"retrieving"`` / ``"thinking"`` /
              ``"answering"``) so the frontend can show a phase-aware loader.
            - ``reasoning``: a chunk of the model's reasoning / thinking trace
              (from ``reasoning_content`` / ``reasoning`` / ``thinking`` fields
              or inline ``<think>`` tags).
            - ``content``: a chunk of the final answer text.
            - ``error``: an error message.
        content: The text payload for this event.
    """

    type: ValidEventType
    content: str

    def __iter__(self):
        """Allow tuple-style unpacking ``type, content = event`` for ergonomics."""
        yield self.type
        yield self.content


__all__ = [
    "ExpertStreamEvent",
    "EVENT_STATUS",
    "EVENT_REASONING",
    "EVENT_CONTENT",
    "EVENT_ERROR",
    "EVENT_RETRIEVING",
    "EVENT_THINKING",
    "EVENT_ANSWERING",
]
