"""In-memory pub/sub event bus stub for Sprint 1.

Phase 1+ replaces the in-memory bus with Kafka or Redpanda; the schema (Topic +
payload Pydantic model) stays stable. Sprint1_Plan §3.
"""
from __future__ import annotations

import enum
import logging
from collections import defaultdict
from collections.abc import Callable

from pydantic import BaseModel

LOG = logging.getLogger(__name__)


class Topic(str, enum.Enum):
    GRAM_GENERATED = "gram.generated"
    LINE_DETECTED = "line.detected"        # Sprint 2+
    CONTACT_INITIATED = "contact.initiated"  # Phase 2+
    CONTACT_UPDATED = "contact.updated"      # Phase 2+


class EventBus:
    """Minimal synchronous in-memory pub/sub. Single-process for Sprint 1."""

    def __init__(self) -> None:
        self._subscribers: dict[Topic, list[Callable[[BaseModel], None]]] = defaultdict(list)

    def subscribe(self, topic: Topic, handler: Callable[[BaseModel], None]) -> None:
        self._subscribers[topic].append(handler)
        LOG.debug("subscribed handler to %s", topic.value)

    def publish(self, topic: Topic, payload: BaseModel) -> None:
        handlers = self._subscribers.get(topic, [])
        LOG.debug("publishing to %s (%d handlers)", topic.value, len(handlers))
        for handler in handlers:
            try:
                handler(payload)
            except Exception:
                LOG.exception("event-bus handler failed for topic %s", topic.value)

    def reset(self) -> None:
        self._subscribers.clear()


_bus: EventBus | None = None


def get_default_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus