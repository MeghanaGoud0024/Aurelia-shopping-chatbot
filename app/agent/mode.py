"""Runtime toggle between the live model and the deterministic planner.

Separate from `app.config.settings` on purpose: settings are read once at
process start from the environment, and nothing in this codebase mutates them
afterward. This toggle is different in kind - an operator flipping it from the
running interface, mid-session, for reasons the environment can't predict
(a quota wall, a cost-control call, wanting to demo the offline path without
restarting the process). Keeping it out of `settings` keeps that distinction
visible in the code rather than papered over by a mutable global masquerading
as configuration.

Process-wide and in-memory, like the checkout quote store and the rate
limiter: correct for the single-process deployment this is, and the same
migration note applies if that ever changes (see docs/SCALING.md).
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class AssistantMode:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._forced_fallback = False
        self._changed_at: str | None = None

    @property
    def forced_fallback(self) -> bool:
        return self._forced_fallback

    def set_forced_fallback(self, value: bool) -> None:
        with self._lock:
            changed = value != self._forced_fallback
            self._forced_fallback = value
            if changed:
                self._changed_at = datetime.now(timezone.utc).isoformat()
        if changed:
            logger.info("assistant_mode.changed", extra={"forced_fallback": value})

    def snapshot(self) -> dict:
        with self._lock:
            return {"forced_fallback": self._forced_fallback, "changed_at": self._changed_at}


#: Process-wide, mirroring quota_tracker and llm_client.
assistant_mode = AssistantMode()
