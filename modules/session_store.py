# modules/session_store.py
"""
In-memory state machine for one keg-scanning session.

Replaces 8 scattered boolean flags in main.py:
    processing, data_ready_to_send, auto_confirm_pending,
    auto_pending_frame, _pending_capture, last_captured_qr_set,
    stability_counter, captured_qr_codes

ScanState transitions:
    SCANNING → (target reached + stable) → READY
    READY    → (user/auto presses SEND)  → SENDING
    SENDING  → (API success)             → SENT
    SENDING  → (API failure)             → READY   (allow retry)
    SENT     → (brief delay)             → SCANNING (reset)
    Any      → (manual clear)            → SCANNING

Continuous-flow behaviour (client requirement):
    After a batch is sent, kegs that remain physically under the camera
    are IGNORED.  Only genuinely new kegs are counted toward the next
    pallet target.  When the next target is reached the system auto-sends
    and auto-prints again — no operator interaction needed.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Set, Optional


class ScanState(Enum):
    SCANNING = "scanning"   # live detection, accumulating QR codes
    READY    = "ready"      # target count reached, awaiting Send
    SENDING  = "sending"    # API call in progress
    SENT     = "sent"       # success — brief hold before reset


@dataclass
class SessionStore:
    # ── Batch config (set from UI dropdowns/fields) ──────────────
    target_count:  int = 6
    beer_type:     str = ""          # human-readable name shown in UI
    beer_type_id:  str = ""          # _id returned from cloud beer-types API
    batch_number:  str = ""
    mode:          str = "AUTO"      # "AUTO" or "MANUAL"

    # ── Live detection state (pure in-memory) ────────────────────
    qr_codes:          Set[str] = field(default_factory=set)
    state:             ScanState = ScanState.SCANNING
    stability_counter: int = 0

    # ── Capture snapshot (populated at moment of capture) ────────
    frame_path:   Optional[str]      = None
    image_name:   Optional[str]      = None
    captured_at:  Optional[datetime] = None
    session_id:   Optional[str]      = None   # single source of truth — set once

    # ── Keg history (continuous-flow guards) ──────────────────────
    last_sent_qr_set:  Set[str] = field(default_factory=set)   # last batch only
    sent_qr_history:   Set[str] = field(default_factory=set)   # ALL kegs ever sent (cumulative)

    # ─────────────────────────────────────────────────────────────
    #  QR helpers
    # ─────────────────────────────────────────────────────────────

    def add_qr(self, code: str) -> bool:
        """
        Add a QR code string.  Returns True if it is genuinely new.

        Kegs that were already sent in a previous pallet (present in
        ``sent_qr_history``) are silently IGNORED — they are not added
        to the current set and do not count toward the target.
        """
        code = code.strip() if code else ""
        if not code:
            return False
        # Ignore kegs already sent in a previous pallet
        if code in self.sent_qr_history:
            return False
        if code not in self.qr_codes:
            self.qr_codes.add(code)
            return True
        return False

    def qr_list(self) -> list:
        """Return QR codes as a plain list (for API payload)."""
        return list(self.qr_codes)

    def count(self) -> int:
        return len(self.qr_codes)

    def target_reached(self) -> bool:
        return len(self.qr_codes) == self.target_count

    def over_target(self) -> bool:
        return len(self.qr_codes) > self.target_count

    def under_target(self) -> bool:
        return len(self.qr_codes) < self.target_count

    # ─────────────────────────────────────────────────────────────
    #  Sent-keg awareness helpers
    # ─────────────────────────────────────────────────────────────

    def is_same_as_last_sent(self) -> bool:
        """True when the current QR set is identical to the last successfully sent one."""
        return bool(self.qr_codes) and self.qr_codes == self.last_sent_qr_set

    def ignored_count(self) -> int:
        """Number of kegs currently being ignored (from previous pallets)."""
        return len(self.sent_qr_history)

    def is_keg_ignored(self, code: str) -> bool:
        """Check if a specific keg QR code is in the ignore list."""
        return code.strip() in self.sent_qr_history if code else False

    def get_ignored_info(self) -> tuple:
        """
        Returns status information about ignored kegs for HMI display.

        Returns
        -------
        (has_ignored: bool, ignored_count: int, message: str)
        """
        ignored = len(self.sent_qr_history)
        if ignored == 0:
            return False, 0, ""
        new_count = len(self.qr_codes)
        return True, ignored, (
            f"{ignored} keg(s) from previous pallet(s) ignored — "
            f"detecting new kegs only ({new_count} new so far)"
        )

    # ─────────────────────────────────────────────────────────────
    #  State transitions
    # ─────────────────────────────────────────────────────────────

    def reset(self):
        """
        Reset to SCANNING state for the next pallet.
        Clears QR accumulator and all capture data.
        Preserves: target_count, beer_type/id, batch_number, mode,
                   last_sent_qr_set, **sent_qr_history** (critical for ignore).
        """
        self.qr_codes          = set()
        self.state             = ScanState.SCANNING
        self.stability_counter = 0
        self.frame_path        = None
        self.image_name        = None
        self.captured_at       = None
        self.session_id        = None

    def mark_sent(self):
        """
        Record the QR set that was just sent.
        Adds all kegs to the cumulative history so they are ignored
        when the camera continues to see them after the pallet is done.
        Transitions to SENT state.
        """
        self.last_sent_qr_set = set(self.qr_codes)
        self.sent_qr_history |= self.qr_codes       # cumulative — kegs are ignored going forward
        self.state = ScanState.SENT

    def clear_history(self):
        """
        Operator manually clears the sent-keg ignore list.
        Use when all kegs have been physically removed from the camera
        area and a fresh pallet load is starting.
        """
        self.sent_qr_history.clear()
        self.last_sent_qr_set.clear()
