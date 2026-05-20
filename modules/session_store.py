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
    SENT     → (2 s delay)              → SCANNING (reset)
    Any      → (manual clear)           → SCANNING
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

    # ── Last-sent guard (prevents re-capture of same pallet) ─────
    last_sent_qr_set: Set[str] = field(default_factory=set)

    # ─────────────────────────────────────────────────────────────
    #  QR helpers
    # ─────────────────────────────────────────────────────────────

    def add_qr(self, code: str) -> bool:
        """Add a QR code string. Returns True if it is genuinely new."""
        code = code.strip() if code else ""
        if code and code not in self.qr_codes:
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

    def is_same_as_last_sent(self) -> bool:
        """True when the current QR set is identical to the last successfully sent one."""
        return bool(self.qr_codes) and self.qr_codes == self.last_sent_qr_set

    def overlap_with_last_sent(self) -> Set[str]:
        """Return the set of QR codes that appear in both the current scan and the last sent batch."""
        if not self.qr_codes or not self.last_sent_qr_set:
            return set()
        return self.qr_codes & self.last_sent_qr_set

    def overlap_ratio_with_last_sent(self) -> float:
        """
        Fraction of *current* QR codes that were also in the last sent batch.
        Returns 0.0 when there is nothing to compare.
        """
        if not self.qr_codes or not self.last_sent_qr_set:
            return 0.0
        overlap = self.qr_codes & self.last_sent_qr_set
        return len(overlap) / len(self.qr_codes)

    def has_duplicate_kegs(self) -> tuple:
        """
        Main duplicate-keg check used by the HMI.

        Returns
        -------
        (is_blocked: bool, is_warning: bool, message: str)
            is_blocked  – True when the current set is an *exact* match
                          of the last sent batch → capture/send MUST be refused.
            is_warning  – True when ≥ 50 % of current kegs overlap with the
                          last sent batch → operator should be alerted but
                          send is still allowed.
            message     – Human-readable description shown in the UI.
        """
        if not self.last_sent_qr_set or not self.qr_codes:
            return False, False, ""

        overlap = self.overlap_with_last_sent()
        if not overlap:
            return False, False, ""

        ratio = len(overlap) / len(self.qr_codes)

        # Exact duplicate → block
        if self.qr_codes == self.last_sent_qr_set:
            return True, False, (
                f"SAME KEGS DETECTED — All {len(overlap)} kegs match the "
                f"previous batch. Remove and place new kegs."
            )

        # High overlap (≥ 50 %) → warning
        if ratio >= 0.5:
            return False, True, (
                f"{len(overlap)} of {len(self.qr_codes)} kegs match the "
                f"previous batch. Replace duplicate kegs before sending."
            )

        return False, False, ""

    # ─────────────────────────────────────────────────────────────
    #  State transitions
    # ─────────────────────────────────────────────────────────────

    def reset(self):
        """
        Reset to SCANNING state.
        Clears QR accumulator and all capture data.
        Preserves: target_count, beer_type/id, batch_number, mode, last_sent_qr_set.
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
        Record the QR set that was just sent so we don't re-capture it.
        Transitions to SENT state.
        """
        self.last_sent_qr_set = set(self.qr_codes)
        self.state = ScanState.SENT
