"""
Individuals — the identity layer sitting on top of raw tracker output.

A tracker produces *fragments*: a shark that turns, dives or crosses a glare
patch comes back with a fresh ID. An :class:`Individual` is the human-made
grouping of those fragments into one animal, which is what the downstream
analysis actually needs.

The GUI does two things and no more: it groups fragments, and it resolves the
frame collisions that grouping creates. Every numeric step — outlier removal,
interpolation, smoothing — stays in ``scripts/track_postprocess.py`` and is
meant to become an opt-in pass over the files written here.

Note on that script: it has **no command line**. Its ``groups`` variable is a
hardcoded list of lists in the ``__main__`` block, and its output names come
from the loop index (``shark_0``, ``shark_1``, …). So:

* :meth:`IndividualStore.to_groups` reproduces that literal shape, which makes
  it pasteable into the script (or loadable, once the script grows a CLI);
* :meth:`IndividualStore.export_individuals` is the path the GUI actually
  uses — it writes one merged JSON per individual, in the schema
  ``export_merged()`` produces, so the downstream readers do not care which of
  the two halves wrote the file.

Two invariants, both taken from that script:

* **One track belongs to at most one individual.** ``track_postprocess.py``
  asserts that no track ID appears twice across groups, so :meth:`assign`
  moves a track rather than duplicating it.
* **Frame collisions are tolerated, not blocked.** When two tracks of the same
  individual cover the same frame, the highest-confidence detection wins —
  :func:`merge_detections` applies exactly the rule ``merge_tracks`` uses.
  :meth:`frame_conflicts` only reports them so the GUI can flag a suspicious
  grouping without refusing it.

No Qt in this module: it is plain data + I/O, so it can be unit-tested alone.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Distinct, high-contrast BGR colours handed out in order. Beyond this the
# store generates new hues, so two individuals never share a colour.
_BASE_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (60, 200, 255),    # amber
    (80, 220, 100),    # green
    (255, 140, 60),    # blue
    (200, 100, 255),   # pink
    (255, 220, 90),    # cyan
    (90, 120, 255),    # red
    (230, 180, 120),   # steel
    (120, 255, 220),   # lime
    (255, 100, 180),   # violet
    (100, 200, 180),   # olive
)

DEFAULT_PREFIX = "shark"


@dataclass
class Individual:
    """One animal, made of one or more tracker fragments."""

    uid: int
    name: str
    track_ids: List[int] = field(default_factory=list)
    color: Tuple[int, int, int] = (200, 200, 200)
    notes: str = ""

    def to_json(self) -> dict:
        return {
            "uid": int(self.uid),
            "name": self.name,
            "track_ids": [int(t) for t in self.track_ids],
            "color": [int(c) for c in self.color],
            "notes": self.notes,
        }

    @staticmethod
    def from_json(data: dict) -> "Individual":
        return Individual(
            uid=int(data["uid"]),
            name=str(data.get("name", "")),
            track_ids=[int(t) for t in data.get("track_ids", [])],
            color=tuple(int(c) for c in data.get("color", (200, 200, 200))),
            notes=str(data.get("notes", "")),
        )


def merge_detections(
    track_ids: Sequence[int],
    detections_by_track: Dict[int, Sequence[dict]],
) -> List[dict]:
    """Concatenate several tracks into one frame-indexed detection list.

    Mirrors ``merge_tracks`` in ``scripts/track_postprocess.py``: tracks are
    consumed in the order given, and when two of them cover the same frame the
    higher-confidence detection wins. Nothing is interpolated or smoothed —
    that is the post-processing script's job, deliberately left out here.

    Each record is copied and tagged with ``source_track_id`` so a merge stays
    auditable, and with ``interpolated: False`` to match the schema
    ``export_merged()`` writes.

    Returns the surviving detections sorted by frame.
    """
    best: Dict[int, dict] = {}
    for tid in track_ids:
        for det in detections_by_track.get(int(tid), ()):
            frame = int(det["frame"])
            conf = float(det.get("confidence", 0.0))
            current = best.get(frame)
            if current is not None and float(current["confidence"]) >= conf:
                continue
            record = dict(det)
            record["source_track_id"] = int(tid)
            record.setdefault("interpolated", False)
            best[frame] = record
    return [best[f] for f in sorted(best)]


class IndividualStore:
    """Holds the individuals of one clip and the track -> individual mapping."""

    def __init__(self, prefix: str = DEFAULT_PREFIX):
        self.prefix = prefix
        self._individuals: Dict[int, Individual] = {}
        self._owner: Dict[int, int] = {}     # track_id -> individual uid
        self._next_uid = 0

    # ---------------- Access ----------------

    def __len__(self) -> int:
        return len(self._individuals)

    def all(self) -> List[Individual]:
        """Individuals in creation order, which is also display order."""
        return [self._individuals[u] for u in sorted(self._individuals)]

    def get(self, uid: int) -> Optional[Individual]:
        return self._individuals.get(uid)

    def individual_of(self, track_id: int) -> Optional[Individual]:
        uid = self._owner.get(int(track_id))
        return self._individuals.get(uid) if uid is not None else None

    def color_for_track(self, track_id: int) -> Optional[Tuple[int, int, int]]:
        """Colour of the owning individual, or None when unassigned."""
        ind = self.individual_of(track_id)
        return ind.color if ind else None

    def unassigned(self, track_ids: Iterable[int]) -> List[int]:
        return sorted({int(t) for t in track_ids if int(t) not in self._owner})

    # ---------------- Colours ----------------

    def _next_color(self) -> Tuple[int, int, int]:
        """Return a colour no existing individual is using.

        Walks the fixed palette first, then generates evenly spaced hues so a
        long session never has to reuse one.
        """
        used = {tuple(i.color) for i in self._individuals.values()}
        for color in _BASE_PALETTE:
            if color not in used:
                return color

        # Palette exhausted: derive further hues by golden-angle rotation,
        # which keeps successive colours far apart.
        import colorsys
        step = 0
        while True:
            hue = (len(_BASE_PALETTE) + step) * 0.618033988749895 % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
            color = (int(b * 255), int(g * 255), int(r * 255))   # BGR
            if color not in used:
                return color
            step += 1

    # ---------------- Mutation ----------------

    def create(self, track_ids: Optional[Sequence[int]] = None,
               name: Optional[str] = None) -> Individual:
        """Create an individual, optionally seeded with tracks."""
        uid = self._next_uid
        self._next_uid += 1
        ind = Individual(
            uid=uid,
            name=name or f"{self.prefix}_{uid}",
            color=self._next_color(),
        )
        self._individuals[uid] = ind
        for tid in (track_ids or []):
            self.assign(int(tid), uid)
        return ind

    def assign(self, track_id: int, uid: int) -> bool:
        """Attach a track to an individual, detaching it from any other.

        Returns False when the target individual does not exist.
        """
        track_id, uid = int(track_id), int(uid)
        if uid not in self._individuals:
            return False

        previous = self._owner.get(track_id)
        if previous == uid:
            return True
        if previous is not None:
            # One track, one individual: the downstream script rejects a track
            # appearing in two groups.
            self._individuals[previous].track_ids.remove(track_id)

        self._individuals[uid].track_ids.append(track_id)
        self._individuals[uid].track_ids.sort()
        self._owner[track_id] = uid
        return True

    def unassign(self, track_id: int) -> bool:
        """Detach a track ("none"). Returns False when it was already free."""
        track_id = int(track_id)
        uid = self._owner.pop(track_id, None)
        if uid is None:
            return False
        ind = self._individuals.get(uid)
        if ind and track_id in ind.track_ids:
            ind.track_ids.remove(track_id)
        return True

    def rename(self, uid: int, name: str) -> bool:
        ind = self._individuals.get(int(uid))
        if ind is None or not name.strip():
            return False
        ind.name = name.strip().replace(" ", "_")
        return True

    def set_notes(self, uid: int, notes: str) -> bool:
        ind = self._individuals.get(int(uid))
        if ind is None:
            return False
        ind.notes = notes
        return True

    def delete(self, uid: int) -> bool:
        """Remove an individual; its tracks become unassigned again."""
        uid = int(uid)
        ind = self._individuals.pop(uid, None)
        if ind is None:
            return False
        for tid in ind.track_ids:
            self._owner.pop(tid, None)
        return True

    def clear(self):
        self._individuals.clear()
        self._owner.clear()
        self._next_uid = 0

    def drop_missing_tracks(self, valid_track_ids: Iterable[int]) -> int:
        """Detach tracks that no longer exist. Returns how many were dropped.

        Used after the tracker is reset: the raw IDs are reallocated, so any
        mapping pointing at a vanished ID is stale.
        """
        valid = {int(t) for t in valid_track_ids}
        stale = [t for t in self._owner if t not in valid]
        for tid in stale:
            self.unassign(tid)
        return len(stale)

    def stale_track_ids(self, valid_track_ids: Iterable[int]) -> List[int]:
        """Assigned tracks absent from ``valid_track_ids`` (no mutation)."""
        valid = {int(t) for t in valid_track_ids}
        return sorted(t for t in self._owner if t not in valid)

    # ---------------- Validation ----------------

    def frame_conflicts(
        self, frames_by_track: Dict[int, Set[int]],
    ) -> Dict[int, int]:
        """Count frames covered by more than one track, per individual.

        A single animal cannot be in two places at once, so a non-zero count
        usually means a wrong grouping. It is reported rather than blocked:
        ``track_postprocess.py`` resolves collisions by keeping the
        highest-confidence detection, and forbidding them would make
        annotation painful.

        Returns ``{individual_uid: n_conflicting_frames}``, omitting the
        individuals that are clean.
        """
        conflicts: Dict[int, int] = {}
        for ind in self._individuals.values():
            seen: Set[int] = set()
            clashing: Set[int] = set()
            for tid in ind.track_ids:
                for frame in frames_by_track.get(tid, ()):
                    if frame in seen:
                        clashing.add(frame)
                    else:
                        seen.add(frame)
            if clashing:
                conflicts[ind.uid] = len(clashing)
        return conflicts

    # ---------------- Export ----------------

    def to_groups(self) -> List[List[int]]:
        """Merge groups for ``track_postprocess.py --groups``.

        Empty individuals are skipped: the script treats a group as one
        output file and an empty one would produce a dangling track.
        """
        return [list(ind.track_ids) for ind in self.all() if ind.track_ids]

    def names(self) -> List[str]:
        """Output names, aligned with :meth:`to_groups`."""
        return [ind.name for ind in self.all() if ind.track_ids]

    def merged_records(
        self, uid: int, detections_by_track: Dict[int, Sequence[dict]],
    ) -> List[dict]:
        """Detections of one individual, collisions already resolved."""
        ind = self._individuals.get(int(uid))
        if ind is None:
            return []
        return merge_detections(ind.track_ids, detections_by_track)

    def export_individuals(
        self, directory: str, detections_by_track: Dict[int, Sequence[dict]],
    ) -> List[dict]:
        """Write ``<directory>/<name>.json``, one file per non-empty individual.

        The payload is a superset of what ``export_merged()`` produces, so the
        post-processing script's own readers accept it unchanged; the extra
        keys carry the identity the script has no notion of (uid, notes,
        colour).

        Returns one summary dict per file written, with ``dropped`` counting
        the detections a frame collision discarded.
        """
        os.makedirs(os.path.abspath(directory), exist_ok=True)
        written: List[dict] = []
        used: Set[str] = set()

        for ind in self.all():
            if not ind.track_ids:
                continue

            records = merge_detections(ind.track_ids, detections_by_track)
            if not records:
                # Grouped, but the tracks carry no detection yet: writing an
                # empty file would look like a real but silent individual.
                continue

            stem = _safe_stem(ind.name) or f"individual_{ind.uid}"
            if stem in used:
                stem = f"{stem}_{ind.uid}"
            used.add(stem)

            raw = sum(len(detections_by_track.get(t, ())) for t in ind.track_ids)
            payload = {
                "uid": int(ind.uid),
                "name": ind.name,
                "notes": ind.notes,
                "color": [int(c) for c in ind.color],
                "merged_track_ids": [int(t) for t in ind.track_ids],
                "num_detections": len(records),
                "first_frame": records[0]["frame"],
                "last_frame": records[-1]["frame"],
                "detections": records,
            }
            path = os.path.join(directory, f"{stem}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            written.append({
                "uid": ind.uid,
                "name": ind.name,
                "path": path,
                "detections": len(records),
                "dropped": raw - len(records),
                "tracks": len(ind.track_ids),
            })
        return written

    # ---------------- Persistence ----------------

    def to_json(self) -> dict:
        return {
            "version": 1,
            "prefix": self.prefix,
            "individuals": [i.to_json() for i in self.all()],
        }

    def load_json(self, data: dict):
        self.clear()
        self.prefix = str(data.get("prefix", self.prefix))
        for entry in data.get("individuals", []):
            ind = Individual.from_json(entry)
            self._individuals[ind.uid] = ind
            # Rebuild the reverse map, dropping any duplicate assignment a
            # hand-edited file might contain.
            kept: List[int] = []
            for tid in ind.track_ids:
                if tid in self._owner:
                    continue
                self._owner[tid] = ind.uid
                kept.append(tid)
            ind.track_ids = kept
            self._next_uid = max(self._next_uid, ind.uid + 1)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> bool:
        """Load from disk. Returns False when the file is absent or unreadable."""
        if not os.path.isfile(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.load_json(json.load(f))
            return True
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return False

    def export_groups_json(self, path: str):
        """Write the bare ``[[2, 34], [6]]`` file the CLI script consumes."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_groups(), f, indent=2)


def _safe_stem(name: str) -> str:
    """Reduce a free-text individual name to a safe file stem.

    Names are user input and end up as filenames, so anything that could climb
    out of the target directory or upset a shell is replaced.
    """
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in name.strip()]
    return "".join(keep).strip("._") or ""


def individuals_path(export_dir: str) -> str:
    """Canonical location of the mapping, next to the clip's exports.

    Track IDs are only meaningful within one clip, so the file is per-clip.
    """
    return os.path.join(export_dir, "individuals.json")


def individuals_dir(export_dir: str) -> str:
    """Directory holding one merged JSON per individual."""
    return os.path.join(export_dir, "individuals")