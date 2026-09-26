# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find and kill a live NeMo Gym browsecomp / gym eval run (and orphaned server trees).

Mirrors the manual recipe:
  1. locate ``gym eval run`` + related server wrappers / ``browsecomp_you.sh``
  2. SIGTERM process groups
  3. escalate to SIGKILL if anything remains

Usage:
  python benchmarks/browsecomp/helper_scripts/kill_eval.py --dry-run
  python benchmarks/browsecomp/helper_scripts/kill_eval.py
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time

# Match the eval driver, launch script, and gym-spawned server wrappers.
_MATCH = re.compile(
    r"(gym eval run|browsecomp_you\.sh|"
    r"NeMo-Gym/(?:responses_api_|resources_servers)/|"
    r"NEMO_GYM_CONFIG)",
    re.I,
)


def _ps_rows() -> list[tuple[int, int, int, str, str]]:
    """Return (pid, ppid, pgid, stat, command) for the current user."""
    out = subprocess.check_output(
        ["ps", "-u", str(os.getuid()), "-o", "pid=,ppid=,pgid=,stat=,command="],
        text=True,
    )
    rows: list[tuple[int, int, int, str, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid, ppid, pgid, stat, cmd = parts
        rows.append((int(pid), int(ppid), int(pgid), stat, cmd))
    return rows


def _find_targets(rows: list[tuple[int, int, int, str, str]]) -> tuple[set[int], set[int]]:
    """Return (pgids, extra_pids) to signal.

    Kill by process-group so orphaned bash wrappers under the eval die with it.
    Also include matching parent shells whose pgid differs (e.g. browsecomp_you.sh).
    """
    matched = [r for r in rows if _MATCH.search(r[4])]
    # python app.py only when it sits in a matched process group
    matched_pgids = {r[2] for r in matched}
    for r in rows:
        if r[2] in matched_pgids and ("python app.py" in r[4] or r[4].endswith("python app.py")):
            matched.append(r)

    pgids = {r[2] for r in matched}
    # Parent shells of matched procs that weren't themselves matched by pgid
    by_pid = {r[0]: r for r in rows}
    extra_pids: set[int] = set()
    for pid, ppid, pgid, _stat, _cmd in matched:
        extra_pids.add(pid)
        parent = by_pid.get(ppid)
        if parent and parent[2] not in pgids and _MATCH.search(parent[4]):
            extra_pids.add(parent[0])
            pgids.add(parent[2])
    return pgids, extra_pids


def _alive(pids: set[int], pgids: set[int]) -> list[tuple[int, int, int, str, str]]:
    rows = _ps_rows()
    return [r for r in rows if r[0] in pids or r[2] in pgids]


def _signal_targets(pgids: set[int], pids: set[int], sig: int, dry_run: bool) -> None:
    name = signal.Signals(sig).name
    for pgid in sorted(pgids):
        print(f"{'[dry-run] ' if dry_run else ''}{name} process group {-pgid}")
        if not dry_run:
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                pass
            except PermissionError as e:
                print(f"  skip pgid {pgid}: {e}", file=sys.stderr)
    # Direct pid signals cover group leaders already signaled by killpg; harmless if gone.
    for pid in sorted(pids):
        if not dry_run:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError as e:
                print(f"  skip pid {pid}: {e}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print targets only; do not signal")
    parser.add_argument("--wait", type=float, default=3.0, help="Seconds to wait after SIGTERM before SIGKILL")
    parser.add_argument("--no-kill-9", action="store_true", help="Do not escalate to SIGKILL")
    args = parser.parse_args()

    rows = _ps_rows()
    pgids, pids = _find_targets(rows)
    if not pgids and not pids:
        print("No matching gym eval / browsecomp server processes found.")
        return 0

    targets = [r for r in rows if r[0] in pids or r[2] in pgids]
    print(f"Found {len(targets)} process(es) across {len(pgids)} process group(s):")
    for pid, ppid, pgid, stat, cmd in targets:
        short = cmd if len(cmd) <= 120 else cmd[:117] + "..."
        print(f"  pid={pid} ppid={ppid} pgid={pgid} stat={stat}  {short}")

    _signal_targets(pgids, pids, signal.SIGTERM, args.dry_run)
    if args.dry_run:
        return 0

    time.sleep(args.wait)
    remaining = _alive(pids, pgids)
    if not remaining:
        print("All clear after SIGTERM.")
        return 0

    print(f"{len(remaining)} process(es) still alive after SIGTERM.")
    if args.no_kill_9:
        for pid, _ppid, pgid, stat, cmd in remaining:
            short = cmd if len(cmd) <= 120 else cmd[:117] + "..."
            print(f"  leftover pid={pid} pgid={pgid} stat={stat}  {short}")
        return 1

    rem_pgids = {r[2] for r in remaining}
    rem_pids = {r[0] for r in remaining}
    _signal_targets(rem_pgids, rem_pids, signal.SIGKILL, dry_run=False)
    time.sleep(1)
    remaining = _alive(pids | rem_pids, pgids | rem_pgids)
    if remaining:
        print("Still alive after SIGKILL:", file=sys.stderr)
        for pid, _ppid, pgid, stat, cmd in remaining:
            print(f"  pid={pid} pgid={pgid} stat={stat}  {cmd[:120]}", file=sys.stderr)
        return 1

    print("All clear after SIGKILL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
