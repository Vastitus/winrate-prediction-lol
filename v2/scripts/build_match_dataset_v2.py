"""V2: High-Elo Match-Dataset (Queue 420) als CSV — fair per Region, Checkpoints/Resume, Cooldown bei leeren Regionen."""

from __future__ import annotations

import argparse
import atexit
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "data_collection"))

from riot_api import REGION_CONFIGS, RiotAPI  # noqa: E402

POSITION_MAP = {
    "TOP": "top",
    "JUNGLE": "jungle",
    "MIDDLE": "mid",
    "BOTTOM": "adc",
    "UTILITY": "support",
}

_STATE: Dict[str, object] = {}


def parse_args():
    p = argparse.ArgumentParser(description="V2 Dataset: High-Elo, faire Regionen, Match+Timeline.")
    p.add_argument(
        "--api-key",
        default="",
        help="Riot API key. Leer = RIOT_API_KEY aus .env / config (wie riot_api.RiotAPI).",
    )
    p.add_argument("--target-matches", type=int, default=10000, help="Anzahl fertige CSV-Zeilen.")
    p.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "datasets" / "v2_match_dataset_10000.csv"),
        help="Ausgabe-CSV (wird bei Checkpoint ueberschrieben).",
    )
    p.add_argument("--regions", default="", help="Komma-getrennte Regionen. Leer = alle REGION_CONFIGS.")
    p.add_argument("--max-players-per-region", type=int, default=2000, help="Max. Spieler pro Region (C/GM/M).")
    p.add_argument("--parallel", action="store_true", help="Parallele Routing-Threads (weniger fair bei kleinem n).")
    p.add_argument("--sleep-ms", type=int, default=0, help="Pause nach erfolgreicher Zeile.")
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="CSV alle N Zeilen speichern (0 = nur am Ende). Zusaetzlich immer nach der 1. Zeile.",
    )
    p.add_argument("--resume", action="store_true", help="Bestehende CSV laden (match_id -> seen) und weitermachen.")
    p.add_argument(
        "--max-worker-iterations",
        type=int,
        default=200_000,
        help="Max. Runden im Global/Routing-Worker (Schutz vor Endlosschleife).",
    )
    p.add_argument(
        "--dry-rounds-before-cooldown",
        type=int,
        default=2,
        help="Wie oft hintereinander eine Region keine neue Zeile liefert, bevor Cooldown.",
    )
    p.add_argument(
        "--cooldown-rounds",
        type=int,
        default=80,
        help="Region wird fuer so viele Worker-Runden uebersprungen (dann erneuter Versuch).",
    )
    p.add_argument(
        "--no-progress-timeout-seconds",
        type=int,
        default=900,
        help="Abbruch wenn so lange keine neue Zeile geschrieben wurde (0 = aus).",
    )
    return p.parse_args()


def _routing_from_match_id(match_id: str) -> str:
    prefix = match_id.split("_", 1)[0].lower()
    if prefix not in REGION_CONFIGS:
        raise ValueError(f"Unknown match prefix: {prefix}")
    return REGION_CONFIGS[prefix]["v5_region"]


def _fetch_timeline(match_id: str, api_key: str, max_retries: int = 5) -> Dict:
    routing = _routing_from_match_id(match_id)
    url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    headers = {"X-Riot-Token": api_key}

    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 429:
            time.sleep(int(resp.headers.get("Retry-After", "2")))
            continue
        if resp.status_code >= 500 and attempt < max_retries - 1:
            time.sleep((attempt + 1) * 2)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Timeline fetch failed for {match_id}")


def _extract_match_fields(match_data: Dict) -> Optional[Tuple[Dict[str, int], int, Dict[int, int]]]:
    participants = match_data.get("info", {}).get("participants", [])
    teams = match_data.get("info", {}).get("teams", [])
    if len(participants) < 10 or len(teams) < 2:
        return None

    team_pos_to_champ: Dict[str, int] = {}
    pid_to_team: Dict[int, int] = {}
    for p in participants:
        team_id = int(p.get("teamId", 0))
        pid = int(p.get("participantId", 0))
        pos_raw = str(p.get("teamPosition", "")).upper()
        champ = int(p.get("championId", -1))
        if pid > 0 and team_id in (100, 200):
            pid_to_team[pid] = team_id
        if pos_raw in POSITION_MAP and team_id in (100, 200):
            key = f"team{1 if team_id == 100 else 2}_{POSITION_MAP[pos_raw]}"
            team_pos_to_champ[key] = champ

    required = [
        "team1_top",
        "team1_jungle",
        "team1_mid",
        "team1_adc",
        "team1_support",
        "team2_top",
        "team2_jungle",
        "team2_mid",
        "team2_adc",
        "team2_support",
    ]
    if any(c not in team_pos_to_champ for c in required):
        return None

    target = 0
    for t in teams:
        if int(t.get("teamId", 0)) == 100:
            target = 1 if bool(t.get("win", False)) else 0
            break
    return team_pos_to_champ, target, pid_to_team


def _gold_diff_at_minute(frames: List[Dict], pid_to_team: Dict[int, int], minute: int) -> Optional[int]:
    target_ms = minute * 60 * 1000
    frame = None
    for f in frames:
        if int(f.get("timestamp", -1)) <= target_ms:
            frame = f
        else:
            break
    if frame is None:
        return None

    gold_100 = gold_200 = 0
    for frame_pid, pf in frame.get("participantFrames", {}).items():
        pid = int(pf.get("participantId", frame_pid))
        team = pid_to_team.get(pid, 0)
        g = int(pf.get("totalGold", 0))
        if team == 100:
            gold_100 += g
        elif team == 200:
            gold_200 += g
    return gold_100 - gold_200


def _diffs_until_minute(frames: List[Dict], pid_to_team: Dict[int, int], minute: int) -> Dict[str, int]:
    target_ms = minute * 60 * 1000
    kills = {100: 0, 200: 0}
    towers = {100: 0, 200: 0}
    dragons = {100: 0, 200: 0}

    for frame in frames:
        for ev in frame.get("events", []):
            if int(ev.get("timestamp", 0)) > target_ms:
                continue
            et = ev.get("type")
            kid = int(ev.get("killerId", 0) or 0)
            kteam = int(ev.get("killerTeamId", 0) or 0)
            team = pid_to_team.get(kid, kteam)
            if team not in (100, 200):
                continue
            if et == "CHAMPION_KILL":
                kills[team] += 1
            elif et == "BUILDING_KILL" and ev.get("buildingType") == "TOWER_BUILDING":
                towers[team] += 1
            elif et == "ELITE_MONSTER_KILL" and ev.get("monsterType") == "DRAGON":
                dragons[team] += 1

    return {
        "kill_diff": kills[100] - kills[200],
        "tower_diff": towers[100] - towers[200],
        "dragon_diff": dragons[100] - dragons[200],
    }


def _build_row(match_id: str, match_data: Dict, timeline: Dict) -> Optional[Dict]:
    extracted = _extract_match_fields(match_data)
    if extracted is None:
        return None
    champs, target, pid_to_team = extracted
    frames = timeline.get("info", {}).get("frames", [])
    if not frames:
        return None

    row: Dict = {"match_id": match_id, "target": target}
    row.update(champs)
    for minute, suffix in ((7, "m7"), (15, "m15")):
        gd = _gold_diff_at_minute(frames, pid_to_team, minute)
        if gd is None:
            return None
        d = _diffs_until_minute(frames, pid_to_team, minute)
        row[f"gold_diff_{suffix}"] = gd
        row[f"kill_diff_{suffix}"] = d["kill_diff"]
        row[f"tower_diff_{suffix}"] = d["tower_diff"]
        row[f"dragon_diff_{suffix}"] = d["dragon_diff"]
    return row


def get_row_counts_per_region(rows: List[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        mid = str(row.get("match_id", ""))
        parts = mid.split("_")
        if len(parts) >= 2:
            r = parts[0].lower()
            out[r] = out.get(r, 0) + 1
    return out


def _rows_done(rows: List[Dict], lock: threading.RLock, target: int) -> bool:
    with lock:
        return len(rows) >= target


def write_checkpoint(
    output_path: Path,
    rows: List[Dict],
    lock: threading.RLock,
    checkpoint_ref: List[int],
    force_full: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        total = len(rows)
        last_checkpoint = checkpoint_ref[0]
        if force_full or last_checkpoint > total:
            snapshot = list(rows)
            checkpoint_ref[0] = total
            mode = "w"
            header = True
            written_now = total
        else:
            if total == last_checkpoint:
                return
            snapshot = list(rows[last_checkpoint:])
            checkpoint_ref[0] = total
            written_now = total - last_checkpoint
            if output_path.exists() and last_checkpoint > 0:
                mode = "a"
                header = False
            else:
                mode = "w"
                header = True

    pd.DataFrame(snapshot).to_csv(output_path, mode=mode, header=header, index=False)
    print(f"[CHECKPOINT] +{written_now} (gesamt={checkpoint_ref[0]}) -> {output_path}", flush=True)


def _flush_state() -> None:
    st = _STATE
    if not st:
        return
    output_path: Path = st["output_path"]  # type: ignore[assignment]
    rows: List[Dict] = st["rows"]  # type: ignore[assignment]
    lock: threading.RLock = st["lock"]  # type: ignore[assignment]
    checkpoint_ref: List[int] = st["checkpoint_ref"]  # type: ignore[assignment]
    if rows:
        try:
            write_checkpoint(output_path, rows, lock, checkpoint_ref)
        except Exception as exc:
            print(f"[WARN] Flush fehlgeschlagen: {exc}")


def collect_from_region(
    api: RiotAPI,
    api_key: str,
    target_matches: int,
    seen_match_ids: Set[str],
    rows: List[Dict],
    claims_ref: List[int],
    sleep_seconds: float,
    output_path: Optional[Path],
    checkpoint_every: int,
    lock: threading.RLock,
    checkpoint_ref: List[int],
    max_players_per_region: int = 2000,
    max_matches_per_region: Optional[int] = None,
) -> int:
    region_name = getattr(api, "region", "unknown")
    players: List[dict] = []
    warn_sample = 0
    try:
        players.extend(api.get_challenger_players())
        players.extend(api.get_grandmaster_players())
        players.extend(api.get_master_players())
    except Exception as exc:
        print(f"[WARN] Spieler-Liste fehlgeschlagen ({region_name}): {exc}", flush=True)
        return claims_ref[0]
    if not players:
        print(f"[WARN] Keine High-Elo Spieler in {region_name} gefunden.", flush=True)
        return claims_ref[0]

    puuids: List[str] = []
    for pl in players[:max_players_per_region]:
        with lock:
            if _rows_done(rows, lock, target_matches):
                break
        if pl.get("puuid"):
            puuids.append(pl["puuid"])

    if not puuids:
        print(f"[WARN] Keine PUUIDs in {region_name}.", flush=True)
        return claims_ref[0]

    print(f"[REGION] {region_name}: {len(puuids)} PUUIDs", flush=True)
    region_hits = 0
    for puuid in puuids:
        with lock:
            if _rows_done(rows, lock, target_matches):
                break
            if max_matches_per_region and region_hits >= max_matches_per_region:
                break

        try:
            with lock:
                ct = len(rows)

            if ct > 30000:
                mc = 10
            elif ct > 10000:
                mc = 15
            elif ct > 5000:
                mc = 30
            else:
                mc = 100

            mids = api.get_match_ids(puuid, count=mc, queue=420)
            dup = new = chk = 0

            for mid in mids:
                chk += 1
                is_dup = False
                with lock:
                    if _rows_done(rows, lock, target_matches):
                        break
                    if mid in seen_match_ids:
                        is_dup = True
                        dup += 1
                    else:
                        if max_matches_per_region and region_hits >= max_matches_per_region:
                            break
                        seen_match_ids.add(mid)
                        new += 1

                if chk >= 5 and dup == chk and new == 0:
                    break
                if chk >= 10 and dup / chk >= 0.8 and new == 0:
                    break
                if is_dup:
                    continue

                try:
                    md = api.get_match_details(mid)
                    tl = _fetch_timeline(mid, api_key)
                    rw = _build_row(mid, md, tl)
                    if rw is None:
                        continue
                    rw["region"] = str(mid).split("_", 1)[0].lower()
                    with lock:
                        rows.append(rw)
                        n = len(rows)
                        claims_ref[0] = n
                    region_hits += 1
                    if n % 100 == 0:
                        print(f"[PROGRESS] {n}/{target_matches} Zeilen", flush=True)
                    if output_path and checkpoint_every and (n == 1 or n % checkpoint_every == 0):
                        write_checkpoint(output_path, rows, lock, checkpoint_ref)
                    time.sleep(sleep_seconds)
                except Exception as exc:
                    if warn_sample < 5:
                        print(f"[WARN] Match-Verarbeitung fehlgeschlagen ({mid}): {exc}", flush=True)
                        warn_sample += 1
                    continue
        except Exception:
            print(f"[WARN] Match-ID Fetch fehlgeschlagen in {region_name} (puuid={puuid[:8]}...)", flush=True)
            continue
    if region_hits == 0:
        print(f"[CYCLE-SKIP] {region_name}: keine neue Zeile.", flush=True)
    return claims_ref[0]


def collect_global_worker(
    api_key: str,
    all_regions: List[str],
    target_matches: int,
    seen_match_ids: Set[str],
    rows: List[Dict],
    claims: List[int],
    lock: threading.RLock,
    sleep_seconds: float,
    max_players_per_region: int,
    output_path: Path,
    checkpoint_every: int,
    checkpoint_ref: List[int],
    max_worker_iterations: int,
    dry_before_cd: int,
    cooldown_rounds: int,
    no_progress_timeout_seconds: int,
):
    cooldown_until: Dict[str, int] = {}
    dry_streak: Dict[str, int] = {r: 0 for r in all_regions}
    it = 0
    last_progress_ts = time.time()

    while it < max_worker_iterations and not _rows_done(rows, lock, target_matches):
        with lock:
            if _rows_done(rows, lock, target_matches):
                break

            eligible = [r for r in all_regions if it >= cooldown_until.get(r, 0)]
            if not eligible:
                cooldown_until.clear()
                eligible = list(all_regions)

            rc = get_row_counts_per_region(rows)
            tpr = max(1, target_matches // len(all_regions))
            order = sorted(eligible, key=lambda r: rc.get(r, 0))

            pick = None
            for r in order:
                if rc.get(r, 0) < tpr:
                    pick = r
                    break
            if pick is None and not _rows_done(rows, lock, target_matches):
                pick = order[0] if order else None

            rows_before = len(rows)

        if not pick:
            break

        try:
            api = RiotAPI(api_key=api_key, region=pick)
            with lock:
                rc = get_row_counts_per_region(rows)
                rem = max(1, tpr - rc.get(pick, 0) + 10)

            collect_from_region(
                api,
                api_key,
                target_matches,
                seen_match_ids,
                rows,
                claims,
                sleep_seconds,
                output_path,
                checkpoint_every,
                lock,
                checkpoint_ref,
                max_players_per_region=max_players_per_region,
                max_matches_per_region=rem,
            )

            with lock:
                rows_after = len(rows)

            if rows_after == rows_before:
                dry_streak[pick] = dry_streak.get(pick, 0) + 1
                if dry_streak[pick] >= dry_before_cd:
                    cooldown_until[pick] = it + cooldown_rounds
                    dry_streak[pick] = 0
                if it % 15 == 0:
                    print(
                        f"[HEARTBEAT] keine neue Zeile | iter={it} region={pick} rows={rows_after}",
                        flush=True,
                    )
            else:
                dry_streak[pick] = 0
                last_progress_ts = time.time()
                print(f"[PROGRESS] rows={rows_after} region={pick}", flush=True)
        except Exception as exc:
            dry_streak[pick] = dry_streak.get(pick, 0) + 1
            print(f"[WARN] Region-Loop Fehler ({pick}): {exc}", flush=True)

        if no_progress_timeout_seconds > 0 and (time.time() - last_progress_ts) > no_progress_timeout_seconds:
            raise RuntimeError(
                f"Seit {no_progress_timeout_seconds}s kein Fortschritt (rows={len(rows)})."
            )

        it += 1


def collect_routing_worker(
    api_key: str,
    _routing_value: str,
    regs: List[str],
    target_matches: int,
    seen_match_ids: Set[str],
    rows: List[Dict],
    claims: List[int],
    lock: threading.RLock,
    n_all: int,
    sleep_seconds: float,
    max_players_per_region: int,
    output_path: Path,
    checkpoint_every: int,
    checkpoint_ref: List[int],
    max_worker_iterations: int,
):
    it = 0
    while it < max_worker_iterations and not _rows_done(rows, lock, target_matches):
        with lock:
            if _rows_done(rows, lock, target_matches):
                break
            rc = get_row_counts_per_region(rows)
            tpr = max(1, target_matches // n_all)
            order = sorted(regs, key=lambda r: rc.get(r, 0))
            pick = None
            for r in order:
                if rc.get(r, 0) < tpr:
                    pick = r
                    break
            if pick is None:
                pick = order[0] if order else None
        if not pick:
            break
        try:
            api = RiotAPI(api_key=api_key, region=pick)
            with lock:
                rc = get_row_counts_per_region(rows)
                rem = max(1, tpr - rc.get(pick, 0) + 10)
            collect_from_region(
                api,
                api_key,
                target_matches,
                seen_match_ids,
                rows,
                claims,
                sleep_seconds,
                output_path,
                checkpoint_every,
                lock,
                checkpoint_ref,
                max_players_per_region=max_players_per_region,
                max_matches_per_region=rem,
            )
        except Exception:
            pass
        it += 1


def build_dataset(
    api_key: str,
    target_matches: int,
    output_path: Path,
    regions: Optional[List[str]],
    use_parallel: bool,
    sleep_seconds: float,
    max_players_per_region: int,
    checkpoint_every: int,
    resume: bool,
    max_worker_iterations: int,
    dry_before_cd: int,
    cooldown_rounds: int,
    no_progress_timeout_seconds: int,
):
    if regions is None:
        regions = list(REGION_CONFIGS.keys())

    groups: Dict[str, List[str]] = {}
    for r in regions:
        groups.setdefault(REGION_CONFIGS[r]["v5_region"], []).append(r)

    rows: List[Dict] = []
    seen: Set[str] = set()
    claims = [0]
    checkpoint_ref = [0]
    lock = threading.RLock()

    if output_path.exists() and not resume:
        output_path.unlink()

    if resume and output_path.exists():
        try:
            df = pd.read_csv(output_path)
            for rec in df.to_dict("records"):
                mid = str(rec.get("match_id", ""))
                if mid:
                    seen.add(mid)
                    rows.append(rec)
            claims[0] = len(rows)
            checkpoint_ref[0] = len(rows)
            print(f"[RESUME] {len(rows)} Zeilen aus {output_path} geladen.")
        except Exception as exc:
            print(f"[WARN] Resume fehlgeschlagen, starte leer: {exc}")
            rows.clear()
            seen.clear()
            claims[0] = 0

    global _STATE
    _STATE = {"output_path": output_path, "rows": rows, "lock": lock, "checkpoint_ref": checkpoint_ref}
    print(
        f"[START] target={target_matches} regions={len(regions)} checkpoint_every={checkpoint_every} "
        f"resume_rows={len(rows)}",
        flush=True,
    )

    try:
        if use_parallel and len(groups) > 1:
            th = []
            for rv, gr in groups.items():
                t = threading.Thread(
                    target=collect_routing_worker,
                    args=(
                        api_key,
                        rv,
                        gr,
                        target_matches,
                        seen,
                        rows,
                        claims,
                        lock,
                        len(regions),
                        sleep_seconds,
                        max_players_per_region,
                        output_path,
                        checkpoint_every,
                        checkpoint_ref,
                        max_worker_iterations,
                    ),
                    daemon=True,
                )
                th.append(t)
                t.start()
            for t in th:
                t.join()
        else:
            collect_global_worker(
                api_key,
                regions,
                target_matches,
                seen,
                rows,
                claims,
                lock,
                sleep_seconds,
                max_players_per_region,
                output_path,
                checkpoint_every,
                checkpoint_ref,
                max_worker_iterations,
                dry_before_cd,
                cooldown_rounds,
                no_progress_timeout_seconds,
            )
    finally:
        if rows:
            write_checkpoint(output_path, rows, lock, checkpoint_ref)

    cnt = get_row_counts_per_region(rows)
    print(f"[DONE] {len(rows)} Zeilen -> {output_path}")
    for r in regions:
        print(f"  {r}: {cnt.get(r, 0)}")


def main():
    args = parse_args()
    output_path = Path(args.output)
    regs = [x.strip().lower() for x in args.regions.split(",") if x.strip()] or None

    try:
        probe_api = RiotAPI(api_key=args.api_key or None, region=next(iter(REGION_CONFIGS)))
        effective_api_key = probe_api.api_key
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    def _on_signal(_signum, _frame):
        _flush_state()

    try:
        signal.signal(signal.SIGINT, _on_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _on_signal)
    except Exception:
        pass
    atexit.register(_flush_state)

    build_dataset(
        effective_api_key,
        args.target_matches,
        output_path,
        regs,
        args.parallel,
        max(0.0, args.sleep_ms / 1000.0),
        args.max_players_per_region,
        args.checkpoint_every,
        args.resume,
        args.max_worker_iterations,
        args.dry_rounds_before_cooldown,
        args.cooldown_rounds,
        args.no_progress_timeout_seconds,
    )


if __name__ == "__main__":
    main()
