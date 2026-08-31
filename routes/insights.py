"""Insights endpoint: focus, velocity, radar vectors, per-project/board/task
telemetry and the timestamped session history for the dashboard.

Everything here is computed from real data already in the database:

* ``activity.drops`` — periodic focus polls ("drops") classified by the user as
  ``Locked-In`` / ``Trying...`` / ``Distracted``. These drive the focus score.
* ``session.logs`` — study segments with accumulated ``total/cam/ss/noact``.
  When a user has no drop data the idle ("noact") drop ratio is used instead.
* ``tasks.log`` — task events with timestamps, used to approximate time spent
  per project/board (gaps between events, capped so idle gaps don't count).
* ``projects.docs`` / ``boards.docs`` — project/board titles and real task
  statuses (todo/cooking/done) for completion counts.
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Request

from . import limiter, rate_limit_ip, verify_token
from .stats import _parse_iso

router = APIRouter()

LOOKBACK_DAYS = 56
EVENT_GAP_CAP_SECONDS = 45 * 60

STUDY_SCORE = {"Locked-In": 100, "Trying...": 60, "Distracted": 25}
RADAR_NAMES = ["Consistency", "Deep Work Length", "Goal Velocity", "Focus Score", "Punctuality"]
RADAR_BASELINE = [70, 62, 65, 71, 58]
PROJECT_COLORS = [
    "#f472b6", "#a78bfa", "#fb923c", "#34d399", "#38bdf8",
    "#fbbf24", "#f87171", "#2dd4bf", "#c084fc", "#94a3b8",
]


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _focus_from_drops(drops):
    """Weighted mean of drop scores; unclaimed drops (claim_rate < 1) weigh less."""
    if not drops:
        return None
    total_weight = sum(d.get("claim", 0) for d in drops) or 1.0
    return round(sum(d["score"] for d in drops) / total_weight, 1)


async def _load_raw(db, user_id, now):
    """Load the raw data needed to derive every metric for one user."""
    segments = []
    async for doc in db["session.logs"].find({"user_id": user_id}):
        joined = doc.get("joined_at") or {}
        start = _parse_iso(joined.get("time"))
        end = _parse_iso(doc.get("left_at")) or now
        if start is None:
            continue
        if end < start:
            end = start
        total = max(0.0, float(joined.get("total") or 0))
        noact = max(0.0, float(joined.get("noact") or 0))
        segments.append(
            {
                "session_id": doc.get("session_id"),
                "start": start,
                "end": end,
                "total": total,
                "cam": max(0.0, float(joined.get("cam") or 0)),
                "ss": max(0.0, float(joined.get("ss") or 0)),
                "noact": noact,
                "live": doc.get("left_at") is None,
            }
        )

    drops = []
    async for doc in db["activity.drops"].find({"user_id": user_id}):
        claimed = _parse_iso(doc.get("claimed_at")) or now
        claim = float(doc.get("claim_rate") or 0.0)
        score = STUDY_SCORE.get(doc.get("study_type"), 0)
        drops.append(
            {"t": claimed, "score": score * claim, "claim": claim, "raw": score}
        )

    events = []
    async for doc in db["tasks.log"].find({"user_id": user_id}):
        t = _parse_iso(doc.get("occured_at"))
        if t is None:
            continue
        events.append(
            {"t": t, "project_id": doc.get("project_id"), "board_id": doc.get("board_id")}
        )

    completed = 0
    total_tasks = 0
    boards = []
    async for doc in db["boards.docs"].find({"user_id": user_id}):
        tasks = doc.get("tasks") or {}
        board_done = 0
        board_total = 0
        for task in tasks.values():
            if not isinstance(task, dict):
                continue
            board_total += 1
            total_tasks += 1
            if task.get("status") == "done":
                board_done += 1
                completed += 1
        boards.append(
            {
                "board_id": doc.get("board_id"),
                "project_id": doc.get("project_id"),
                "title": doc.get("title") or doc.get("board_id"),
                "tasks": tasks,
                "done": board_done,
                "total": board_total,
            }
        )

    projects = {}
    async for doc in db["projects.docs"].find({"user_id": user_id}):
        b = doc.get("boards") or {}
        projects[doc.get("project_id")] = {
            "title": doc.get("title") or doc.get("project_id"),
            "done": int(b.get("done") or 0),
            "total": int((b.get("todo") or 0) + (b.get("cooking") or 0) + (b.get("done") or 0)),
        }

    return {
        "segments": segments,
        "drops": drops,
        "events": events,
        "boards": boards,
        "projects": projects,
        "completed": completed,
        "total_tasks": total_tasks,
    }


def _vectors(raw, now, today):
    """Derive the five radar vectors (0-100) from raw data."""
    segments = raw["segments"]

    days = set()
    first_start = {}
    for s in segments:
        d = s["end"].date()
        if s["total"] > 0 and (today - d).days <= LOOKBACK_DAYS:
            days.add(d)
            first_start.setdefault(d, s["start"])
    days_studied = len(days)

    consistency = round(days_studied / LOOKBACK_DAYS * 100)

    active_mins = [s["total"] / 60 for s in segments if s["total"] > 0]
    avg_session_min = _mean(active_mins)
    deep_work = min(100, round(avg_session_min / 75 * 100))

    study_hours = sum(s["total"] for s in segments) / 3600
    tasks_per_day = round(raw["completed"] / max(1, days_studied), 2)
    velocity = min(100, round(tasks_per_day / 4 * 100))

    focus = _focus_from_drops(raw["drops"])
    if focus is None:
        total = sum(s["total"] for s in segments)
        noact = sum(s["noact"] for s in segments)
        focus = round(100 * (1 - noact / total), 1) if total > 0 else 0.0

    # ── Per-period focus: drops bucketed by claim date, else segment ratio ──
    day_scores = defaultdict(lambda: [0.0, 0.0])
    for d in raw["drops"]:
        day = d["t"].date()
        day_scores[day][0] += d["score"]
        day_scores[day][1] += d["claim"]
    focus_by_day = {
        day.isoformat(): round(s / (c or 1.0), 1)
        for day, (s, c) in sorted(day_scores.items())
    }

    def _period_focus(since_days):
        since = today - timedelta(days=since_days)
        scores = [(s, c) for day, (s, c) in day_scores.items() if day >= since]
        if scores:
            s = sum(x[0] for x in scores)
            c = sum(x[1] for x in scores)
            return round(s / (c or 1.0), 1)
        total = sum(x["total"] for x in segments if x["end"].date() >= since)
        noact = sum(x["noact"] for x in segments if x["end"].date() >= since)
        return round(100 * (1 - noact / total), 1) if total > 0 else None

    focus_30d = _period_focus(30)
    focus_7d = _period_focus(7)
    focus_today = _period_focus(0)

    punctuality = 0
    if first_start:
        hours = [fs.hour for fs in first_start.values()]
        modal = max(set(hours), key=hours.count)
        punctuality = round(
            sum(1 for h in hours if abs(h - modal) <= 2 or abs(h - modal) >= 22) / len(hours) * 100
        )

    return {
        "vectors": [
            {"vector": RADAR_NAMES[0], "userScore": consistency},
            {"vector": RADAR_NAMES[1], "userScore": deep_work},
            {"vector": RADAR_NAMES[2], "userScore": velocity},
            {"vector": RADAR_NAMES[3], "userScore": round(focus)},
            {"vector": RADAR_NAMES[4], "userScore": punctuality},
        ],
        "focus": round(focus, 1),
        "focus_30d": focus_30d,
        "focus_7d": focus_7d,
        "focus_today": focus_today,
        "focus_by_day": focus_by_day,
        "tasks_per_day": tasks_per_day,
        "days_studied": days_studied,
        "completed": raw["completed"],
        "total_tasks": raw["total_tasks"],
        "study_hours": round(study_hours, 1),
        "avg_session_min": round(avg_session_min, 1),
    }


def _attributed_time(events, now, cap=EVENT_GAP_CAP_SECONDS):
    """Approximate time spent per project/board by gap-attributing task events.

    Each inter-event gap (capped at ``cap``) is charged to the project/board of
    the event that follows it; a trailing capped gap is added toward now.
    """
    proj_secs = defaultdict(float)
    board_secs = defaultdict(float)
    proj_events = defaultdict(int)
    board_events = defaultdict(int)

    prev_t = None
    for ev in events:
        if prev_t is not None:
            gap = min((ev["t"] - prev_t).total_seconds(), cap)
            if ev["project_id"]:
                proj_secs[ev["project_id"]] += gap
            if ev["board_id"]:
                board_secs[ev["board_id"]] += gap
        if ev["project_id"]:
            proj_events[ev["project_id"]] += 1
        if ev["board_id"]:
            board_events[ev["board_id"]] += 1
        prev_t = ev["t"]
    if events:
        last = events[-1]
        trail = min(max(0.0, (now - last["t"]).total_seconds()), cap)
        if last["project_id"]:
            proj_secs[last["project_id"]] += trail
        if last["board_id"]:
            board_secs[last["board_id"]] += trail

    return proj_secs, board_secs, proj_events, board_events


def _session_list(raw):
    """Group segments by session and shape the timestamped history log."""
    merged = defaultdict(
        lambda: {"start": None, "end": None, "total": 0.0, "cam": 0.0, "ss": 0.0, "noact": 0.0, "live": False}
    )
    for s in raw["segments"]:
        key = s["session_id"] or s["start"].isoformat()
        m = merged[key]
        if m["start"] is None or s["start"] < m["start"]:
            m["start"] = s["start"]
        if m["end"] is None or s["end"] > m["end"]:
            m["end"] = s["end"]
        m["total"] += s["total"]
        m["cam"] += s["cam"]
        m["ss"] += s["ss"]
        m["noact"] += s["noact"]
        m["live"] = m["live"] or s["live"]

    sessions = []
    for key, m in merged.items():
        if m["start"] is None:
            continue
        both = max(0.0, m["cam"] + m["ss"] + m["noact"] - m["total"])
        cam_only = max(0.0, m["cam"] - both)
        ss_only = max(0.0, m["ss"] - both)
        if both > 0:
            mode = "cam_ss"
        elif cam_only > 0:
            mode = "cam_only"
        elif ss_only > 0:
            mode = "ss_only"
        else:
            mode = "noact"
        active = max(0.0, m["total"] - m["noact"])
        focus = round(100 * (active / m["total"])) if m["total"] > 0 else 0
        sessions.append(
            {
                "id": key,
                "start": m["start"].isoformat(),
                "end": m["end"].isoformat(),
                "duration_minutes": round(max(m["total"], (m["end"] - m["start"]).total_seconds()) / 60),
                "mode": mode,
                "focus": focus,
                "status": "cooking" if m["live"] else "done",
            }
        )
    sessions.sort(key=lambda s: s["start"], reverse=True)
    return sessions[:10]


def _per_drop_week(drops, now, days):
    """Split drops into the current window vs the window before it."""
    split = {"current": [], "previous": []}
    for d in drops:
        age = (now - d["t"]).total_seconds() / 86400
        if age < days:
            split["current"].append(d)
        elif age < days * 2:
            split["previous"].append(d)
    return split


@router.get("/insights")
@limiter.limit("60/minute", key_func=rate_limit_ip)
async def get_insights(request: Request, payload: dict = Depends(verify_token)):
    user_id = payload.get("sub")
    now = datetime.now(timezone.utc)
    today = now.date()
    db = request.app.db

    raw = await _load_raw(db, user_id, now)
    metrics = _vectors(raw, now, today)
    vectors = metrics["vectors"]

    # ── Peer / benchmark across users that have data ─────────────────────
    # Bounded: we only need each peer's focus + radar vectors, which derive
    # from segments, drops and completed-task counts (see _vectors). Loading a
    # full _load_raw per peer (5 collection scans) is quadratic and stalls the
    # dashboard, so we fetch just those three, cheaply, and cap the sample.
    peer_ids = set()
    for coll in ("session.logs", "activity.drops", "boards.docs"):
        async for doc in db[coll].find(
            {}, {"user_id": 1}
        ):
            if doc.get("user_id") and doc["user_id"] != user_id:
                peer_ids.add(doc["user_id"])

    PEER_SAMPLE_CAP = 40
    peer_sample = sorted(peer_ids)[:PEER_SAMPLE_CAP]

    peer_vectors = []
    peer_scores = []
    for pid in peer_sample:
        try:
            segments = []
            async for doc in db["session.logs"].find(
                {"user_id": pid}, {"joined_at": 1, "left_at": 1}
            ):
                joined = doc.get("joined_at") or {}
                start = _parse_iso(joined.get("time"))
                if start is None:
                    continue
                end = _parse_iso(doc.get("left_at")) or now
                if end < start:
                    end = start
                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "total": max(0.0, float(joined.get("total") or 0)),
                        "noact": max(0.0, float(joined.get("noact") or 0)),
                    }
                )

            drops = []
            async for doc in db["activity.drops"].find(
                {"user_id": pid}, {"claimed_at": 1, "claim_rate": 1, "study_type": 1}
            ):
                claimed = _parse_iso(doc.get("claimed_at")) or now
                claim = float(doc.get("claim_rate") or 0.0)
                score = STUDY_SCORE.get(doc.get("study_type"), 0)
                drops.append({"t": claimed, "score": score * claim, "claim": claim})

            completed = 0
            async for doc in db["boards.docs"].find(
                {"user_id": pid}, {"tasks": 1}
            ):
                for task in (doc.get("tasks") or {}).values():
                    if isinstance(task, dict) and task.get("status") == "done":
                        completed += 1

            prow = {
                "segments": segments,
                "drops": drops,
                "completed": completed,
                "total_tasks": 0,
            }
            pmetrics = _vectors(prow, now, today)
            peer_vectors.append(pmetrics["vectors"])
            peer_scores.append(pmetrics["focus"])
        except Exception:
            continue

    focus = metrics["focus"]
    global_avg = []
    for idx in range(len(RADAR_NAMES)):
        vals = [v[idx]["userScore"] for v in peer_vectors if v]
        global_avg.append(round(_mean(vals)) if vals else RADAR_BASELINE[idx])
    for idx, vec in enumerate(vectors):
        vec["globalAvg"] = global_avg[idx]

    # "Top X%" focus benchmark among all tracked users (self included).
    all_scores = sorted([focus] + [s for s in peer_scores if s is not None], reverse=True)
    rank = all_scores.index(focus) + 1 if focus in all_scores else len(all_scores)
    top_percent = round(100 * rank / len(all_scores)) if all_scores else 100

    # ── Week-over-week deltas ──────────────────────────────────────────────
    week_secs = 0.0
    prev_week_secs = 0.0
    for s in raw["segments"]:
        age = (now - s["end"]).total_seconds() / 86400
        if age <= 7:
            week_secs += s["total"]
        elif age <= 14:
            prev_week_secs += s["total"]
    week_delta = round((week_secs - prev_week_secs) / prev_week_secs * 100) if prev_week_secs > 0 else 0

    focus_weeks = _per_drop_week(raw["drops"], now, 7)
    f_cur = _focus_from_drops(focus_weeks["current"])
    f_prev = _focus_from_drops(focus_weeks["previous"])
    focus_delta = round(f_cur - f_prev, 1) if (f_cur is not None and f_prev is not None) else 0

    def _event_rate(events, now, window_days):
        cutoff = now - timedelta(days=window_days)
        return sum(1 for ev in events if ev["t"] >= cutoff)

    events_week = _event_rate(raw["events"], now, 7)
    events_prev = _event_rate(raw["events"], now, 14) - events_week
    velocity_delta = round((events_week - events_prev) / events_prev * 100) if events_prev > 0 else 0

    # ── Per-project / board / task telemetry ───────────────────────────────
    proj_secs, board_secs, proj_events, board_events = _attributed_time(raw["events"], now)

    projects = []
    for pid, secs in sorted(proj_secs.items(), key=lambda kv: kv[1], reverse=True)[:8]:
        if pid not in raw["projects"]:
            continue
        meta = raw["projects"][pid]
        projects.append(
            {
                "id": pid,
                "title": meta.get("title") or "Untitled project",
                "hours": round(secs / 3600, 1),
                "tasks": proj_events.get(pid, 0),
                "done": meta.get("done", 0),
                "focus": round(focus),
                "color": PROJECT_COLORS[len(projects) % len(PROJECT_COLORS)],
            }
        )

    board_by_id = {b["board_id"]: b for b in raw["boards"]}
    boards = []
    for bid, secs in sorted(board_secs.items(), key=lambda kv: kv[1], reverse=True)[:8]:
        b = board_by_id.get(bid)
        if not b:
            continue
        project_meta = raw["projects"].get(b.get("project_id")) or {}
        boards.append(
            {
                "id": bid,
                "title": b.get("title") or "Untitled board",
                "project": project_meta.get("title") or "Unknown project",
                "hours": round(secs / 3600, 1),
                "tasks": board_events.get(bid, 0),
                "done": b.get("done", 0),
                "focus": round(focus),
                "color": PROJECT_COLORS[len(boards) % len(PROJECT_COLORS)],
            }
        )

    tasks = []
    for b in raw["boards"]:
        board_sec = board_secs.get(b["board_id"], 0.0)
        task_items = [t for t in b["tasks"].values() if isinstance(t, dict) and t.get("text")]
        if not task_items:
            continue
        per_task = board_sec / len(task_items) / 3600
        project_meta = raw["projects"].get(b["project_id"]) or {}
        for t in task_items:
            tasks.append(
                {
                    "id": t.get("created_at") or t.get("text"),
                    "text": t.get("text"),
                    "project": project_meta.get("title") or "Unknown project",
                    "board": b["title"],
                    "status": t.get("status", "todo"),
                    "durationHrs": round(per_task, 1),
                    "focus": round(focus),
                }
            )
    tasks.sort(key=lambda t: t["durationHrs"], reverse=True)
    tasks = tasks[:8]

    # ── Peak concentration hour (drops-weighted, else session hours) ──────
    peak_hour = None
    if raw["drops"]:
        hour_scores = defaultdict(float)
        for d in raw["drops"]:
            hour_scores[d["t"].hour] += d["score"]
        if hour_scores:
            peak_hour = max(hour_scores, key=hour_scores.get)
    if peak_hour is None and raw["segments"]:
        hour_scores = defaultdict(float)
        for s in raw["segments"]:
            hour_scores[s["start"].hour] += s["total"]
        if hour_scores:
            peak_hour = max(hour_scores, key=hour_scores.get)

    return {
        "ok": 1,
        "insights": {
            "focus": {
                "score": round(focus, 1),
                "delta": focus_delta,
                "top_percent": top_percent,
                "peers": len(peer_ids) + 1,
                "samples": len(raw["drops"]),
                "peak_hour": peak_hour,
                "focus_30d": metrics["focus_30d"],
                "focus_7d": metrics["focus_7d"],
                "focus_today": metrics["focus_today"],
                "focus_by_day": metrics["focus_by_day"],
            },
            "velocity": {
                "tasks_per_day": metrics["tasks_per_day"],
                "delta": velocity_delta,
                "completed": metrics["completed"],
                "total_tasks": metrics["total_tasks"],
            },
            "radar": vectors,
            "projects": projects,
            "boards": boards,
            "tasks": tasks,
            "sessions": _session_list(raw),
            "deltas": {"week": week_delta, "focus": focus_delta, "velocity": velocity_delta},
        },
    }
