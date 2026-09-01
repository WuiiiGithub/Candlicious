import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from . import verify_token, limiter, rate_limit_ip, rate_limit_user, _client_ip
from library.avatars import resolve_avatar_url, default_avatar
import config as app_config

logger = logging.getLogger(__name__)

router = APIRouter()

# ── SSE connection caps (anti DDoS / socket-exhaustion) ──
# Each open /stream connection holds a socket (and a Mongo queue subscription),
# so a small number of clients can exhaust file descriptors. Cap concurrent
# connections per client IP and per session.
from collections import defaultdict

_sse_by_ip: dict[str, int] = defaultdict(int)
_sse_by_session: dict[str, int] = defaultdict(int)
MAX_SSE_PER_IP = 5
MAX_SSE_PER_SESSION = 10


class JoinRequest(BaseModel):
    initial_time: dict | None = None


class AbsorbRequest(BaseModel):
    target_session_id: str


def _normalize_initial_time(raw: dict | None) -> dict:
    """Whitelist initial_time so clients cannot smuggle arbitrary fields into
    the members document."""
    if not isinstance(raw, dict):
        return {"cam": 0, "ss": 0, "noact": 0, "total": 0}
    clean = {}
    for key in ("cam", "ss", "noact", "total"):
        try:
            value = int(raw.get(key, 0))
        except (TypeError, ValueError):
            value = 0
        clean[key] = max(0, value)
    clean["total"] = clean["cam"] + clean["ss"] + clean["noact"]
    return clean


async def _is_live_guild_member(
    request: Request,
    guild_id: str,
    user_id: str,
    in_guild_claim: bool,
) -> bool:
    """Prefer a live membership check against the bot's member cache.

    The `in_guild` JWT claim is a login-time snapshot, so a user kicked from
    the guild would keep access until the cookie expires. When the bot is
    connected and the guild has been chunked, the live check decides. When the
    bot is unavailable or the guild cache is still loading, fall back to the
    claim so legitimate members are never locked out."""
    if not guild_id:
        return True
    bot = getattr(request.app.state, "bot", None)
    if bot:
        try:
            guild = bot.get_guild(int(guild_id))
        except (TypeError, ValueError):
            guild = None
        if guild and getattr(guild, "chunked", True):
            try:
                return guild.get_member(int(user_id)) is not None
            except (TypeError, ValueError):
                return False
    return bool(in_guild_claim)


async def _enrich_members(request: Request, session_doc: dict) -> list:
    raw_members = session_doc.get("members", {})
    if not raw_members:
        return []

    member_ids = list(raw_members.keys())
    if not member_ids:
        return []

    user_cursor = request.app.db["users"].find(
        {"_id": {"$in": member_ids}},
        {"name": 1, "display_name": 1, "pfp": 1, "profile_pfp": 1},
    )

    user_map = {}
    async for doc in user_cursor:
        uid = doc["_id"]
        username = doc.get("name", "Unknown")
        display_name = doc.get("display_name") or username
        avatar_url = resolve_avatar_url(uid, doc)
        user_map[uid] = {
            "username": username,
            "display_name": display_name,
            "avatar_url": avatar_url,
        }

    owner_id = session_doc.get("owner_id")
    members = []
    for uid, mdata in raw_members.items():
        if not isinstance(mdata, dict):
            continue
        uinfo = user_map.get(uid, {
            "username": "Unknown",
            "display_name": "Unknown",
            "avatar_url": default_avatar(uid),
        })
        members.append({
            "user_id": uid,
            "username": uinfo["username"],
            "display_name": uinfo["display_name"],
            "avatar_url": uinfo["avatar_url"],
            "activity": mdata.get("last_activity", "noact"),
            "net_time": mdata.get("net_time", {"cam": 0, "ss": 0, "noact": 0, "total": 0}),
            "is_owner": uid == owner_id,
            "is_web_user": mdata.get("is_web_user", False),
        })

    members.sort(key=lambda m: (not m["is_owner"], m["user_id"]))
    return members



def _session_response(session_doc: dict, members: list) -> dict:
    return {
        "ok": 1,
        "session": {
            "session_id": session_doc.get("session_id"),
            "owner_id": session_doc.get("owner_id"),
            "guild_id": session_doc.get("guild_id"),
            "channel_id": session_doc.get("channel_id"),
            "session_type": session_doc.get("session_type", "*"),
            "vc_level": session_doc.get("vc_level", 1),
            "vc_xp": session_doc.get("vc_xp", 0),
            "members_count": session_doc.get("members_count", {}),
            "members": members,
            "pending_level_up": session_doc.get("pending_level_up"),
            "pomodoro": {
                "enabled": session_doc.get("pomodoro_enabled", False),
                "running": session_doc.get("pomodoro_running", False),
                "state": session_doc.get("pomodoro_state", "idle"),
                "focus_min": session_doc.get("pomodoro_focus_min", 25),
                "break_min": session_doc.get("pomodoro_break_min", 5),
                "ends_at": session_doc.get("pomodoro_ends_at"),
                "cycles": session_doc.get("pomodoro_cycles", 0),
            },
        },
    }


async def _remove_user_from_session(request: Request, user_id: str, session_doc: dict, event_bus=None):
    members_raw = session_doc.get("members", {})
    if user_id not in members_raw:
        return

    session_id = session_doc.get("session_id")

    user_act = members_raw[user_id].get("last_activity", "noact") if isinstance(members_raw[user_id], dict) else "noact"
    inc_fields = {"members_count.total": -1}
    if user_act == "cam+ss":
        inc_fields["members_count.cam"] = -1
        inc_fields["members_count.ss"] = -1
    elif user_act in ("cam", "ss", "noact"):
        inc_fields[f"members_count.{user_act}"] = -1

    updated_doc = await request.app.db["sessions"].find_one_and_update(
        {"session_id": session_id},
        {
            "$unset": {f"members.{user_id}": ""},
            "$inc": inc_fields,
        },
        return_document=True,
    )

    owner_id = session_doc.get("owner_id")
    new_owner_id = owner_id
    remaining = [mid for mid in members_raw if mid != user_id]

    if user_id == owner_id:
        new_owner_id = remaining[0] if remaining else None
    if not updated_doc:
        sm = getattr(getattr(request.app.state, "bot", None), "session_manager", None)
        if sm and session_id in sm.active_sessions:
            sm._cleanup_session(sm.active_sessions[session_id])
        return

    total = updated_doc.get("members_count", {}).get("total", 0)
    if total <= 0:
        sm = getattr(getattr(request.app.state, "bot", None), "session_manager", None)
        if sm and session_id in sm.active_sessions:
            sm._cleanup_session(sm.active_sessions[session_id])
        else:
            await request.app.db["users"].update_one(
                {"_id": user_id},
                {"$unset": {"current_session": "", "webToken": ""}},
            )
            await request.app.db["sessions"].delete_one({"session_id": session_id})
            if sm:
                for uid, sid in list(sm.user_sessions.items()):
                    if sid == session_id:
                        del sm.user_sessions[uid]
                for ch, sid in list(sm.channel_sessions.items()):
                    if sid == session_id:
                        del sm.channel_sessions[ch]
        if event_bus:
            await event_bus.publish(session_id, "session_closed", {})
        return

    if new_owner_id and new_owner_id != owner_id:
        await request.app.db["sessions"].update_one(
            {"session_id": session_id},
            {"$set": {"owner_id": new_owner_id}},
        )

    has_discord = any(
        not m.get("is_web_user", False)
        for m in updated_doc.get("members", {}).values()
        if isinstance(m, dict)
    )
    channel_id = updated_doc.get("channel_id", "")
    if not channel_id.startswith("w") and not has_discord and remaining:
        await request.app.db["sessions"].update_one(
            {"session_id": session_id},
            {"$set": {
                "channel_id": f"w{new_owner_id}",
                "guild_id": "web",
            }},
        )

    if event_bus:
        await event_bus.publish(session_id, "member_leave", {"user_id": user_id})
        if new_owner_id != owner_id:
            await event_bus.publish(session_id, "owner_change", {"owner_id": new_owner_id})

    await request.app.db["users"].update_one(
        {"_id": user_id},
        {"$unset": {"current_session": ""}},
    )

    sm = getattr(getattr(request.app.state, "bot", None), "session_manager", None)
    if sm and session_id in sm.active_sessions:
        sess = sm.active_sessions[session_id]
        sess.members.pop(user_id, None)
        sess.owner_id = new_owner_id
        old_ch = sess.channel_id
        new_ch = updated_doc.get("channel_id", sess.channel_id)
        sess.channel_id = new_ch
        sess.guild_id = updated_doc.get("guild_id", sess.guild_id)
        sess._update_members_count()
        sm.user_sessions.pop(user_id, None)
        if new_ch != old_ch:
            sm.channel_sessions.pop(old_ch, None)
            sm.channel_sessions[new_ch] = session_id
        await request.app.db["sessions"].update_one(
            {"session_id": session_id},
            {"$set": {"members_count": sess.members_count}},
        )


@router.get("/my-active")
@limiter.limit("30/minute", key_func=rate_limit_ip)
@limiter.limit("60/hour", key_func=rate_limit_user)
async def get_my_active_session(
    request: Request,
    payload: dict = Depends(verify_token),
):
    user_id = payload.get("sub")
    user_doc = await request.app.db["users"].find_one(
        {"_id": user_id},
        {"current_session": 1},
    )
    if not user_doc or not user_doc.get("current_session"):
        return {"session_id": None}

    session_id = user_doc["current_session"]
    session_doc = await request.app.db["sessions"].find_one({"session_id": session_id})
    if not session_doc:
        await request.app.db["users"].update_one(
            {"_id": user_id},
            {"$unset": {"current_session": ""}},
        )
        return {"session_id": None}

    members_raw = session_doc.get("members", {})
    if user_id not in members_raw:
        return {"session_id": None}

    return {
        "session_id": session_id,
        "guild_id": session_doc.get("guild_id", ""),
        "channel_id": session_doc.get("channel_id", ""),
        "owner_id": session_doc.get("owner_id"),
        "is_owner": session_doc.get("owner_id") == user_id,
    }


@router.post("/{session_id}/absorb")
@limiter.limit("10/minute", key_func=rate_limit_ip)
@limiter.limit("20/hour", key_func=rate_limit_user)
async def absorb_session(
    request: Request,
    session_id: str,
    body: AbsorbRequest,
    payload: dict = Depends(verify_token),
):
    user_id = payload.get("sub")
    target_session_id = body.target_session_id

    web_doc = await request.app.db["sessions"].find_one({"session_id": session_id})
    if not web_doc:
        raise HTTPException(status_code=404, detail="Web session not found")

    web_members = web_doc.get("members", {})
    if user_id not in web_members:
        raise HTTPException(status_code=400, detail="You are not in this session")

    if web_doc.get("owner_id") != user_id:
        raise HTTPException(status_code=403, detail="Only the session owner can absorb")

    target_doc = await request.app.db["sessions"].find_one({"session_id": target_session_id})
    if not target_doc:
        raise HTTPException(status_code=404, detail="Target Discord session not found")

    target_members = target_doc.get("members", {})
    if user_id not in target_members:
        raise HTTPException(status_code=400, detail="You are not in the target session")

    web_member_data = web_members[user_id]
    if isinstance(web_member_data, dict):
        web_time = web_member_data.get("net_time", {"cam": 0, "ss": 0, "noact": 0, "total": 0})
    else:
        web_time = {"cam": 0, "ss": 0, "noact": 0, "total": 0}

    target_member_data = target_members.get(user_id, {})
    if isinstance(target_member_data, dict):
        existing_time = target_member_data.get("net_time", {"cam": 0, "ss": 0, "noact": 0, "total": 0})
    else:
        existing_time = {"cam": 0, "ss": 0, "noact": 0, "total": 0}

    merged_time = {
        "cam": existing_time.get("cam", 0) + web_time.get("cam", 0),
        "ss": existing_time.get("ss", 0) + web_time.get("ss", 0),
        "noact": existing_time.get("noact", 0) + web_time.get("noact", 0),
        "total": existing_time.get("total", 0) + web_time.get("total", 0),
    }

    await request.app.db["sessions"].update_one(
        {"session_id": target_session_id},
        {"$set": {f"members.{user_id}.net_time": merged_time}},
    )

    sm = getattr(getattr(request.app.state, "bot", None), "session_manager", None)
    if sm and target_session_id in sm.active_sessions:
        sess = sm.active_sessions[target_session_id]
        if user_id in sess.members and isinstance(sess.members[user_id], dict):
            sess.members[user_id]["net_time"] = merged_time
        sm.sync(sess)

    event_bus = getattr(request.app.state, "event_bus", None)
    await _remove_user_from_session(request, user_id, web_doc, event_bus)

    await request.app.db["users"].update_one(
        {"_id": user_id},
        {"$set": {"current_session": target_session_id}},
    )

    if sm:
        sm.user_sessions[user_id] = target_session_id

    return {"ok": 1, "merged_time": merged_time}


@router.get("/{session_id}")
@limiter.limit("60/minute", key_func=rate_limit_ip)
@limiter.limit("120/hour", key_func=rate_limit_user)
async def get_session_state(
    request: Request,
    session_id: str,
    payload: dict = Depends(verify_token),
):
    user_id = payload.get("sub")

    session_doc = await request.app.db["sessions"].find_one({"session_id": session_id})
    if not session_doc:
        raise HTTPException(status_code=404, detail="Session not found")

    members_raw = session_doc.get("members", {})
    is_member = user_id in members_raw
    guild_id = session_doc.get("guild_id", "")

    if not is_member and guild_id != "web":
        in_guild = payload.get("in_guild", False)
        if not await _is_live_guild_member(request, guild_id, user_id, bool(in_guild)):
            raise HTTPException(status_code=403, detail="You must be in the guild to view this session")

    members = await _enrich_members(request, session_doc)
    return _session_response(session_doc, members)


@router.post("/{session_id}/join")
@limiter.limit("20/minute", key_func=rate_limit_ip)
@limiter.limit("40/hour", key_func=rate_limit_user)
async def join_session(
    request: Request,
    session_id: str,
    body: JoinRequest = JoinRequest(),
    payload: dict = Depends(verify_token),
):
    user_id = payload.get("sub")

    existing_session = await request.app.db["sessions"].find_one(
        {"members": {f"$exists": True}, f"members.{user_id}": {"$exists": True}},
    )
    if existing_session:
        existing_sid = existing_session.get("session_id")
        if existing_sid != session_id:
            await _remove_user_from_session(request, user_id, existing_session, getattr(request.app.state, "event_bus", None))

    session_doc = await request.app.db["sessions"].find_one({"session_id": session_id})
    if not session_doc:
        raise HTTPException(status_code=404, detail="Session not found. Make sure someone is in the voice channel.")

    members_raw = session_doc.get("members", {})
    if user_id in members_raw:
        members = await _enrich_members(request, session_doc)
        return _session_response(session_doc, members)

    guild_id = session_doc.get("guild_id", "")
    if guild_id != "web":
        if not await _is_live_guild_member(request, guild_id, user_id, bool(payload.get("in_guild", False))):
            raise HTTPException(status_code=403, detail="You must be in the Discord guild to join this session")

    now = datetime.now(timezone.utc)

    initial_time = _normalize_initial_time(body.initial_time)

    await request.app.db["sessions"].update_one(
        {"session_id": session_id},
        {
            "$set": {
                f"members.{user_id}": {
                    "net_time": initial_time,
                    "last_activity": "noact",
                    "_seg": now.isoformat(),
                    "is_web_user": True,
                },
            },
            "$inc": {"members_count.total": 1, "members_count.noact": 1},
        },
    )

    await request.app.db["users"].update_one(
        {"_id": user_id},
        {"$set": {"current_session": session_id}},
    )

    user_data = await request.app.db["users"].find_one(
        {"_id": user_id},
        {"name": 1, "display_name": 1, "pfp": 1, "profile_pfp": 1},
    )
    username = "Unknown"
    display_name = "Unknown"
    avatar_url = default_avatar(user_id)
    if user_data:
        username = user_data.get("name", "Unknown")
        display_name = user_data.get("display_name") or username
        avatar_url = resolve_avatar_url(user_id, user_data)

    event_bus = getattr(request.app.state, "event_bus", None)
    if event_bus:
        await event_bus.publish(session_id, "member_join", {
            "user_id": user_id,
            "username": username,
            "display_name": display_name,
            "avatar_url": avatar_url,
            "activity": "noact",
            "is_web_user": True,
        })

    sm = getattr(getattr(request.app.state, "bot", None), "session_manager", None)
    if sm and session_id in sm.active_sessions:
        sess = sm.active_sessions[session_id]
        sess.members[user_id] = {
            "net_time": initial_time,
            "last_activity": "noact",
            "_seg": now.isoformat(),
            "is_web_user": True,
        }
        sess._update_members_count()
        sm.user_sessions[user_id] = session_id
        await request.app.db["sessions"].update_one(
            {"session_id": session_id},
            {"$set": {"members_count": sess.members_count}},
        )

    updated_doc = await request.app.db["sessions"].find_one({"session_id": session_id})
    members = await _enrich_members(request, updated_doc)
    return _session_response(updated_doc, members)


@router.post("/{session_id}/leave")
@limiter.limit("30/minute", key_func=rate_limit_ip)
@limiter.limit("60/hour", key_func=rate_limit_user)
async def leave_session(
    request: Request,
    session_id: str,
    payload: dict = Depends(verify_token),
):
    user_id = payload.get("sub")

    session_doc = await request.app.db["sessions"].find_one({"session_id": session_id})
    if not session_doc:
        raise HTTPException(status_code=404, detail="Session not found")

    members_raw = session_doc.get("members", {})
    if user_id not in members_raw:
        raise HTTPException(status_code=400, detail="You are not in this session")

    member_data = members_raw[user_id]
    is_web_user = isinstance(member_data, dict) and member_data.get("is_web_user", False)

    if not is_web_user:
        bot = getattr(request.app.state, "bot", None)
        guild_id = session_doc.get("guild_id")
        channel_id = session_doc.get("channel_id")
        if bot and guild_id and not channel_id.startswith("w"):
            try:
                guild = bot.get_guild(int(guild_id))
                if guild:
                    member = guild.get_member(int(user_id))
                    if member and member.voice:
                        await member.move_to(None)
            except Exception:
                pass

    event_bus = getattr(request.app.state, "event_bus", None)
    await _remove_user_from_session(request, user_id, session_doc, event_bus)
    return {"ok": 1}


@router.post("/create")
@limiter.limit("10/minute", key_func=rate_limit_ip)
@limiter.limit("20/hour", key_func=rate_limit_user)
async def create_web_session(
    request: Request,
    payload: dict = Depends(verify_token),
):
    user_id = payload.get("sub")

    existing = await request.app.db["sessions"].find_one(
        {"members": {f"$exists": True}, f"members.{user_id}": {"$exists": True}},
    )
    if existing:
        existing_sid = existing.get("session_id") or existing.get("_id")
        guild_id = existing.get("guild_id", "web")
        channel_id = existing.get("channel_id", "")
        members_raw = existing.get("members", {})

        # Never destroy this user's active session just because they asked for
        # a fresh web session. A solo Discord VC session is a real, live session
        # the user is sitting in — wiping it removes them from the DB, emits
        # session_closed (the frontend shows "session ended"), while the bot
        # never actually moves them out of the VC. Only clean up an orphaned
        # web-only session, and otherwise just reuse the existing session.
        is_live_discord = guild_id != "web" and bool(channel_id) and not channel_id.startswith("w")
        if not is_live_discord and len(members_raw) == 1:
            sm = getattr(getattr(request.app.state, "bot", None), "session_manager", None)
            if sm and existing_sid in sm.active_sessions:
                sm._cleanup_session(sm.active_sessions[existing_sid])
            else:
                await request.app.db["users"].update_one(
                    {"_id": user_id},
                    {"$unset": {"current_session": "", "webToken": ""}},
                )
                await request.app.db["sessions"].delete_one({"session_id": existing_sid})
                if sm:
                    for uid, sid in list(sm.user_sessions.items()):
                        if sid == existing_sid:
                            del sm.user_sessions[uid]
                    for ch, sid in list(sm.channel_sessions.items()):
                        if sid == existing_sid:
                            del sm.channel_sessions[ch]
            event_bus = getattr(request.app.state, "event_bus", None)
            if event_bus:
                await event_bus.publish(existing_sid, "session_closed", {})
        else:
            # User already belongs to a live session (Discord VC or multi-member).
            # Return it instead of destroying it / forcing a duplicate web session.
            members = await _enrich_members(request, existing)
            return _session_response(existing, members)

    from library.dseshpy.session import generate_session_id
    channel_id = f"w{user_id}"
    session_id = generate_session_id(channel_id)
    now = datetime.now(timezone.utc)

    session_doc = {
        "session_id": session_id,
        "owner_id": user_id,
        "guild_id": "web",
        "channel_id": channel_id,
        "members": {
            user_id: {
                "net_time": {"cam": 0, "ss": 0, "noact": 0, "total": 0},
                "last_activity": "noact",
                "_seg": now.isoformat(),
                "is_web_user": True,
            }
        },
        "members_count": {"total": 1, "noact": 1, "ss": 0, "cam": 0},
        "vc_level": 1,
        "vc_xp": 0,
        "session_type": "*",
    }

    await request.app.db["sessions"].insert_one(session_doc)
    await request.app.db["users"].update_one(
        {"_id": user_id},
        {"$set": {"current_session": session_id}},
    )

    import config as _cfg
    sm = getattr(getattr(request.app.state, "bot", None), "session_manager", None)
    if sm:
        from library.dseshpy.session import Session
        sess_obj = Session(
            session_id=session_id,
            owner_id=user_id,
            guild_id="web",
            channel_id=channel_id,
            members={
                user_id: {
                    "net_time": {"cam": 0, "ss": 0, "noact": 0, "total": 0},
                    "last_activity": "noact",
                    "_seg": now.isoformat(),
                    "is_web_user": True,
                }
            },
            members_count={"total": 1, "noact": 1, "ss": 0, "cam": 0},
            vc_level=1,
            vc_xp=0,
            routine_callback_mean_time=int(_cfg.DROP_MEAN_TIME),
            session_type="*",
        )
        sess_obj.event_bus = getattr(request.app.state, "event_bus", None)
        sm.active_sessions[session_id] = sess_obj
        sm.channel_sessions[channel_id] = session_id
        sm.user_sessions[user_id] = session_id
        import asyncio as _asyncio
        sess_obj.drop_task = _asyncio.create_task(sess_obj.drop_routine(None))
        logger.info("Web session registered in SessionManager")
    else:
        logger.warning("SessionManager not found — drops will not fire")

    user_data = await request.app.db["users"].find_one(
        {"_id": user_id},
        {"name": 1, "display_name": 1, "pfp": 1, "profile_pfp": 1},
    )
    username = "Unknown"
    display_name = "Unknown"
    avatar_url = default_avatar(user_id)
    if user_data:
        username = user_data.get("name", "Unknown")
        display_name = user_data.get("display_name") or username
        avatar_url = resolve_avatar_url(user_id, user_data)

    members = await _enrich_members(request, session_doc)
    return _session_response(session_doc, members)


@router.get("/{session_id}/stream")
@limiter.limit("30/minute", key_func=rate_limit_ip)
@limiter.limit("60/hour", key_func=rate_limit_user)
async def stream_session_events(
    request: Request,
    session_id: str,
    payload: dict = Depends(verify_token),
):
    client_ip = _client_ip(request)
    _sse_by_ip[client_ip] += 1
    _sse_by_session[session_id] += 1

    def _release_slot():
        _sse_by_ip[client_ip] -= 1
        _sse_by_session[session_id] -= 1

    if _sse_by_ip[client_ip] > MAX_SSE_PER_IP or _sse_by_session[session_id] > MAX_SSE_PER_SESSION:
        _release_slot()
        raise HTTPException(status_code=429, detail="Too many active stream connections")

    event_bus = getattr(request.app.state, "event_bus", None)
    if not event_bus:
        _release_slot()
        raise HTTPException(status_code=503, detail="Event bus not available")

    sm = getattr(getattr(request.app.state, "bot", None), "session_manager", None)
    if sm and session_id not in sm.active_sessions:
        session_doc = await request.app.db["sessions"].find_one({"session_id": session_id})
        if session_doc:
            from library.dseshpy.session import Session
            sess_obj = Session.from_dict(session_doc)
            sess_obj.event_bus = event_bus
            sm.active_sessions[session_id] = sess_obj
            sm.channel_sessions[sess_obj.channel_id] = session_id
            for uid in sess_obj.members:
                sm.user_sessions[uid] = session_id
            sess_obj.drop_task = asyncio.create_task(sess_obj.drop_routine(None))
            logger.info("SSE reconnect: re-created session in SessionManager")

    try:
        queue = await event_bus.subscribe(session_id)
    except Exception:
        _release_slot()
        raise

    async def event_generator():
        try:
            yield f"data: {json.dumps({'event': 'connected', 'data': {'session_id': session_id}})}\n\n"

            while True:
                if await request.is_disconnected():
                    break

                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {message}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'event': 'heartbeat', 'data': {}})}\n\n"

        except asyncio.CancelledError:
            pass
        finally:
            await event_bus.unsubscribe(session_id, queue)
            _sse_by_ip[client_ip] -= 1
            _sse_by_session[session_id] -= 1

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{session_id}/level-up/pay")
@limiter.limit("30/minute", key_func=rate_limit_ip)
@limiter.limit("60/hour", key_func=rate_limit_user)
async def pay_level_up(
    request: Request,
    session_id: str,
    payload: dict = Depends(verify_token),
):
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    bot = getattr(request.app.state, "bot", None)
    sm = getattr(bot, "session_manager", None) if bot else None
    sess = sm.active_sessions.get(session_id) if sm else None

    if not sess:
        doc = await request.app.db["sessions"].find_one({"session_id": session_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Session not found")
        from library.dseshpy.session import Session
        event_bus = getattr(request.app.state, "event_bus", None)
        sess = Session.from_dict(doc)
        if event_bus:
            sess.event_bus = event_bus
        if sm:
            sm.active_sessions[session_id] = sess
            sm.channel_sessions[sess.channel_id] = session_id
            for uid in sess.members:
                sm.user_sessions[uid] = session_id

    if not sess.pending_level_up:
        raise HTTPException(status_code=400, detail="No pending level-up")

    if user_id not in sess.members:
        raise HTTPException(status_code=403, detail="You must be a session member to pay")

    new_level = sess.pending_level_up["new_level"]
    wood_cost = sess.pending_level_up["wood_cost"]

    from library import degrade
    user_data = await request.app.db["users"].find_one({"_id": user_id})
    resources = (user_data or {}).get("economy", {}).get("resources", {})
    wood_data = resources.get("wood", {})

    rates_doc = await request.app.db["config"].find_one({"_id": "degradation_rates"})
    wood_rate = rates_doc.get("wood", 0.05) if rates_doc else 0.05

    raw_amount = wood_data.get("amount", 0)
    degraded_at = wood_data.get("degraded_at")
    existing_wood, wood_dt = degrade.apply(raw_amount, degraded_at, wood_rate)

    if existing_wood < wood_cost:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient wood ({existing_wood}/{wood_cost})",
        )

    result = await request.app.db["users"].find_one_and_update(
        {
            "_id": user_id,
            "economy.resources.wood.amount": raw_amount,
        },
        {"$set": {
            "economy.resources.wood.amount": existing_wood - wood_cost,
            "economy.resources.wood.degraded_at": wood_dt,
        }},
    )

    if not result:
        raise HTTPException(status_code=409, detail="Concurrent modification, please retry")

    sess.vc_level = new_level
    sess.vc_xp = 0
    from datetime import datetime as _dt, timezone as _tz
    sess.last_level_up_at = _dt.now(_tz.utc).isoformat()
    sess.pending_level_up = None
    sess._sync_session_now()

    if sess.level_up_message_id and sess.guild_id != "web" and bot:
        try:
            import discord as _discord
            channel = bot.get_channel(int(sess.channel_id))
            if channel:
                msg = await channel.fetch_message(int(sess.level_up_message_id))
                await msg.delete()
        except Exception:
            pass

        try:
            import discord as _discord
            channel = bot.get_channel(int(sess.channel_id))
            if channel:
                celeb_embed = _discord.Embed(
                    title="\U0001f525 Level Up!",
                    description=f"Level **{new_level}** achieved!\n\nPaid by <@{user_id}>",
                    color=_discord.Color.gold(),
                )
                await channel.send(embed=celeb_embed, delete_after=30)
        except Exception:
            pass

        sess.level_up_message_id = None
        sess._sync_session_now()

    await sess._emit_event("level_up_complete", {
        "new_level": new_level,
        "wood_cost": wood_cost,
    })

    return {
        "ok": 1,
        "new_level": new_level,
        "wood_cost": wood_cost,
    }




@router.post("/{session_id}/boostxp")
@limiter.limit("6/minute", key_func=rate_limit_ip)
@limiter.limit("12/hour", key_func=rate_limit_user)
async def boostxp_session(
    request: Request,
    session_id: str,
    payload: dict = Depends(verify_token),
):
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    now = time.time()

    bot = getattr(request.app.state, "bot", None)
    sm = getattr(bot, "session_manager", None) if bot else None
    sess = sm.active_sessions.get(session_id) if sm else None

    if not sess:
        doc = await request.app.db["sessions"].find_one({"session_id": session_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Session not found")
        from library.dseshpy.session import Session
        event_bus = getattr(request.app.state, "event_bus", None)
        sess = Session.from_dict(doc)
        if event_bus:
            sess.event_bus = event_bus
        if sm:
            sm.active_sessions[session_id] = sess
            sm.channel_sessions[sess.channel_id] = session_id
            for uid in sess.members:
                sm.user_sessions[uid] = session_id

    if user_id not in sess.members:
        raise HTTPException(status_code=403, detail="You must be a session member to boost")

    if sess.pending_level_up:
        raise HTTPException(status_code=400, detail="Level-up pending — pay wood first")

    last_boost = getattr(sess, 'last_boost_at', None)
    if last_boost:
        last_ts = datetime.fromisoformat(last_boost).timestamp() if isinstance(last_boost, str) else float(last_boost)
    else:
        if sess.started_at:
            last_ts = datetime.fromisoformat(sess.started_at).timestamp() if isinstance(sess.started_at, str) else float(sess.started_at)
        else:
            last_ts = now

    elapsed_sec = now - last_ts
    elapsed_min = elapsed_sec / 60.0
    xp_gain = max(1, int(elapsed_min * app_config.LEVEL_UP_XP_PER_MINUTE))

    new_xp = sess.vc_xp + xp_gain
    new_level = sess.vc_level
    level_up = False
    pending_level_up_data = None

    if new_xp >= app_config.LEVEL_UP_XP_THRESHOLD:
        new_level = sess.vc_level + 1
        wood_cost = app_config.LEVEL_UP_WOOD_BASE * new_level
        pending_level_up_data = {
            "new_level": new_level,
            "wood_cost": wood_cost,
        }
        sess.pending_level_up = pending_level_up_data
        level_up = True

    sess.vc_xp = new_xp
    sess.vc_level = new_level
    sess.last_boost_at = datetime.now(timezone.utc).isoformat()
    sess._sync_session_now()

    await sess._emit_event("boostxp", {
        "user_id": user_id,
        "xp_gained": xp_gain,
        "new_xp": new_xp,
        "vc_level": new_level,
        "level_up": level_up,
    })

    if level_up and pending_level_up_data:
        await sess._emit_event("level_up", {
            "new_level": new_level,
            "wood_cost": pending_level_up_data["wood_cost"],
        })

        if sess.guild_id != "web" and bot:
            try:
                import discord as _discord
                channel = bot.get_channel(int(sess.channel_id))
                if channel:
                    domain = os.getenv("FRONTEND_DOMAIN", "")
                    if not domain.endswith("/"):
                        domain += "/"
                    link = f"{domain}projects?level_up={session_id}"
                    embed = _discord.Embed(
                        title="\u2b06\ufe0f Level Up Available!",
                        description=f"Level **{sess.vc_level - 1}** \u2192 **{new_level}**\nCost: **{pending_level_up_data['wood_cost']}** \U0001fab5 Wood",
                        color=_discord.Color.green(),
                    )
                    view = _discord.ui.View()
                    view.add_item(_discord.ui.Button(
                        label=f"Pay {pending_level_up_data['wood_cost']} Wood",
                        style=_discord.ButtonStyle.link,
                        url=link,
                        emoji="\U0001fab5",
                    ))
                    msg = await channel.send(embed=embed, view=view)
                    sess.level_up_message_id = str(msg.id)
                    sess._sync_session_now()
            except Exception:
                pass

    return {
        "ok": 1,
        "xp_gained": xp_gain,
        "new_xp": sess.vc_xp,
        "vc_level": sess.vc_level,
        "level_up": level_up,
    }
