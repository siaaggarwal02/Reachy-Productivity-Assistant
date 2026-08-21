"""
Reachy Mini - Asynchronous Personal Assistant + Behavior Watch
==============================================================================

Full architectural rewrite of the original combined passive-listening /
passive-vision script. Everything that talks to hardware, a network, or a
model now runs on a non-blocking asyncio event loop, with genuinely blocking
work (microphone capture, TTS synthesis + playback, motor moves, local model
inference, Google/weather HTTP calls) pushed onto a bounded thread pool via
`loop.run_in_executor(...)`. Nothing in the event loop itself ever blocks.

Four modular pieces, per the spec:

  1. ReachyHardwareManager  - all direct hardware I/O: camera frames (with
     downsampling + rate limiting), microphone capture (the system default
     input device - deliberately simple, see the comment on
     open_microphone() for why), TTS synthesis/playback, and motor/antenna
     gestures. Every hardware call is wrapped so a single bad frame or a
     stalled peripheral can never take down the process.

  2. VisionBehaviorTracker - a thread-safe state machine for hydration,
     work-focus, and phone-distraction tracking. Debounced: a held cup
     can't be miscounted as multiple drinks (15s cooldown between counted
     events), a work streak only praises once per continuous streak
     (60s+), and a phone only triggers after 3+ continuous seconds in
     frame (not a single flickering frame).

  3. AssistantServices - Claude conversation (web-search enabled), Google
     Calendar, Google Tasks, and weather (Open-Meteo), plus the "good
     morning" briefing aggregator.

  4. MainExecutionLoop - the asyncio event bus. Spins up the vision loop,
     the audio/wake-word loop, and dispatches respond-to-request / morning
     briefing work as background tasks so none of them ever stall the
     others.

Setup:
    pip install sounddevice scipy SpeechRecognition anthropic pyttsx3 \
                pillow requests \
                google-api-python-client google-auth-httplib2 google-auth-oauthlib

Before running:
    $env:ANTHROPIC_API_KEY = "your-key-here"
    Place Google OAuth `credentials.json` (Calendar + Tasks scopes) next to
    this script. First run opens a browser to authorize; a `token.json` is
    cached afterward.

Requires the Reachy Mini daemon already running in another window.

Run:
    python reachy_assistant.py
    python reachy_assistant.py --location "Santa Clara, CA" --interval 1.5
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import difflib
import io
import json
import os
import random
import signal
import socket
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone, tzinfo
from enum import Enum, auto
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pyttsx3
import requests
import speech_recognition as sr
from anthropic import Anthropic
from PIL import Image
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from scipy.signal import resample

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    _GOOGLE_LIBS_AVAILABLE = True
except ImportError:
    _GOOGLE_LIBS_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog
    _TKINTER_AVAILABLE = True
except ImportError:
    _TKINTER_AVAILABLE = False


# =============================================================================
# Shared configuration
# =============================================================================

SPEECH_OUTPUT_WAV = "claude_reply.wav"
EXECUTOR_MAX_WORKERS = 8

# BUG FIX: a global fallback network timeout. Without this, a single stalled
# network call (to Claude, Google Calendar, Google Tasks, etc. - any of
# which could hang indefinitely under poor network conditions with no
# application-level timeout of their own) could block a background task
# forever, which is exactly what was causing shutdown to hang "most of the
# time" - waiting on a backend thread that could never actually finish.
# This bounds every socket-level call in the process, not just some of them.
socket.setdefaulttimeout(30.0)

# --- Voice / wake word ---
WAKE_HEY_WORDS = {"hey", "hay", "hi", "a"}
WAKE_NAME_ALIASES = {"reachy", "richie", "ricci", "reechy", "reachie", "richy", "reeky"}
WAKE_SIMILARITY_THRESHOLD = 0.55
GOOD_MORNING_SIMILARITY_THRESHOLD = 0.72

DEFAULT_SILENCE_SECONDS = 3.5
LISTEN_TIMEOUT_SECONDS = 8
# BUG FIX: previously None (unbounded) - a single continuous utterance with
# no long-enough pause could block this call indefinitely, which is exactly
# why shutdown could stall for as long as someone kept talking (this
# blocking call runs on its own thread and can't check the shutdown signal
# until it returns). 25 seconds is generous for any real voice command
# while still guaranteeing a hard upper bound on how long a single listen
# can ever run.
MAX_PHRASE_SECONDS = 25
ENERGY_THRESHOLD = 300
FOLLOW_UP_WINDOW_SECONDS = 15.0

CLAUDE_MODEL = "claude-sonnet-5"
CLAUDE_MAX_TOKENS = 500
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}

# --- Client-executed tools (new): calendar/task actions Claude can request.
# Unlike WEB_SEARCH_TOOL (a server-side Anthropic tool that executes
# automatically), these are custom tools: when Claude decides to call one,
# the API response's stop_reason is "tool_use" and OUR code must actually
# run it and send the result back before Claude can finish its reply. See
# ConversationManager.blocking_ask for that loop.
LIST_CALENDAR_EVENTS_TOOL = {
    "name": "list_calendar_events",
    "description": (
        "Reads events already on the user's Google Calendar. Use this whenever the user asks "
        "what's on their calendar, what their schedule looks like, whether they're free/busy, "
        "or about any existing event - not just during a morning briefing. This is a READ-only "
        "lookup; use add_calendar_event separately if they want something created."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": (
                    "ISO 8601 date, e.g. '2026-08-12', for the day to check. Omit this field "
                    "entirely to check today. Only a single day is supported per call - for a "
                    "date range, call this tool once per day."
                ),
            },
        },
        "required": [],
    },
}

ADD_CALENDAR_EVENT_TOOL = {
    "name": "add_calendar_event",
    "description": (
        "Creates a new event on the user's Google Calendar. Use this when the user asks to "
        "schedule, add, book, or create a calendar event."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Event title/summary."},
            "start_datetime": {
                "type": "string",
                "description": (
                    "Event start time as an ISO 8601 datetime, e.g. '2026-08-12T15:00:00'. "
                    "Assume the user's local timezone (do not add a UTC offset yourself)."
                ),
            },
            "duration_minutes": {
                "type": "integer",
                "description": "Event duration in minutes. Default to 60 if the user doesn't specify.",
            },
            "location": {
                "type": "string",
                "description": "Event location, if the user mentions one. Omit this field entirely if not mentioned.",
            },
        },
        "required": ["title", "start_datetime"],
    },
}

DELETE_CALENDAR_EVENT_TOOL = {
    "name": "delete_calendar_event",
    "description": (
        "Deletes an existing event from the user's Google Calendar. Use this when the user asks "
        "to cancel, remove, or delete a calendar event."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title_query": {
                "type": "string",
                "description": "The event's title, or a close description of it, used to find the matching event.",
            },
            "date_hint": {
                "type": "string",
                "description": (
                    "ISO 8601 date, e.g. '2026-08-12', if the user specifies which day the event "
                    "is on - helps disambiguate if multiple similarly-named events exist. Omit if "
                    "not specified (searches roughly the last week through the next two months)."
                ),
            },
        },
        "required": ["title_query"],
    },
}

EDIT_CALENDAR_EVENT_TOOL = {
    "name": "edit_calendar_event",
    "description": (
        "Modifies an existing event on the user's Google Calendar - rename it, reschedule it, "
        "change its duration, or change its location. Use this when the user asks to move, "
        "reschedule, rename, or change an existing event. Only include the fields the user "
        "actually wants changed; omit the rest."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title_query": {
                "type": "string",
                "description": "The event's CURRENT title, or a close description, used to find the matching event.",
            },
            "date_hint": {
                "type": "string",
                "description": "ISO 8601 date of the event's current day, to help disambiguate if needed. Omit if not needed.",
            },
            "new_title": {
                "type": "string",
                "description": "New title for the event. Omit this field entirely if the user isn't renaming it.",
            },
            "new_start_datetime": {
                "type": "string",
                "description": (
                    "New start time as an ISO 8601 datetime, e.g. '2026-08-12T15:00:00', in the "
                    "user's local timezone. Omit this field entirely if the user isn't rescheduling it."
                ),
            },
            "new_duration_minutes": {
                "type": "integer",
                "description": "New duration in minutes. Omit this field entirely if the user isn't changing the length.",
            },
            "new_location": {
                "type": "string",
                "description": "New location. Omit this field entirely if the user isn't changing it.",
            },
        },
        "required": ["title_query"],
    },
}

ADD_TASK_TOOL = {
    "name": "add_task",
    "description": (
        "Creates a new task on the user's Google Tasks list. Use this when the user asks to "
        "add a task, to-do, or reminder item (that is NOT tied to a specific calendar time)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Task title."},
            "due_date": {
                "type": "string",
                "description": (
                    "Due date as an ISO 8601 date, e.g. '2026-08-12'. Omit if the user doesn't "
                    "specify one - it will automatically default to today's date."
                ),
            },
        },
        "required": ["title"],
    },
}

COMPLETE_TASK_TOOL = {
    "name": "complete_task",
    "description": (
        "Marks an existing Google Task as completed. Use this when the user says they finished, "
        "completed, or want to check off a task."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title_query": {
                "type": "string",
                "description": "The task's title, or a close description of it, used to find the matching task.",
            },
        },
        "required": ["title_query"],
    },
}

LIST_TASKS_TOOL = {
    "name": "list_tasks",
    "description": (
        "Retrieves ALL of the user's Google Tasks (to-do items) across every task list, so you "
        "can tell them everything on their list at once. Use this whenever the user asks what "
        "tasks/to-dos they have, what's on their task list, or anything similar - not just when "
        "asking about one specific task (use complete_task for that). Do NOT guess or make up tasks."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "include_completed": {
                "type": "boolean",
                "description": (
                    "Whether to include already-completed tasks in the list. Default to false "
                    "(pending/incomplete tasks only) unless the user specifically asks about "
                    "completed or finished tasks too."
                ),
            },
        },
        "required": [],
    },
}

DELETE_TASK_TOOL = {
    "name": "delete_task",
    "description": (
        "Permanently deletes an existing Google Task. Use this when the user asks to delete, "
        "remove, or get rid of a task entirely - this is DIFFERENT from marking it done; use "
        "complete_task instead if they just finished it and want it checked off."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title_query": {
                "type": "string",
                "description": "The task's title, or a close description of it, used to find the matching task.",
            },
        },
        "required": ["title_query"],
    },
}

EDIT_TASK_TOOL = {
    "name": "edit_task",
    "description": (
        "Modifies an existing Google Task - rename it or change its due date. Use this when the "
        "user asks to reschedule, rename, or change when a task is due. IMPORTANT: Google Tasks "
        "only supports a due DATE (a day), not a specific time of day - if the user mentions a "
        "time, just use the date they mean and ignore the time portion; do not claim you set a "
        "specific time. Only include the fields the user actually wants changed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title_query": {
                "type": "string",
                "description": "The task's CURRENT title, or a close description, used to find the matching task.",
            },
            "new_title": {
                "type": "string",
                "description": "New title for the task. Omit this field entirely if the user isn't renaming it.",
            },
            "new_due_date": {
                "type": "string",
                "description": (
                    "New due date as an ISO 8601 date, e.g. '2026-08-15'. Omit this field entirely "
                    "if the user isn't changing the due date."
                ),
            },
            "clear_due_date": {
                "type": "boolean",
                "description": (
                    "Set true only if the user wants to REMOVE the due date entirely, rather than "
                    "changing it to a new one. Omit or false otherwise."
                ),
            },
        },
        "required": ["title_query"],
    },
}

RECORD_VISION_FEEDBACK_TOOL = {
    "name": "record_vision_feedback",
    "description": (
        "Records the user's feedback on Reachy's most recent camera observation. Use this "
        "whenever the user confirms an observation was accurate ('that's right', 'yes that's "
        "correct') or corrects one that was wrong ('no, I'm actually eating a sandwich', "
        "'that's wrong, I wasn't on my phone'). This is how Reachy's vision gets calibrated over "
        "time from real feedback - always use this tool when the user is clearly commenting on "
        "whether a recent description of what they were doing was accurate, rather than just "
        "replying conversationally."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "was_correct": {
                "type": "boolean",
                "description": "True if the user confirmed the observation was accurate; false if they said it was wrong.",
            },
            "corrected_description": {
                "type": "string",
                "description": (
                    "If was_correct is false, a short factual description of what was ACTUALLY "
                    "happening, based on what the user said. Omit this field entirely if "
                    "was_correct is true."
                ),
            },
            "corrected_phone_visible": {
                "type": "boolean",
                "description": (
                    "If the user's correction clearly implies whether they were or weren't "
                    "actively using a phone, set this. Omit this field entirely if not implied."
                ),
            },
            "corrected_drinking": {
                "type": "boolean",
                "description": (
                    "If the user's correction clearly implies whether they were or weren't "
                    "drinking. Omit this field entirely if not implied."
                ),
            },
            "corrected_at_desk_working": {
                "type": "boolean",
                "description": (
                    "If the user's correction clearly implies whether they were or weren't "
                    "working at a desk. Omit this field entirely if not implied."
                ),
            },
        },
        "required": ["was_correct"],
    },
}

INCOMPLETE_TRAILING_WORDS = {
    "and", "or", "but", "so", "because", "the", "a", "an", "to", "of", "in",
    "is", "are", "was", "plus", "minus", "times", "what's", "whats", "how",
    "why", "when", "will", "would", "should", "can", "could",
    "+", "-", "*", "/",
}

# --- Vision watch ---
# Vision runs on Claude's own multimodal API rather than a small local
# captioning model. A local model like BLIP-base is CPU-bound autoregressive
# generation - on hardware without a GPU that routinely takes 10-20+ seconds
# per frame, which is what was actually producing the multi-second gaps
# between cycles, and it's also just not a strong enough model to reliably
# tell "two people at a table" apart from "two people playing a video game."
# A network call to Claude trades local compute for a round trip, but in
# practice that round trip is both faster and dramatically more accurate on
# CPU-only hardware. Haiku is used here (not the conversation's Sonnet
# model) since this fires on a short interval and Haiku is both cheaper and
# fast enough for a per-frame classification task.
CLAUDE_VISION_MODEL = "claude-haiku-4-5-20251001"
VISION_MAX_TOKENS = 200

# BUG FIX: keyword sets used to cross-validate the structured booleans
# against the free-text activity_summary from the SAME model response -
# these two signals can be internally inconsistent (the model describing
# something in text while setting a contradicting boolean). See
# ClaudeVisionDescriber._reconcile_observation for the directed correction
# logic this backs.
PHONE_KEYWORDS_IN_TEXT = {"phone", "cell", "cellphone", "smartphone", "texting", "calling", "scrolling"}
# BUG FIX: this was previously a set of plain NOUNS ("bottle", "cup",
# "water", ...), which meant a caption like "a person typing at a desk
# with a water bottle nearby" - merely describing an unused object in the
# scene - would match and force drinking=True, logging a false hydration
# count. Now requires an ACTUAL drinking verb/phrase describing active
# consumption, not just the presence of a container noun anywhere in the
# text - this is what "on the table but not being drunk from" needed.
DRINK_ACTIVE_PHRASES_IN_TEXT = (
    "drinking",
    "sipping",
    "taking a sip",
    "taking a drink",
    "takes a sip",
    "takes a drink",
    "gulping",
    "drinking from",
    "sips from",
    "raises the bottle",
    "raises the cup",
    "raising the bottle",
    "raising the cup",
    "raising a bottle",
    "raising a cup",
)

# ACCURACY FIX (further increased): 85 -> 95. Near-maximum JPEG quality -
# preserves almost all fine detail at a still-small file size, directly
# addressing "heavily increase vision accuracy."
VISION_JPEG_QUALITY = 95
FRAME_CAPTURE_TIMEOUT_SECONDS = 20
DEFAULT_CAPTURE_INTERVAL_SECONDS = 3.0   # network-call cost/rate-limit tradeoff - see note in --interval help
CAPTION_CONFIRMATION_CYCLES = 2          # a described activity must repeat before it's trusted
# ACCURACY FIX (further increased): 768 -> 1024. Meaningfully sharper input
# for distinguishing small handheld objects (phone vs. food, bottle color/
# shape) at the cost of a modestly larger payload per vision cycle - a
# deliberate accuracy-over-latency tradeoff given the explicit priority here.
VISION_DOWNSAMPLE_MAX_DIMENSION = 1024   # longest edge, px, before it's sent to the API

TASK_SIMILARITY_THRESHOLD = 0.65
NO_PERSON_LABEL = "No person detected in frame"

# BUG FIX: kinder tone, replacing the harsh original wording. Used only for
# the FIRST alert in a new distraction period - see PHONE_ESCALATION_MESSAGES
# for the tone used on repeated detections within the same ongoing period.
PHONE_CALLOUT_MESSAGE = "Hey, just a gentle nudge - looks like it might be a good time to put the phone down for a bit."
# BUG FIX: bumped from 3.0 - continued false-positive reports suggested
# occasional multi-frame misclassification bursts (not just single-frame
# noise, which PHONE_STREAK_MERGE_SECONDS already handles) - requiring a
# longer sustained duration before ever alerting is extra defense-in-depth
# against nagging over something that isn't actually happening, at the
# small cost of slightly slower detection of genuine phone use.
PHONE_SUSTAINED_SECONDS = 6.0
# BUG FIX: added - a single misclassified frame within this window no
# longer resets an otherwise-continuous phone-use streak (see
# VisionBehaviorTracker.observe_phone for the full explanation).
PHONE_STREAK_MERGE_SECONDS = 6.0
# NEW: how long someone needs to be phone-free before the NEXT detection is
# treated as the start of a brand new distraction period (tier 1, gentle,
# with a task suggestion) rather than a continuation of an ongoing one
# (tier 2+, firmer but still kind, no repeated task-list reading - this is
# what made repeated alerts feel naggy).
PHONE_DISTRACTION_PERIOD_RESET_SECONDS = 300.0  # 5 minutes
# NEW: tone escalates gently across repeated detections within the SAME
# distraction period - deliberately stays kind rather than becoming harsh,
# and does NOT re-read the task list every time (that repetition was the
# main source of annoyance, not the reminder itself).
PHONE_ESCALATION_MESSAGES = [
    "Hey, I notice you're back on your phone - let's try to stay focused a bit longer.",
    "Still on the phone - totally understandable, but let's get back to it when you can.",
    "Another phone check - you've got this, let's refocus.",
    "I see the phone again - a quick break is fine, just don't want to lose the thread.",
]

HYDRATION_MILESTONE_MESSAGE = "Good job staying hydrated!"
HYDRATION_MILESTONE_INTERVAL = 2  # fires on 2nd, 4th, 6th, 8th... counted drink
# BUG FIX (two-tier timing, replacing the old single HYDRATION_DEBOUNCE_
# SECONDS = 15.0): a brief drop in the 'drinking' reading within this many
# seconds of the last true reading is treated as the SAME instance, not a
# new one - handles ordinary vision noise mid-drink without double-counting.
DRINKING_SAME_INSTANCE_MERGE_SECONDS = 60.0
# Minimum gap required between the start of one COUNTED instance and the
# next before a new one can be counted at all.
HYDRATION_DEBOUNCE_SECONDS = 300.0  # 5 minutes

# BUG FIX: was WORK_MILESTONE_SECONDS = 60.0 (1 minute), fired only once
# ever per streak. Now fires repeatedly, once per completed interval.
WORK_MILESTONE_INTERVAL_SECONDS = 30 * 60.0  # 30 minutes
WORK_MILESTONE_MESSAGE = "Good job, keep up the good work."

DEFAULT_MOVE_DURATION_SECONDS = 0.8

# --- Weather ---
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# --- Google API scopes ---
# CHANGED (necessary for this feature): upgraded from read-only to
# read-write scopes. Creating events/tasks is structurally impossible under
# calendar.readonly / tasks.readonly - Google's API rejects write calls
# outright with insufficient-scope regardless of code correctness. This is
# additive in effect (everything previously readable is still readable);
# the one real consequence is that a pre-existing cached token.json was
# authorized under the old, narrower scopes and will need one fresh
# re-authorization (a browser prompt) the first time this runs.
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
]
# FIX: these must be resolved relative to the SCRIPT's own location, not
# whatever directory the shell happens to be "in" when python is launched.
# A bare relative path like "credentials.json" is looked up relative to the
# process's current working directory - which can silently differ from the
# script's folder depending on how it's launched (a different starting cwd
# in PowerShell, running via an IDE's own working directory setting, etc.).
# That mismatch is exactly what "the file is right there but it says not
# found" looks like. Anchoring to Path(__file__).resolve().parent makes the
# lookup work correctly regardless of the caller's current directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
GOOGLE_CREDENTIALS_PATH = str(_SCRIPT_DIR / "credentials.json")
GOOGLE_TOKEN_PATH = str(_SCRIPT_DIR / "token.json")

# --- Vision calibration (NEW): persisted user feedback, used to build
# few-shot examples injected into future vision prompts. See
# VisionCalibrationStore for the full explanation of what this can and
# cannot do - it does NOT retrain or fine-tune the underlying model.
VISION_CALIBRATION_PATH = _SCRIPT_DIR / "vision_calibration.json"

# --- Task rollover (new) ---
# Checked hourly - more than sufficient since Google Tasks only supports
# DAY-level due-date granularity anyway (see blocking_rollover_overdue_tasks).
DEFAULT_TASK_ROLLOVER_POLL_INTERVAL_SECONDS = 3600.0

# --- Calendar reminders (new) ---
DEFAULT_REMINDER_POLL_INTERVAL_SECONDS = 60.0
REMINDER_ADVANCE_WINDOW_MINUTES = 30.0
# BUG FIX: replaced the old "at start" (0-minute) tier with a proper
# 5-minutes-before tier, matching what was actually asked for (30-min AND
# 5-min-before, not 30-min AND at-start). This is also structurally more
# robust than the old start-time check: since it always fires BEFORE the
# event begins, it can never suffer the "event already ended and vanished
# from the query window" failure mode the old at-start check had.
REMINDER_FIVE_MINUTE_WINDOW_MINUTES = 5.0
# Small backward-looking buffer kept for general robustness against a poll
# landing exactly on a boundary - not load-bearing the way it was for the
# old at-start check, but cheap insurance.
REMINDER_LOOKBACK_BUFFER_MINUTES = 2.0
# Bounds memory growth for the fired-reminder tracking set over long-running
# sessions - entries for events long past are pruned rather than kept forever.
STALE_REMINDER_RETENTION_HOURS = 24.0

REMINDER_THIRTY_MINUTE_KEY = "30min"
REMINDER_FIVE_MINUTE_KEY = "5min"

# Several phrasings per situation, chosen at random each time, per the
# "be creative" request - avoids the same canned line every single time.
THIRTY_MINUTE_REMINDER_TEMPLATES_WITH_LOCATION = [
    "Heads up — you should start heading out for {title} soon, it's coming up in about half an hour.",
    "Thirty minutes until {title} at {location} — might be time to start getting ready.",
    "Just a heads up, {title} is in thirty minutes over at {location}. Probably a good time to head out.",
    "Half an hour until {title}. Since it's at {location}, you'll probably want to get moving soon.",
]
THIRTY_MINUTE_REMINDER_TEMPLATES_NO_LOCATION = [
    "Reminder: {title} starts in thirty minutes.",
    "You've got {title} in about half an hour — might want to start wrapping up what you're doing.",
    "Heads up, {title} is coming up in thirty minutes.",
    "Thirty minutes until {title}. Just a heads up.",
]
FIVE_MINUTE_REMINDER_TEMPLATES_WITH_LOCATION = [
    "Almost time — {title} starts in 5 minutes at {location}. Have fun!",
    "{title} is starting in five minutes over at {location} — better head that way.",
    "Five minutes until {title} at {location}!",
    "Quick heads up — {title} kicks off in 5 minutes at {location}.",
]
FIVE_MINUTE_REMINDER_TEMPLATES_NO_LOCATION = [
    "Almost time — {title} starts in 5 minutes.",
    "{title} is starting in five minutes.",
    "Five minutes until {title}!",
    "Quick heads up — {title} kicks off in 5 minutes.",
]


class ListenStatus(Enum):
    OK = auto()
    TIMEOUT = auto()
    UNCLEAR = auto()
    ERROR = auto()


def enable_windows_ansi_support() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        STD_OUTPUT_HANDLE = -11
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        current_mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(current_mode)):
            kernel32.SetConsoleMode(handle, current_mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except Exception:
        pass


def truncate_for_display(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


# =============================================================================
# 1. ReachyHardwareManager
#    Owns every direct hardware touch point: camera, microphone(s), speaker,
#    and motors. All blocking calls are exposed as coroutines that hand the
#    actual work to a shared ThreadPoolExecutor, so the asyncio event loop
#    is never stalled by a slow peripheral.
# =============================================================================

class ReachyHardwareManager:
    def __init__(
        self,
        mini: ReachyMini,
        loop: asyncio.AbstractEventLoop,
        executor: concurrent.futures.ThreadPoolExecutor,
        output_lock: threading.Lock,
    ) -> None:
        self._mini = mini
        self._loop = loop
        self._executor = executor
        self._output_lock = output_lock

        self._speaker_lock = threading.Lock()
        self._motion_lock = threading.Lock()

        self._microphone: Optional[sr.Microphone] = None
        self._recognizer = sr.Recognizer()
        self._recognizer.dynamic_energy_threshold = False
        self._recognizer.energy_threshold = ENERGY_THRESHOLD

    # ---- setup -------------------------------------------------------

    def configure_recognizer(self, silence_seconds: float) -> None:
        self._recognizer.pause_threshold = silence_seconds

    def open_microphone(self, device_index: Optional[int] = None) -> sr.Microphone:
        """Opens the system's default microphone - matching the approach
        that was actually working, rather than the auto-detection logic
        this had before. Explicitly hunting for a 'native mic array' by
        device name and opening it directly was very likely selecting a
        raw/multi-channel capture endpoint: audio with plenty of energy
        (so it crossed the threshold fine) but not a coherent mono voice
        waveform, which is exactly what 'heard something, couldn't make
        out words' every single time looks like. The plain default device
        lets the OS's own audio stack do the channel/format negotiation,
        which is what the working version relied on.

        device_index: optional explicit PyAudio device index, for the rare
        case where the OS default device isn't the right one. Defaults to
        None, i.e. whatever the system's default input device is."""
        candidate = sr.Microphone(device_index=device_index, sample_rate=16000)
        self._microphone = candidate
        label = "system default microphone" if device_index is None else f"microphone device #{device_index}"
        with self._output_lock:
            print(f"[audio] Using {label}. energy_threshold={self._recognizer.energy_threshold:.0f} (fixed).")
        return candidate

    # ---- audio in ------------------------------------------------------

    def blocking_listen_once(self, source) -> tuple[ListenStatus, Optional[str]]:
        """One listen-and-transcribe cycle. Deliberately simple - no
        threshold calibration, no phrase-length cap, no self-mute during
        TTS playback, no artifact filtering. This matches the version that
        was actually hearing wake words correctly."""
        try:
            audio = self._recognizer.listen(source, timeout=LISTEN_TIMEOUT_SECONDS, phrase_time_limit=MAX_PHRASE_SECONDS)
        except sr.WaitTimeoutError:
            return ListenStatus.TIMEOUT, None
        except (OSError, IOError) as exc:
            with self._output_lock:
                print(f"[audio] Microphone error: {exc}")
            return ListenStatus.ERROR, None

        try:
            transcript = self._recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            with self._output_lock:
                print("(heard sound, but couldn't make out words)")
            return ListenStatus.UNCLEAR, None
        except sr.RequestError as exc:
            with self._output_lock:
                print(f"[audio] Speech recognition service error: {exc}")
            return ListenStatus.ERROR, None

        with self._output_lock:
            print(f'[heard]: "{transcript}"')
        return ListenStatus.OK, transcript

    # ---- speech out ------------------------------------------------------

    async def speak(self, text: str) -> None:
        await self._loop.run_in_executor(self._executor, self._blocking_speak, text)

    def _blocking_speak(self, text: str) -> None:
        with self._speaker_lock:
            try:
                samples, sample_rate = self._synthesize(text)
                self._play(samples, sample_rate)
            except Exception as exc:
                with self._output_lock:
                    print(f"[speech] Failed to speak '{truncate_for_display(text, 40)}': {exc}")

    @staticmethod
    def _synthesize(text: str) -> tuple[np.ndarray, int]:
        engine = pyttsx3.init()
        engine.save_to_file(text, SPEECH_OUTPUT_WAV)
        engine.runAndWait()

        with wave.open(SPEECH_OUTPUT_WAV, "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            raw_bytes = wav_file.readframes(wav_file.getnframes())

        if sample_width != 2:
            raise ValueError(f"Expected 16-bit PCM audio, got sample width {sample_width}")

        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return samples, sample_rate

    def _play(self, samples: np.ndarray, source_rate: int) -> None:
        self._mini.media.start_playing()
        try:
            output_rate = self._mini.media.get_output_audio_samplerate()
            output_channels = self._mini.media.get_output_channels()

            if source_rate != output_rate:
                target_length = int(len(samples) * output_rate / source_rate)
                samples = resample(samples, target_length).astype(np.float32)

            if output_channels == 2:
                playback_audio = np.column_stack([samples, samples]).astype(np.float32)
            else:
                playback_audio = samples.reshape(-1, 1).astype(np.float32)

            self._mini.media.push_audio_sample(playback_audio)
            time.sleep(len(playback_audio) / output_rate + 0.5)
        finally:
            self._mini.media.stop_playing()

    # ---- vision in ------------------------------------------------------

    async def capture_downsampled_frame(self) -> np.ndarray:
        """Grabs the latest camera frame and downsamples it before it is
        ever handed to a vision model, which is most of the latency fix:
        a smaller tensor means faster inference on every single cycle."""
        frame = await self._loop.run_in_executor(self._executor, self._blocking_capture_frame)
        return self._downsample(frame)

    def _blocking_capture_frame(self) -> np.ndarray:
        start_time = time.monotonic()
        frame = self._mini.media.get_frame()
        while frame is None:
            if time.monotonic() - start_time > FRAME_CAPTURE_TIMEOUT_SECONDS:
                raise TimeoutError(f"Failed to grab a camera frame within {FRAME_CAPTURE_TIMEOUT_SECONDS}s.")
            time.sleep(0.2)
            frame = self._mini.media.get_frame()
        return frame

    @staticmethod
    def _downsample(frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        longest_edge = max(height, width)
        if longest_edge <= VISION_DOWNSAMPLE_MAX_DIMENSION:
            return frame_bgr
        scale = VISION_DOWNSAMPLE_MAX_DIMENSION / float(longest_edge)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = Image.fromarray(frame_bgr[:, :, ::-1])
        resized = image.resize(new_size, Image.BILINEAR)
        return np.array(resized)[:, :, ::-1]

    # ---- motion ------------------------------------------------------

    async def run_gesture(self, gesture_name: str) -> None:
        """Fires a gesture on the executor without ever awaiting its
        completion from the caller's perspective if the caller chooses to
        create_task() this coroutine - so a dance never freezes vision or
        voice loops."""
        gesture_fn = getattr(self, f"_gesture_{gesture_name}", None)
        if gesture_fn is None:
            raise ValueError(f"Unknown gesture: {gesture_name}")
        try:
            await self._loop.run_in_executor(self._executor, gesture_fn)
        except RuntimeError as exc:
            with self._output_lock:
                print(f"[motion] Gesture '{gesture_name}' failed: {exc}")

    def _goto(
        self,
        *,
        head: Optional[np.ndarray] = None,
        antennas: Optional[np.ndarray] = None,
        body_yaw: Optional[float] = None,
        duration: float = DEFAULT_MOVE_DURATION_SECONDS,
    ) -> None:
        normalized_body_yaw = float(body_yaw) if body_yaw is not None else None
        try:
            self._mini.goto_target(
                head=head,
                antennas=antennas,
                body_yaw=normalized_body_yaw,
                duration=float(duration),
                method="minjerk",
            )
        except Exception as exc:
            raise RuntimeError(f"Motor command failed: {exc}") from exc

    def _gesture_dance(self) -> None:
        with self._motion_lock:
            sequence = (
                (create_head_pose(pitch=10, roll=10, degrees=True), np.deg2rad([45, -45]), np.deg2rad(20)),
                (create_head_pose(pitch=-10, roll=-10, degrees=True), np.deg2rad([-45, 45]), np.deg2rad(-20)),
                (create_head_pose(pitch=10, roll=-10, degrees=True), np.deg2rad([45, 45]), np.deg2rad(20)),
                (create_head_pose(), np.deg2rad([0, 0]), 0.0),
            )
            for head_pose, antenna_pose, yaw in sequence:
                self._goto(head=head_pose, antennas=antenna_pose, body_yaw=yaw, duration=0.4)

    def _gesture_wave_antennas(self) -> None:
        with self._motion_lock:
            for _ in range(2):
                self._goto(antennas=np.deg2rad([60, 60]), duration=0.25)
                self._goto(antennas=np.deg2rad([10, 10]), duration=0.25)
            self._goto(antennas=np.deg2rad([0, 0]), duration=0.3)

    def _gesture_greet(self) -> None:
        """Gentle morning-briefing greeting: a slow head tilt and a soft
        antenna perk, distinct from the higher-energy 'dance' gesture."""
        with self._motion_lock:
            self._goto(head=create_head_pose(pitch=-8, degrees=True), antennas=np.deg2rad([20, 20]), duration=0.9)
            self._goto(head=create_head_pose(pitch=5, roll=5, degrees=True), antennas=np.deg2rad([35, 35]), duration=0.9)
            self._goto(head=create_head_pose(), antennas=np.deg2rad([0, 0]), duration=0.9)


# =============================================================================
# 2. VisionBehaviorTracker
#    Thread-safe state machine for hydration, work-focus, and phone
#    distraction. All mutable state lives behind a single lock per event
#    type so concurrent vision cycles (or a slow one overlapping a fast one)
#    can never corrupt counts.
# =============================================================================

@dataclass(frozen=True)
class HydrationEvent:
    timestamp: datetime
    source_description: str

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(f"HydrationEvent.timestamp must be timezone-aware, got {self.timestamp!r}")
        if not self.source_description.strip():
            raise ValueError("HydrationEvent.source_description must be non-empty")


@dataclass
class ActivityRecord:
    description: str
    duration_seconds: float


class VisionBehaviorTracker:
    """Owns three independent debounced state machines and one generic
    activity-duration tracker, all behind their own locks."""

    def __init__(self, local_timezone: Optional[tzinfo] = None) -> None:
        self._local_timezone = local_timezone

        # generic ongoing-activity tracking (for live display + Claude context)
        self._task_lock = threading.Lock()
        self._current_description: Optional[str] = None
        self._current_start_time: Optional[float] = None
        self._completed: List[ActivityRecord] = []
        self._last_raw_observation: Optional[dict] = None

        # hydration
        self._hydration_lock = threading.Lock()
        self._hydration_events: List[HydrationEvent] = []
        self._is_drinking_now = False
        self._last_counted_drink_at: Optional[float] = None  # time.monotonic()
        # BUG FIX: added to merge brief single-frame misreads into the same
        # ongoing instance instead of ending/restarting the streak.
        self._last_true_drinking_at: Optional[float] = None

        # work focus
        self._work_lock = threading.Lock()
        self._work_streak_start: Optional[float] = None
        # BUG FIX: renamed/repurposed from "fired once ever per streak" to
        # "last time praise fired within this streak" - now re-arms every
        # WORK_MILESTONE_INTERVAL_SECONDS instead of firing only once.
        self._work_last_milestone_at: Optional[float] = None

        # phone distraction
        self._phone_lock = threading.Lock()
        self._phone_first_seen_at: Optional[float] = None
        self._phone_alert_fired_for_streak: Optional[float] = None
        # BUG FIX: added, same merge-tolerance concept as hydration above -
        # a single misclassified frame no longer wipes out an otherwise
        # continuous phone-use streak.
        self._phone_last_true_at: Optional[float] = None
        # NEW: escalation-tier tracking across a "distraction period" -
        # see observe_phone for the full explanation.
        self._phone_last_alert_at: Optional[float] = None
        self._phone_alert_streak_count: int = 0

    # ---- generic activity tracking ------------------------------------------------------

    def record_observation(self, description: str) -> bool:
        now = time.monotonic()
        with self._task_lock:
            if self._current_description is None:
                self._current_description = description
                self._current_start_time = now
                return True
            if self._is_same_activity(self._current_description, description):
                return False
            self._finalize_current_locked(now)
            self._current_description = description
            self._current_start_time = now
            return True

    def _finalize_current_locked(self, end_time: float) -> None:
        if self._current_description is None or self._current_start_time is None:
            return
        duration = end_time - self._current_start_time
        self._completed.append(ActivityRecord(self._current_description, duration))

    @staticmethod
    def _is_same_activity(a: str, b: str) -> bool:
        return is_same_activity(a, b)

    def get_current_task_snapshot(self) -> Optional[tuple[str, float, float]]:
        with self._task_lock:
            if self._current_description is None or self._current_start_time is None:
                return None
            elapsed = time.monotonic() - self._current_start_time
            return self._current_description, elapsed, self._current_start_time

    # ---- NEW: most recent RAW observation (pre-confirmation), used only for
    # tagging user feedback to a specific vision reading ------------------------------------------------------

    def set_last_raw_observation(
        self, description: str, phone_visible: bool, drinking: bool, at_desk_working: bool
    ) -> None:
        with self._task_lock:
            self._last_raw_observation = {
                "description": description,
                "phone_visible": phone_visible,
                "drinking": drinking,
                "at_desk_working": at_desk_working,
            }

    def get_last_raw_observation(self) -> Optional[dict]:
        with self._task_lock:
            return self._last_raw_observation

    def finalize_and_get_summary(self) -> List[ActivityRecord]:
        with self._task_lock:
            self._finalize_current_locked(time.monotonic())
            self._current_description = None
            self._current_start_time = None
            totals: dict[str, float] = {}
            order: List[str] = []
            for record in self._completed:
                if record.description not in totals:
                    totals[record.description] = 0.0
                    order.append(record.description)
                totals[record.description] += record.duration_seconds
            return [ActivityRecord(desc, totals[desc]) for desc in order]

    # ---- hydration: two-tier timing (see DRINKING_SAME_INSTANCE_MERGE_SECONDS
    # and HYDRATION_DEBOUNCE_SECONDS) ------------------------------------------------------

    def observe_drinking(self, is_drinking_now: bool, description: str) -> Optional[int]:
        """Returns the new today's-count if a new drink event was counted
        this call, else None.

        BUG FIX (two-tier timing): previously a single false->true
        transition after just HYDRATION_DEBOUNCE_SECONDS (15s) counted as a
        brand new instance - so one misclassified frame in the middle of a
        single continuous drink could split it into two counted events, or
        two genuinely separate sips 20 seconds apart could get merged
        incorrectly. Now:
          - a brief drop to 'not drinking' within DRINKING_SAME_INSTANCE_
            MERGE_SECONDS of the last true reading is treated as noise, not
            the end of the instance (the streak doesn't reset)
          - a genuinely NEW instance only counts if at least
            HYDRATION_DEBOUNCE_SECONDS has passed since the previous
            COUNTED instance
        """
        now = time.monotonic()
        with self._hydration_lock:
            if is_drinking_now:
                self._last_true_drinking_at = now

            effective_is_drinking = is_drinking_now or (
                self._last_true_drinking_at is not None
                and (now - self._last_true_drinking_at) < DRINKING_SAME_INSTANCE_MERGE_SECONDS
            )

            became_true = effective_is_drinking and not self._is_drinking_now
            self._is_drinking_now = effective_is_drinking

            if not became_true:
                return None

            cooldown_elapsed = (
                self._last_counted_drink_at is None
                or (now - self._last_counted_drink_at) >= HYDRATION_DEBOUNCE_SECONDS
            )
            if not cooldown_elapsed:
                return None

            self._last_counted_drink_at = now
            event = HydrationEvent(timestamp=datetime.now(timezone.utc), source_description=description)
            self._hydration_events.append(event)
            return self._get_todays_count_locked()

    def _get_todays_count_locked(self) -> int:
        today_local = datetime.now(timezone.utc).astimezone(self._local_timezone).date()
        return sum(
            1
            for event in self._hydration_events
            if event.timestamp.astimezone(self._local_timezone).date() == today_local
        )

    def get_todays_hydration_count(self) -> int:
        with self._hydration_lock:
            return self._get_todays_count_locked()

    def total_hydration_events(self) -> int:
        with self._hydration_lock:
            return len(self._hydration_events)

    # ---- work focus: re-arms every WORK_MILESTONE_INTERVAL_SECONDS ------------------------------------------------------

    def observe_work(self, is_working_now: bool) -> bool:
        """Returns True once per completed WORK_MILESTONE_INTERVAL_SECONDS
        interval within a continuous streak - fires repeatedly (at 30min,
        60min, 90min, ...), not just once ever per streak.

        BUG FIX: previously used a single 'has this exact streak already
        been praised' flag that permanently latched after the first fire,
        so a 2-hour work session only ever got praised once, at the 1-minute
        mark. Now tracks the last time praise fired and re-arms once a full
        interval has elapsed again, while still firing at most once per
        interval (not once per vision cycle)."""
        now = time.monotonic()
        with self._work_lock:
            if not is_working_now:
                self._work_streak_start = None
                self._work_last_milestone_at = None
                return False

            if self._work_streak_start is None:
                self._work_streak_start = now
                self._work_last_milestone_at = None
                return False

            elapsed_since_start = now - self._work_streak_start
            if elapsed_since_start < WORK_MILESTONE_INTERVAL_SECONDS:
                return False

            elapsed_since_last_milestone = (
                now - self._work_last_milestone_at
                if self._work_last_milestone_at is not None
                else elapsed_since_start
            )
            if elapsed_since_last_milestone < WORK_MILESTONE_INTERVAL_SECONDS:
                return False

            self._work_last_milestone_at = now
            return True

    # ---- phone distraction: merge-tolerant, fires once per (real) continuous streak ------------------------------------------------------

    def observe_phone(self, is_phone_now: bool) -> int:
        """Returns 0 if no alert should fire this call. Returns 1 for the
        FIRST alert in a new distraction period (gentle tone, suggests one
        task). Returns 2, 3, ... for escalating alerts within the SAME
        ongoing distraction period (firmer but still kind, no repeated
        task-list reading). A period resets back to tier 1 after
        PHONE_DISTRACTION_PERIOD_RESET_SECONDS of being phone-free.

        BUG FIX: previously reset _phone_first_seen_at to None the instant
        ANY single frame read False, so ordinary single-frame vision noise
        during genuine, sustained phone use could repeatedly restart the
        sustained-duration timer and prevent it from ever firing - this is
        exactly what 'terminal shows phone=True but no alert' looked like.
        Now a brief false reading within PHONE_STREAK_MERGE_SECONDS of the
        last true reading is treated as noise, not the end of the streak."""
        now = time.monotonic()
        with self._phone_lock:
            if is_phone_now:
                self._phone_last_true_at = now

            effective_is_phone = is_phone_now or (
                self._phone_last_true_at is not None
                and (now - self._phone_last_true_at) < PHONE_STREAK_MERGE_SECONDS
            )

            if not effective_is_phone:
                self._phone_first_seen_at = None
                self._phone_alert_fired_for_streak = None
                return 0

            if self._phone_first_seen_at is None:
                self._phone_first_seen_at = now
                return 0

            elapsed = now - self._phone_first_seen_at
            if elapsed < PHONE_SUSTAINED_SECONDS:
                return 0
            if self._phone_alert_fired_for_streak == self._phone_first_seen_at:
                return 0

            self._phone_alert_fired_for_streak = self._phone_first_seen_at

            # NEW: escalation-tier bookkeeping, separate from the per-streak
            # dedup above - tracks how many alerts have fired since the
            # last sufficiently-long phone-free gap.
            if (
                self._phone_last_alert_at is not None
                and (now - self._phone_last_alert_at) < PHONE_DISTRACTION_PERIOD_RESET_SECONDS
            ):
                self._phone_alert_streak_count += 1
            else:
                self._phone_alert_streak_count = 1
            self._phone_last_alert_at = now
            return self._phone_alert_streak_count


# =============================================================================
# Vision description helpers
# =============================================================================

def is_same_activity(description_a: str, description_b: str) -> bool:
    ratio = difflib.SequenceMatcher(None, description_a.lower(), description_b.lower()).ratio()
    return ratio >= TASK_SIMILARITY_THRESHOLD


class CaptionConfirmer:
    """Filters single-frame vision hallucinations before they ever reach
    the behavior tracker or trigger an announcement.

    Local image-captioning models like BLIP-base produce slightly different
    wording from cycle to cycle even for a static scene, and occasionally
    hallucinate an activity outright (e.g. "playing a video game" from a
    monitor's glow). Requiring the same activity to appear on
    CAPTION_CONFIRMATION_CYCLES consecutive cycles before it's treated as
    real state filters out one-off noise without meaningfully hurting
    latency - at a 1.5-2s sampling interval, confirmation costs roughly
    3-4 seconds, which is well within the milestone timescales (15s+ for
    hydration, 60s+ for work, 3s+ for phone) this feeds into.

    Not thread-safe by design: it's only ever touched from the single
    vision loop task, never concurrently.
    """

    def __init__(self) -> None:
        self._pending_description: Optional[str] = None
        self._pending_streak = 0
        self._confirmed_description: Optional[str] = None

    def observe(self, raw_description: str) -> Optional[str]:
        """Returns the current confirmed description (which may lag one
        cycle behind `raw_description` while a new candidate is still
        being confirmed), or None if nothing has been confirmed yet this
        session."""
        if self._pending_description is not None and is_same_activity(self._pending_description, raw_description):
            self._pending_streak += 1
        else:
            self._pending_description = raw_description
            self._pending_streak = 1

        if self._pending_streak >= CAPTION_CONFIRMATION_CYCLES:
            self._confirmed_description = self._pending_description

        return self._confirmed_description


@dataclass(frozen=True)
class VisionObservation:
    """Structured output from ClaudeVisionDescriber for a single frame -
    replaces the old approach of regex/keyword-matching a free-text caption,
    which was a real source of both false positives and false negatives
    since a caption like 'two women at a table with drinks' still had to be
    parsed back into booleans after the fact. Claude reasons about the
    frame directly and returns these signals already extracted."""

    person_present: bool
    activity_summary: str
    phone_visible: bool
    drinking: bool
    at_desk_working: bool


def detect_wake_word(transcript: str) -> tuple[bool, str]:
    words = transcript.lower().replace(",", "").replace(".", "").split()
    for index, word in enumerate(words[:-1]):
        if word not in WAKE_HEY_WORDS:
            continue
        candidate = words[index + 1]
        is_close_match = (
            candidate in WAKE_NAME_ALIASES
            or difflib.SequenceMatcher(None, candidate, "reachy").ratio() >= WAKE_SIMILARITY_THRESHOLD
        )
        if is_close_match:
            remainder = " ".join(words[index + 2:]).strip()
            return True, remainder
    return False, ""


def is_sentence_incomplete(transcript: str) -> bool:
    words = transcript.strip().lower().rstrip(".,!?").split()
    return (not words) or (words[-1] in INCOMPLETE_TRAILING_WORDS)


def is_good_morning_command(text: str) -> bool:
    normalized = text.strip().lower().rstrip(".,!?")
    if not normalized:
        return False
    if "good morning" in normalized:
        return True
    ratio = difflib.SequenceMatcher(None, normalized, "good morning").ratio()
    return ratio >= GOOD_MORNING_SIMILARITY_THRESHOLD


VISION_SYSTEM_PROMPT = (
    "You are the vision-analysis component of a desktop robot. You are shown one webcam frame "
    "at a time and must classify it carefully and precisely - look closely at shape, color, and "
    "context before deciding, rather than a quick guess. Respond with ONLY a single JSON object "
    "and nothing else - no prose, no markdown code fences - matching exactly this schema:\n"
    '{"person_present": boolean, "activity_summary": string, "phone_visible": boolean, '
    '"drinking": boolean, "at_desk_working": boolean}\n\n'
    "Field rules:\n"
    "- person_present: true only if at least one person is visible in the frame.\n"
    "- activity_summary: one short factual sentence (max ~15 words), present tense, describing "
    "ONLY what you can clearly and confidently see. Do NOT guess, infer, or assume specific "
    "objects or actions (a phone call, a microphone, a sandwich, etc.) just because they'd be "
    "plausible for the scene - if you're not confident about a specific detail, describe more "
    "generally instead (e.g. 'a person sitting at a desk' rather than guessing what they're "
    "holding). A vaguer-but-accurate description is always better than a specific-but-uncertain "
    f'one. If person_present is false, use exactly "{NO_PERSON_LABEL}".\n'
    "- phone_visible: true ONLY if the person is PHYSICALLY HOLDING a phone IN THEIR HAND while "
    "actively using it right now - looking at its screen, talking into it, or visibly typing/"
    "tapping on it. The phone MUST be gripped/held in a hand - a phone resting on a desk, table, "
    "stand, or any other surface does NOT count, even if its screen is visible, even if a hand is "
    "near it, and even if the person appears to be glancing at it. A "
    "phone that is merely visible (sitting on the desk, in a pocket, glanced at only briefly) "
    "does NOT count - only sustained, active, HELD, in-the-moment use counts. Also do NOT set this true "
    "for other objects that might be held in a similar way or position: food (pizza slices, "
    "sandwiches, snacks), remote controls, wallets, notebooks, or drink containers are NOT "
    "phones, even when held near the face or hand. If the object's shape doesn't clearly match a "
    "phone, or the person isn't clearly actively holding and engaged with it, set this false "
    "rather than guessing.\n"
    "- drinking: true if a cup, mug, glass, water bottle, or similar drink container is visibly "
    "being held, raised toward the mouth, or actively used - regardless of its color or material. "
    "A dark, black, or opaque bottle or cup still counts just as much as a clear/light-colored "
    "one - look for the container's overall shape and how it's being held, not its color, and "
    "don't let a dim or low-contrast object make you default to 'no' if you can still make out "
    "someone gripping a bottle/cup shape. IMPORTANT: do NOT set this true just because a bottle, "
    "cup, or mug is visible on a desk/table - it must be actively being drunk from right now "
    "(lifted toward the mouth, tilted back, visibly sipped) - a container just sitting nearby, "
    "even next to someone's hand, does not count.\n"
    "- at_desk_working: true only if a person is seated at a desk/table and actively using a "
    "laptop or computer.\n\n"
    "INTERNAL CONSISTENCY (important): your boolean fields MUST agree with what you say in "
    "activity_summary - never let them contradict each other. If activity_summary mentions phone "
    "use, phone_visible must be true, and if it doesn't mention a phone, phone_visible must be "
    "false. If activity_summary mentions drinking, drinking must be true, and if it doesn't, "
    "drinking must be false. Decide what's actually happening once, then make every field reflect "
    "that same single judgment - don't let one field 'guess' something the others disagree with.\n\n"
    "Calibration: be reasonably confident (more likely true than not, based on clear visual "
    "evidence) before setting any boolean field to true, or before including a specific detail in "
    "activity_summary - but don't be so cautious that you miss things that are genuinely, clearly "
    "happening. A missed real event and a false alarm are both bad; the goal is matching what a "
    "careful, attentive human observer would actually report, not what's merely possible."
)


# =============================================================================
# NEW: Vision calibration from real user feedback
#
# HONEST SCOPE: this does NOT retrain or fine-tune the underlying Claude
# vision model - there is no mechanism to do that through the API, and nothing
# here claims otherwise. What it DOES do: every confirmation/correction the
# user gives is persisted to disk, and the most relevant recent examples are
# injected into future vision prompts as concrete few-shot guidance ("last
# time you said X, that was wrong - it was actually Y"). This measurably
# improves in-context accuracy on similar future frames, but the improvement
# is gradual and diminishing as examples accumulate - NOT exponential.
# =============================================================================

@dataclass(frozen=True)
class CalibrationExample:
    observed_description: str
    was_correct: bool
    corrected_description: Optional[str]
    corrected_phone_visible: Optional[bool]
    corrected_drinking: Optional[bool]
    corrected_at_desk_working: Optional[bool]
    timestamp: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.observed_description, str) or not self.observed_description.strip():
            raise ValueError("CalibrationExample.observed_description must be a non-empty string.")
        if self.timestamp.tzinfo is None:
            raise ValueError("CalibrationExample.timestamp must be timezone-aware.")


class VisionCalibrationStore:
    """Thread-safe, disk-persisted, APPEND-ONLY log of user-confirmed/
    corrected vision observations - stored as JSON Lines (one JSON object
    per line), not a single JSON array.

    BUG FIX: the previous design read the entire array into memory at
    startup, then rewrote the ENTIRE FILE on every single save. If loading
    ever failed for ANY reason (a malformed entry, a version mismatch,
    etc.), the in-memory list silently reset to empty, and the very next
    save would then overwrite the disk file with just that session's new
    data - permanently erasing every prior session's history. This is
    exactly "second run replaced the first run's data."

    This JSON-Lines, append-only design is structurally immune to that
    failure mode: each new example is written with a single append (mode
    "a") operation, which can only ever ADD bytes to the end of the file -
    it is physically incapable of overwriting or erasing anything already
    on disk, no matter what happens during loading. A single malformed
    line during loading is logged and skipped individually, not treated
    as a reason to discard everything else. The FILE on disk is never
    truncated/capped - only what's held in memory for prompt-building is
    bounded, for sane token cost."""

    MAX_EXAMPLES_IN_PROMPT = 5
    MAX_LOADED_EXAMPLES = 200  # bounds in-memory/prompt use only - the file itself is never trimmed

    def __init__(self, storage_path: Path) -> None:
        self._storage_path = storage_path
        self._lock = threading.Lock()
        self._examples: List[CalibrationExample] = self._load()

    def _load(self) -> List[CalibrationExample]:
        if not self._storage_path.exists():
            print(f"[calibration] No prior calibration history found at {self._storage_path.name} (starting fresh).")
            return []

        # MIGRATION: an earlier version of this store saved a single JSON
        # array (pretty-printed across many lines) rather than JSON Lines.
        # If the file starts with '[', it's that old format - parse it as
        # ONE JSON value (not line-by-line, which would fail on every
        # line of a pretty-printed array) and losslessly convert it to the
        # new append-only format, so existing history is preserved rather
        # than discarded.
        try:
            raw_content = self._storage_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            print(f"[calibration] Failed to read calibration history: {exc}")
            return []

        if raw_content.startswith("["):
            try:
                old_format_items = json.loads(raw_content)
                examples = [self._parse_item(item) for item in old_format_items]
                print(
                    f"[calibration] Migrating {len(examples)} example(s) from the old file format to the "
                    "new append-only format - your existing history is preserved."
                )
                self._rewrite_as_jsonl(examples)
                if len(examples) > self.MAX_LOADED_EXAMPLES:
                    examples = examples[-self.MAX_LOADED_EXAMPLES :]
                return examples
            except Exception as exc:
                print(f"[calibration] Old-format file couldn't be migrated ({exc}) - starting fresh.")
                return []

        # Normal path: JSON Lines, one object per line. A single malformed
        # line is skipped individually rather than discarding everything.
        examples = []
        skipped_lines = 0
        for line_number, raw_line in enumerate(raw_content.splitlines(), start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                examples.append(self._parse_item(json.loads(raw_line)))
            except Exception as exc:
                skipped_lines += 1
                print(f"[calibration] Skipped unreadable line {line_number}: {exc}")

        if skipped_lines:
            print(
                f"[calibration] Loaded {len(examples)} calibration example(s) from previous sessions "
                f"({skipped_lines} unreadable line(s) skipped)."
            )
        else:
            print(f"[calibration] Loaded {len(examples)} calibration example(s) from previous sessions.")

        if len(examples) > self.MAX_LOADED_EXAMPLES:
            examples = examples[-self.MAX_LOADED_EXAMPLES :]
        return examples

    @staticmethod
    def _parse_item(item: dict) -> CalibrationExample:
        return CalibrationExample(
            observed_description=item["observed_description"],
            was_correct=bool(item["was_correct"]),
            corrected_description=item.get("corrected_description"),
            corrected_phone_visible=item.get("corrected_phone_visible"),
            corrected_drinking=item.get("corrected_drinking"),
            corrected_at_desk_working=item.get("corrected_at_desk_working"),
            timestamp=datetime.fromisoformat(item["timestamp"]),
        )

    @staticmethod
    def _serialize_item(example: CalibrationExample) -> str:
        return json.dumps(
            {
                "observed_description": example.observed_description,
                "was_correct": example.was_correct,
                "corrected_description": example.corrected_description,
                "corrected_phone_visible": example.corrected_phone_visible,
                "corrected_drinking": example.corrected_drinking,
                "corrected_at_desk_working": example.corrected_at_desk_working,
                "timestamp": example.timestamp.isoformat(),
            }
        )

    def _rewrite_as_jsonl(self, examples: List[CalibrationExample]) -> None:
        """One-time migration write only - every example has ALREADY been
        successfully parsed into memory before this is called, so this is
        a lossless format conversion, not a "failed to load, so wipe"
        situation. After this runs once, all future writes use add_example's
        pure-append path and never call this again for this file."""
        try:
            lines = [self._serialize_item(example) for example in examples]
            content = "\n".join(lines) + ("\n" if lines else "")
            self._storage_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            print(f"[calibration] Migration write failed (will retry next run): {exc}")

    def add_example(
        self,
        observed_description: str,
        was_correct: bool,
        corrected_description: Optional[str] = None,
        corrected_phone_visible: Optional[bool] = None,
        corrected_drinking: Optional[bool] = None,
        corrected_at_desk_working: Optional[bool] = None,
    ) -> bool:
        """Returns True if the example was recorded, False if
        observed_description was invalid (never raises)."""
        if not isinstance(observed_description, str) or not observed_description.strip():
            return False

        example = CalibrationExample(
            observed_description=observed_description.strip(),
            was_correct=bool(was_correct),
            corrected_description=corrected_description.strip() if corrected_description else None,
            corrected_phone_visible=corrected_phone_visible if isinstance(corrected_phone_visible, bool) else None,
            corrected_drinking=corrected_drinking if isinstance(corrected_drinking, bool) else None,
            corrected_at_desk_working=(
                corrected_at_desk_working if isinstance(corrected_at_desk_working, bool) else None
            ),
            timestamp=datetime.now(timezone.utc),
        )

        payload_line = self._serialize_item(example)

        with self._lock:
            try:
                # BUG FIX: append mode ("a") can only add bytes to the end
                # of the file - structurally incapable of overwriting or
                # erasing anything already saved, unlike the previous
                # "rewrite the whole file from an in-memory snapshot"
                # approach that caused prior sessions' data to be lost.
                with self._storage_path.open("a", encoding="utf-8") as file_handle:
                    file_handle.write(payload_line + "\n")
            except OSError as exc:
                print(f"[calibration] Failed to persist calibration data: {exc}")
                return False

            self._examples.append(example)
            if len(self._examples) > self.MAX_LOADED_EXAMPLES:
                self._examples = self._examples[-self.MAX_LOADED_EXAMPLES :]
        return True

    def build_prompt_addendum(self) -> str:
        """Builds a few-shot calibration block from recent examples,
        prioritizing corrections (mistakes carry more signal than
        confirmations). Returns an empty string if no examples exist yet -
        callers must not append an empty addendum with extra whitespace."""
        with self._lock:
            examples_snapshot = list(self._examples)
        if not examples_snapshot:
            return ""

        corrections = [example for example in examples_snapshot if not example.was_correct]
        confirmations = [example for example in examples_snapshot if example.was_correct]
        selected = (corrections[-4:] + confirmations[-1:])[-self.MAX_EXAMPLES_IN_PROMPT :]
        if not selected:
            return ""

        lines = [
            "\n\nCALIBRATION NOTES from real user feedback on past frames from this exact "
            "camera/setup - use these to avoid repeating past mistakes:"
        ]
        for example in selected:
            if example.was_correct:
                lines.append(
                    f'- Confirmed accurate before: "{example.observed_description}" - keep '
                    "recognizing this kind of scene the same way."
                )
                continue

            correction_bits = []
            if example.corrected_description:
                correction_bits.append(f'what was actually happening: "{example.corrected_description}"')
            if example.corrected_phone_visible is not None:
                correction_bits.append(f"phone_visible should have been {example.corrected_phone_visible}")
            if example.corrected_drinking is not None:
                correction_bits.append(f"drinking should have been {example.corrected_drinking}")
            if example.corrected_at_desk_working is not None:
                correction_bits.append(f"at_desk_working should have been {example.corrected_at_desk_working}")
            correction_text = "; ".join(correction_bits) if correction_bits else "details unspecified"
            lines.append(
                f'- MISTAKE to avoid: previously reported "{example.observed_description}" but '
                f"this was WRONG - {correction_text}."
            )

        return "\n".join(lines)


class VisionFeedbackGUI:
    """NEW: desktop window for confirming/correcting vision observations
    WITHOUT speaking. Polls the tracker's last raw observation and
    refreshes the displayed text; button clicks write directly into the
    calibration store - no voice/Claude round trip needed for this path.
    Runs alongside the existing voice-based record_vision_feedback tool,
    not instead of it - both write into the same store.

    BUG FIX: this MUST be run via run_on_main_thread() from the process's
    actual OS main thread. An earlier version spawned its own background
    thread and called Tk's mainloop() there - Tkinter is not reliably safe
    to create/display windows on a non-main thread across all platforms
    and configurations, and on at least some Windows setups this failed
    completely silently (no exception, no window, nothing visible). See
    main()/_run_with_gui_on_main_thread for how the asyncio backend is
    moved to a background thread instead, freeing up the main thread for
    Tk.

    Uses tkinter.messagebox.askyesnocancel for the phone/drinking/desk
    corrections, which conveniently returns True/False/None - matching
    this codebase's existing Optional[bool] convention for 'yes' / 'no' /
    'don't specify' exactly, with no translation needed."""

    POLL_INTERVAL_MS = 1000

    def __init__(
        self,
        tracker: "VisionBehaviorTracker",
        calibration_store: VisionCalibrationStore,
        stop_event: threading.Event,
    ) -> None:
        self._tracker = tracker
        self._calibration_store = calibration_store
        self._stop_event = stop_event
        self._last_shown_description: Optional[str] = None
        self._root: Optional["tk.Tk"] = None

    def run_on_main_thread(self) -> None:
        """Blocks until the window is closed or stop_event is set. Must be
        called directly (not via threading.Thread) from the real main
        thread."""
        self._root = tk.Tk()
        self._root.title("Reachy Vision Feedback")
        self._root.geometry("480x260")
        self._root.attributes("-topmost", True)
        # BUG FIX: forces the window to actually render and grab focus
        # immediately, rather than potentially appearing behind other
        # windows or not drawing until the first user interaction.
        self._root.lift()
        self._root.focus_force()
        self._root.update()

        tk.Label(self._root, text="Reachy currently sees:", font=("Segoe UI", 11, "bold")).pack(pady=(14, 4))
        self._description_var = tk.StringVar(value="(waiting for first observation...)")
        tk.Label(
            self._root,
            textvariable=self._description_var,
            wraplength=440,
            justify="center",
            font=("Segoe UI", 11),
        ).pack(pady=(0, 18), padx=16)

        button_frame = tk.Frame(self._root)
        button_frame.pack()
        tk.Button(
            button_frame,
            text="\u2713 That's correct",
            bg="#2e7d32",
            fg="white",
            width=18,
            command=self._on_confirm,
        ).grid(row=0, column=0, padx=8)
        tk.Button(
            button_frame,
            text="\u2717 That's wrong",
            bg="#c62828",
            fg="white",
            width=18,
            command=self._on_correct,
        ).grid(row=0, column=1, padx=8)

        self._status_var = tk.StringVar(value="")
        tk.Label(self._root, textvariable=self._status_var, fg="#555555", wraplength=440).pack(pady=(18, 0))

        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_for_updates()
        self._root.mainloop()

    def _poll_for_updates(self) -> None:
        if self._stop_event.is_set():
            self._root.destroy()
            return
        observation = self._tracker.get_last_raw_observation()
        if observation is not None and observation["description"] != self._last_shown_description:
            self._last_shown_description = observation["description"]
            self._description_var.set(observation["description"])
            self._status_var.set("")
        self._root.after(self.POLL_INTERVAL_MS, self._poll_for_updates)

    def _on_confirm(self) -> None:
        observation = self._tracker.get_last_raw_observation()
        if observation is None:
            self._status_var.set("No observation to confirm yet.")
            return
        self._calibration_store.add_example(observed_description=observation["description"], was_correct=True)
        self._status_var.set("Logged as correct. Thanks!")

    def _on_correct(self) -> None:
        observation = self._tracker.get_last_raw_observation()
        if observation is None:
            self._status_var.set("No observation to correct yet.")
            return

        correction_text = simpledialog.askstring(
            "Correct the observation", "What was ACTUALLY happening?", parent=self._root
        )
        if not correction_text or not correction_text.strip():
            self._status_var.set("Correction cancelled - no text entered.")
            return

        corrected_phone = messagebox.askyesnocancel(
            "Phone use?", "Were you actively using your phone? (Cancel = don't specify)", parent=self._root
        )
        corrected_drinking = messagebox.askyesnocancel(
            "Drinking?", "Were you drinking something? (Cancel = don't specify)", parent=self._root
        )
        corrected_desk = messagebox.askyesnocancel(
            "At desk working?",
            "Were you at a desk working on a laptop/computer? (Cancel = don't specify)",
            parent=self._root,
        )

        self._calibration_store.add_example(
            observed_description=observation["description"],
            was_correct=False,
            corrected_description=correction_text.strip(),
            corrected_phone_visible=corrected_phone,
            corrected_drinking=corrected_drinking,
            corrected_at_desk_working=corrected_desk,
        )
        self._status_var.set("Correction logged. Thanks!")

    def _on_close(self) -> None:
        self._root.destroy()


class ClaudeVisionDescriber:
    """Frame-level scene understanding via Claude's multimodal API rather
    than a local captioning model. See the module-level comment near
    CLAUDE_VISION_MODEL for why this replaced BLIP-base."""

    def __init__(
        self,
        client: Anthropic,
        model: str,
        output_lock: threading.Lock,
        calibration_store: Optional[VisionCalibrationStore] = None,
    ) -> None:
        self._client = client
        self._model = model
        self._output_lock = output_lock
        # NEW, optional/defaulted to None: when provided, real user feedback
        # is woven into the system prompt for every frame. Omitting it (the
        # default) leaves behavior completely unchanged from before.
        self._calibration_store = calibration_store

    def describe(self, frame_bgr: np.ndarray) -> VisionObservation:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Received an empty or invalid camera frame.")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 color frame, got shape {frame_bgr.shape}.")

        jpeg_base64 = self._encode_jpeg_base64(frame_bgr)

        system_prompt = VISION_SYSTEM_PROMPT
        if self._calibration_store is not None:
            addendum = self._calibration_store.build_prompt_addendum()
            if addendum:
                system_prompt = VISION_SYSTEM_PROMPT + addendum

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=VISION_MAX_TOKENS,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_base64},
                            },
                            {"type": "text", "text": "Analyze this frame and respond with the JSON object only."},
                        ],
                    }
                ],
            )
        except Exception as exc:
            raise RuntimeError(f"Claude vision request failed: {exc}") from exc

        raw_text = self._extract_text(response)
        observation = self._parse_observation(raw_text)
        return self._reconcile_observation(observation)

    def _reconcile_observation(self, observation: VisionObservation) -> VisionObservation:
        """BUG FIX: cross-validates phone_visible/drinking against the
        free-text activity_summary from the SAME response. Two DIRECTED
        corrections, each matching a specific reported failure mode:
          - phone_visible=True but the text never mentions a phone ->
            downgrade to False (fixes false phone alerts).
          - drinking=False but the text clearly describes drinking ->
            upgrade to True (fixes 'terminal shows drinking, no hydration
            logged' - the text is the more detailed signal here).
        Every override is printed so it's visible/verifiable, and this is
        a backstop behind the prompt's own consistency instruction, not a
        substitute for it."""
        text_lower = observation.activity_summary.lower()

        phone_visible = observation.phone_visible
        if phone_visible and not any(word in text_lower for word in PHONE_KEYWORDS_IN_TEXT):
            with self._output_lock:
                print(
                    f'(vision reconciliation: phone_visible=True but no phone mentioned in '
                    f'"{observation.activity_summary}" - overriding to False)'
                )
            phone_visible = False

        drinking = observation.drinking
        # BUG FIX: was `any(word in text_lower for word in DRINK_KEYWORDS_IN_TEXT)`
        # with plain nouns - matched "water bottle" even when merely sitting
        # unused on a desk. Now requires an active-consumption phrase.
        if not drinking and any(phrase in text_lower for phrase in DRINK_ACTIVE_PHRASES_IN_TEXT):
            with self._output_lock:
                print(
                    f'(vision reconciliation: drinking=False but description mentions active drinking in '
                    f'"{observation.activity_summary}" - overriding to True)'
                )
            drinking = True

        if phone_visible == observation.phone_visible and drinking == observation.drinking:
            return observation

        return VisionObservation(
            person_present=observation.person_present,
            activity_summary=observation.activity_summary,
            phone_visible=phone_visible,
            drinking=drinking,
            at_desk_working=observation.at_desk_working,
        )

    @staticmethod
    def _encode_jpeg_base64(frame_bgr: np.ndarray) -> str:
        image = Image.fromarray(frame_bgr[:, :, ::-1])
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=VISION_JPEG_QUALITY)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _extract_text(response) -> str:
        segments = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return "".join(segments).strip()

    def _parse_observation(self, raw_text: str) -> VisionObservation:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Vision response was not valid JSON: {exc}. Raw response: {raw_text!r}") from exc

        required_keys = {"person_present", "activity_summary", "phone_visible", "drinking", "at_desk_working"}
        missing = required_keys - payload.keys()
        if missing:
            raise RuntimeError(f"Vision response missing required fields {missing}. Raw response: {raw_text!r}")

        try:
            return VisionObservation(
                person_present=bool(payload["person_present"]),
                activity_summary=str(payload["activity_summary"]).strip() or NO_PERSON_LABEL,
                phone_visible=bool(payload["phone_visible"]),
                drinking=bool(payload["drinking"]),
                at_desk_working=bool(payload["at_desk_working"]),
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Vision response had unexpected field types: {exc}. Raw response: {raw_text!r}") from exc


def describe_current_vision_state(tracker: VisionBehaviorTracker) -> Optional[str]:
    """BUG FIX: previously used the CONFIRMED (CaptionConfirmer-gated)
    description, which can lag the terminal's raw observation by up to
    CAPTION_CONFIRMATION_CYCLES cycles (~6s at the default interval) - this
    is exactly what caused Reachy to verbally describe a shirt color that
    had already changed on screen. Now uses the most recent RAW
    observation, which is EXACTLY what's printed in the terminal each
    cycle, so a verbal answer can never disagree with what's on screen.
    Elapsed-time context from the confirmed tracker is still included
    opportunistically when it's still describing the same activity - this
    is purely additive detail, never a source of inaccuracy."""
    raw_observation = tracker.get_last_raw_observation()
    if raw_observation is None:
        return None

    description = raw_observation["description"]

    snapshot = tracker.get_current_task_snapshot()
    if snapshot is not None:
        confirmed_description, elapsed, _ = snapshot
        if is_same_activity(confirmed_description, description):
            return f"{description} (ongoing for {format_duration(elapsed)})"

    return description


def print_session_summary(records: List[ActivityRecord]) -> None:
    column_width = 55
    print("\n" + "=" * 74)
    print("VISION SESSION SUMMARY")
    print("=" * 74)
    if not records:
        print("No activity was observed during this session.")
        print("=" * 74)
        return
    sorted_records = sorted(records, key=lambda r: r.duration_seconds, reverse=True)
    print(f"{'Task':<{column_width}} | Time Spent")
    print("-" * (column_width + 15))
    total_seconds = 0.0
    for record in sorted_records:
        label = truncate_for_display(record.description, column_width)
        print(f"{label:<{column_width}} | {format_duration(record.duration_seconds)}")
        total_seconds += record.duration_seconds
    print("-" * (column_width + 15))
    print(f"{'Total observed time':<{column_width}} | {format_duration(total_seconds)}")
    print("=" * 74)


# =============================================================================
# 3. AssistantServices
#    Claude conversation, Google Calendar, Google Tasks, and weather. Every
#    network call is blocking client code, wrapped for the executor.
# =============================================================================

@dataclass
class CalendarEventSummary:
    title: str
    start_label: str
    # NEW, additive: Optional with a default, so every existing call site
    # that constructs this without the new argument (the untouched morning
    # briefing path) keeps working exactly as before. None means "unknown/
    # all-day event", never a fabricated value.
    duration_label: Optional[str] = None


@dataclass(frozen=True)
class CalendarEventDetail:
    """Richer event representation used ONLY by the new reminder scheduler
    and the new write-path tools - CalendarEventSummary (above) is
    completely untouched and still used exactly as before by the morning
    briefing."""

    event_id: str
    title: str
    start_datetime: datetime  # timezone-aware, UTC
    location: Optional[str]


@dataclass
class TaskSummary:
    title: str
    due_label: Optional[str]


@dataclass
class WeatherSummary:
    condition: str
    high_c: float
    low_c: float
    high_f: float
    low_f: float


WEATHER_CODE_DESCRIPTIONS = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy with rime", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "severe thunderstorms with hail",
}


class GoogleAuthManager:
    """Handles the OAuth installed-app flow once and caches the resulting
    token so subsequent runs don't re-open a browser."""

    def __init__(self, output_lock: threading.Lock) -> None:
        self._output_lock = output_lock
        self._credentials: Optional["Credentials"] = None
        self._lock = threading.Lock()

    def get_credentials(self) -> Optional["Credentials"]:
        if not _GOOGLE_LIBS_AVAILABLE:
            with self._output_lock:
                print("[google] google-api-python-client not installed; calendar/tasks disabled.")
            return None

        with self._lock:
            if self._credentials is not None and self._credentials.valid:
                return self._credentials

            creds = None
            token_path = Path(GOOGLE_TOKEN_PATH)
            if token_path.exists():
                try:
                    creds = Credentials.from_authorized_user_file(str(token_path), GOOGLE_SCOPES)
                except Exception as exc:
                    with self._output_lock:
                        print(f"[google] Failed to load cached token: {exc}")
                    creds = None

            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as exc:
                    with self._output_lock:
                        print(f"[google] Token refresh failed: {exc}")
                    creds = None

            if not creds or not creds.valid:
                credentials_path = Path(GOOGLE_CREDENTIALS_PATH)
                if not credentials_path.exists():
                    with self._output_lock:
                        print(
                            f"[google] '{GOOGLE_CREDENTIALS_PATH}' not found; "
                            "calendar/tasks briefing will be skipped."
                        )
                    return None
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), GOOGLE_SCOPES)
                    creds = flow.run_local_server(port=0)
                except Exception as exc:
                    with self._output_lock:
                        print(f"[google] OAuth authorization failed: {exc}")
                    return None
                token_path.write_text(creds.to_json())

            self._credentials = creds
            return creds


class GoogleCalendarUnavailableError(RuntimeError):
    """Raised when Google Calendar credentials aren't available, so a
    caller can tell 'couldn't check' apart from 'checked and found
    nothing'. Silently treating those as the same thing is exactly what
    produced a false 'you're free today' answer when the calendar
    connection genuinely wasn't working - this exception exists so that
    failure mode can never happen again for the conversational
    list_calendar_events path (see _execute_list_calendar_events below)."""


class GoogleCalendarService:
    def __init__(self, auth_manager: GoogleAuthManager, output_lock: threading.Lock) -> None:
        self._auth_manager = auth_manager
        self._output_lock = output_lock

    def blocking_get_today_events(self) -> List[CalendarEventSummary]:
        """UNCHANGED behavior/signature - still exactly what the morning
        briefing calls, still silently returns [] if credentials aren't
        available. Left exactly as-is per the zero-regression requirement;
        the honesty fix below applies only to the newer conversational
        list_calendar_events path, not this one."""
        now_local = datetime.now().astimezone()
        start_of_day = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        return self._blocking_list_events_between(start_of_day, end_of_day)

    def blocking_get_events_for_date(self, target_date: Optional[date] = None) -> List[CalendarEventSummary]:
        """Reads events for ANY single day - what lets Reachy answer
        'what's on my calendar' / 'am I free Thursday' in normal
        conversation. target_date defaults to today (local) if omitted.

        Raises:
            GoogleCalendarUnavailableError: credentials aren't available.
            Deliberately does NOT swallow this into an empty list - the
            caller (_execute_list_calendar_events) must be able to say
            'I can't check right now' rather than falsely implying it
            checked and found your day empty."""
        if target_date is None:
            start_of_day = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            local_tz = datetime.now().astimezone().tzinfo
            start_of_day = datetime(target_date.year, target_date.month, target_date.day, tzinfo=local_tz)
        end_of_day = start_of_day + timedelta(days=1)
        return self._blocking_list_events_between_strict(start_of_day, end_of_day)

    def _blocking_list_events_between(self, start_of_day: datetime, end_of_day: datetime) -> List[CalendarEventSummary]:
        """Original behavior, unchanged: swallows a missing-credentials
        state into an empty list. Used ONLY by blocking_get_today_events
        (morning briefing) to preserve its exact existing behavior."""
        creds = self._auth_manager.get_credentials()
        if creds is None:
            return []
        return self._fetch_events_between(creds, start_of_day, end_of_day)

    def _blocking_list_events_between_strict(self, start_of_day: datetime, end_of_day: datetime) -> List[CalendarEventSummary]:
        """Same fetch, but RAISES GoogleCalendarUnavailableError instead of
        silently returning [] when credentials aren't available."""
        creds = self._auth_manager.get_credentials()
        if creds is None:
            raise GoogleCalendarUnavailableError(
                "Google Calendar isn't connected (missing/invalid credentials)."
            )
        return self._fetch_events_between(creds, start_of_day, end_of_day)

    def _fetch_events_between(self, creds, start_of_day: datetime, end_of_day: datetime) -> List[CalendarEventSummary]:
        """The actual API call - shared by both variants above. Genuine API
        errors (network issues, rate limits, etc.) still degrade to an
        empty list here, same as before; only the 'no credentials at all'
        case is now distinguishable via the strict variant."""
        try:
            service = build("calendar", "v3", credentials=creds)
            response = (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=start_of_day.isoformat(),
                    timeMax=end_of_day.isoformat(),
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            summaries = []
            for item in response.get("items", []):
                title = item.get("summary", "Untitled event")
                start_info = item.get("start", {})
                start_raw = start_info.get("dateTime", start_info.get("date"))
                end_info = item.get("end", {})
                end_raw = end_info.get("dateTime", end_info.get("date"))
                start_label = self._format_start_label(start_raw)
                duration_label = self._format_duration_label(start_raw, end_raw)
                summaries.append(
                    CalendarEventSummary(title=title, start_label=start_label, duration_label=duration_label)
                )
            return summaries
        except HttpError as exc:
            with self._output_lock:
                print(f"[calendar] Google Calendar API error: {exc}")
            return []
        except Exception as exc:
            with self._output_lock:
                print(f"[calendar] Unexpected error fetching events: {exc}")
            return []

    @staticmethod
    def _format_start_label(start_raw: Optional[str]) -> str:
        if not start_raw:
            return "all day"
        try:
            if "T" in start_raw:
                parsed = datetime.fromisoformat(start_raw)
                return parsed.strftime("%-I:%M %p") if os.name != "nt" else parsed.strftime("%#I:%M %p")
            return "all day"
        except ValueError:
            return start_raw

    @staticmethod
    def _format_duration_label(start_raw: Optional[str], end_raw: Optional[str]) -> Optional[str]:
        """NEW: computes a human-readable duration (e.g. '1 hour', '45
        minutes', '1h 30m') from an event's raw start/end fields. Returns
        None for all-day events (no specific dateTime) or malformed data -
        callers must treat None as 'unknown', never fabricate a duration."""
        if not start_raw or not end_raw or "T" not in start_raw or "T" not in end_raw:
            return None
        try:
            start_dt = datetime.fromisoformat(start_raw)
            end_dt = datetime.fromisoformat(end_raw)
        except ValueError:
            return None

        total_minutes = int((end_dt - start_dt).total_seconds() / 60)
        if total_minutes <= 0:
            return None

        hours, minutes = divmod(total_minutes, 60)
        if hours == 0:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        if minutes == 0:
            return f"{hours} hour{'s' if hours != 1 else ''}"
        return f"{hours}h {minutes}m"

    # ---- NEW: write path (event creation) ------------------------------------------------------

    def blocking_create_event(
        self,
        title: str,
        start_datetime: datetime,
        duration_minutes: int = 60,
        location: Optional[str] = None,
    ) -> str:
        """Creates a calendar event. Returns a short human-readable result
        string (success or a plain-language error) - designed to be spoken
        directly or fed back to Claude as a tool_result. Never raises: any
        failure is caught and turned into an explanatory string instead, so
        one bad request can't take down the calling conversation turn."""
        if not isinstance(title, str) or not title.strip():
            return "I need a title to create that event."
        if start_datetime.tzinfo is None:
            start_datetime = start_datetime.astimezone()
        if not isinstance(duration_minutes, int) or duration_minutes <= 0 or duration_minutes > 24 * 60:
            duration_minutes = 60

        creds = self._auth_manager.get_credentials()
        if creds is None:
            return "I can't add that event - Google Calendar isn't connected."

        try:
            service = build("calendar", "v3", credentials=creds)
            end_datetime = start_datetime + timedelta(minutes=duration_minutes)
            event_body: dict = {
                "summary": title.strip(),
                "start": {"dateTime": start_datetime.isoformat()},
                "end": {"dateTime": end_datetime.isoformat()},
            }
            if location:
                event_body["location"] = str(location).strip()

            service.events().insert(calendarId="primary", body=event_body).execute()
            when_label = start_datetime.strftime("%A at %I:%M %p").replace(" 0", " ")
            return f"Added '{title.strip()}' to your calendar for {when_label}."
        except HttpError as exc:
            with self._output_lock:
                print(f"[calendar] Google Calendar API error while creating event: {exc}")
            return f"I ran into a problem adding that event: {exc}"
        except Exception as exc:
            with self._output_lock:
                print(f"[calendar] Unexpected error creating event: {exc}")
            return f"I ran into an unexpected problem adding that event: {exc}"

    # ---- NEW: write path (delete + edit) ------------------------------------------------------

    def _blocking_find_event_window(self, date_hint: Optional[str]) -> "tuple[Optional[str], str, str]":
        """Shared by blocking_delete_event/blocking_edit_event: resolves the
        search window (a single day if date_hint given, otherwise roughly
        the last week through the next two months). Returns (error_message
        or None, time_min_iso, time_max_iso)."""
        if date_hint:
            try:
                target_date = date.fromisoformat(date_hint)
            except ValueError:
                return f"'{date_hint}' isn't a date I recognize (expected YYYY-MM-DD).", "", ""
            local_tz = datetime.now().astimezone().tzinfo
            day_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=local_tz)
            return None, day_start.isoformat(), (day_start + timedelta(days=1)).isoformat()

        now = datetime.now(timezone.utc)
        return None, (now - timedelta(days=7)).isoformat(), (now + timedelta(days=60)).isoformat()

    def _blocking_find_best_matching_event(self, service, title_query: str, time_min: str, time_max: str) -> Optional[dict]:
        """Fuzzy-matches title_query against events in [time_min, time_max),
        same threshold/approach as GoogleTasksService.blocking_complete_task.
        Returns the raw Google event dict (has 'id', 'summary', 'start',
        'end', ...) or None if nothing matches well enough."""
        response = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy="startTime",
                maxResults=250,
            )
            .execute()
        )
        best_match = None
        best_ratio = 0.0
        for item in response.get("items", []):
            ratio = difflib.SequenceMatcher(None, item.get("summary", "").lower(), title_query.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = item
        return best_match if best_ratio >= 0.4 else None

    def blocking_delete_event(self, title_query: str, date_hint: Optional[str] = None) -> str:
        """Finds the best-matching event by fuzzy title match and deletes
        it. Returns a short human-readable result string, never raises."""
        if not isinstance(title_query, str) or not title_query.strip():
            return "I need an event name to delete."

        creds = self._auth_manager.get_credentials()
        if creds is None:
            return "I can't do that - Google Calendar isn't connected."

        window_error, time_min, time_max = self._blocking_find_event_window(date_hint)
        if window_error:
            return window_error

        try:
            service = build("calendar", "v3", credentials=creds)
            best_match = self._blocking_find_best_matching_event(service, title_query, time_min, time_max)
            if best_match is None:
                return f"I couldn't find an event matching '{title_query}'."

            service.events().delete(calendarId="primary", eventId=best_match["id"]).execute()
            return f"Deleted '{best_match.get('summary', title_query)}' from your calendar."
        except HttpError as exc:
            with self._output_lock:
                print(f"[calendar] Google Calendar API error while deleting event: {exc}")
            return f"I ran into a problem deleting that event: {exc}"
        except Exception as exc:
            with self._output_lock:
                print(f"[calendar] Unexpected error deleting event: {exc}")
            return f"I ran into an unexpected problem deleting that event: {exc}"

    def blocking_edit_event(
        self,
        title_query: str,
        new_title: Optional[str] = None,
        new_start_datetime: Optional[datetime] = None,
        new_duration_minutes: Optional[int] = None,
        new_location: Optional[str] = None,
        date_hint: Optional[str] = None,
    ) -> str:
        """Finds the best-matching event by fuzzy title match and applies
        only the fields provided (title/time/duration/location), leaving
        everything else on the event unchanged. Returns a short
        human-readable result string, never raises."""
        if not isinstance(title_query, str) or not title_query.strip():
            return "I need to know which event to edit."
        if new_title is None and new_start_datetime is None and new_duration_minutes is None and new_location is None:
            return "You didn't tell me what to change about that event."

        creds = self._auth_manager.get_credentials()
        if creds is None:
            return "I can't do that - Google Calendar isn't connected."

        window_error, time_min, time_max = self._blocking_find_event_window(date_hint)
        if window_error:
            return window_error

        try:
            service = build("calendar", "v3", credentials=creds)
            best_match = self._blocking_find_best_matching_event(service, title_query, time_min, time_max)
            if best_match is None:
                return f"I couldn't find an event matching '{title_query}'."

            patch_body: dict = {}
            changes_made: List[str] = []

            if new_title:
                patch_body["summary"] = new_title.strip()
                changes_made.append(f"renamed to '{new_title.strip()}'")

            if new_start_datetime is not None:
                if new_start_datetime.tzinfo is None:
                    new_start_datetime = new_start_datetime.astimezone()
                if new_duration_minutes is not None:
                    duration = timedelta(minutes=new_duration_minutes)
                else:
                    # Preserve the event's existing duration when only the start time is changing.
                    duration = timedelta(hours=1)
                    original_start_raw = best_match.get("start", {}).get("dateTime")
                    original_end_raw = best_match.get("end", {}).get("dateTime")
                    if original_start_raw and original_end_raw:
                        try:
                            duration = datetime.fromisoformat(original_end_raw) - datetime.fromisoformat(original_start_raw)
                        except ValueError:
                            pass
                new_end_datetime = new_start_datetime + duration
                patch_body["start"] = {"dateTime": new_start_datetime.isoformat()}
                patch_body["end"] = {"dateTime": new_end_datetime.isoformat()}
                when_label = new_start_datetime.strftime("%A at %I:%M %p").replace(" 0", " ")
                changes_made.append(f"moved to {when_label}")
            elif new_duration_minutes is not None:
                original_start_raw = best_match.get("start", {}).get("dateTime")
                if original_start_raw:
                    try:
                        original_start = datetime.fromisoformat(original_start_raw)
                        patch_body["end"] = {"dateTime": (original_start + timedelta(minutes=new_duration_minutes)).isoformat()}
                        changes_made.append(f"duration changed to {new_duration_minutes} minutes")
                    except ValueError:
                        pass

            if new_location:
                patch_body["location"] = new_location.strip()
                changes_made.append(f"location set to '{new_location.strip()}'")

            if not patch_body:
                return "I couldn't apply any changes to that event."

            service.events().patch(calendarId="primary", eventId=best_match["id"], body=patch_body).execute()
            return f"Updated '{best_match.get('summary', title_query)}': {'; '.join(changes_made)}."
        except HttpError as exc:
            with self._output_lock:
                print(f"[calendar] Google Calendar API error while editing event: {exc}")
            return f"I ran into a problem editing that event: {exc}"
        except Exception as exc:
            with self._output_lock:
                print(f"[calendar] Unexpected error editing event: {exc}")
            return f"I ran into an unexpected problem editing that event: {exc}"

    # ---- NEW: used ONLY by CalendarReminderScheduler ------------------------------------------------------

    def blocking_get_upcoming_events_detailed(
        self, window_minutes: float, lookback_minutes: float = 0.0
    ) -> List[CalendarEventDetail]:
        """Returns timed events (skips all-day events, which have no
        specific start time to remind about) starting within the next
        `window_minutes`, and optionally reaching `lookback_minutes` INTO
        THE PAST too. Entirely separate from blocking_get_today_events /
        CalendarEventSummary above, which remain untouched and still serve
        the morning briefing exactly as before.

        BUG FIX: lookback_minutes was previously always 0 (timeMin=now),
        which is what caused the missed 'starting now' reminder - Google's
        timeMin actually filters on an event's END time, not start time, so
        an event that both started AND ended within the gap between two
        polls could silently fall out of the query window before the
        scheduler ever got a chance to see it cross into its start-time
        firing range. Looking backward by the caller's grace period closes
        that gap regardless of an event's duration or exact poll timing."""
        creds = self._auth_manager.get_credentials()
        if creds is None:
            return []
        try:
            service = build("calendar", "v3", credentials=creds)
            now = datetime.now(timezone.utc)
            time_min = now - timedelta(minutes=max(0.0, lookback_minutes))
            time_max = now + timedelta(minutes=max(0.0, window_minutes))
            response = (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=time_min.isoformat(),
                    timeMax=time_max.isoformat(),
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            details: List[CalendarEventDetail] = []
            for item in response.get("items", []):
                event_id = item.get("id")
                start_info = item.get("start", {})
                start_raw = start_info.get("dateTime")  # only timed events; all-day events omit dateTime
                if not event_id or not start_raw:
                    continue
                try:
                    start_dt = datetime.fromisoformat(start_raw)
                except ValueError:
                    continue
                if start_dt.tzinfo is None:
                    start_dt = start_dt.astimezone()
                details.append(
                    CalendarEventDetail(
                        event_id=event_id,
                        title=item.get("summary", "Untitled event"),
                        start_datetime=start_dt.astimezone(timezone.utc),
                        location=item.get("location") or None,
                    )
                )
            return details
        except HttpError as exc:
            with self._output_lock:
                print(f"[calendar] Google Calendar API error while fetching upcoming events: {exc}")
            return []
        except Exception as exc:
            with self._output_lock:
                print(f"[calendar] Unexpected error fetching upcoming events: {exc}")
            return []


class GoogleTasksService:
    def __init__(self, auth_manager: GoogleAuthManager, output_lock: threading.Lock) -> None:
        self._auth_manager = auth_manager
        self._output_lock = output_lock

    def blocking_get_today_tasks(self) -> List[TaskSummary]:
        creds = self._auth_manager.get_credentials()
        if creds is None:
            return []
        try:
            service = build("tasks", "v1", credentials=creds)
            tasklists = service.tasklists().list(maxResults=20).execute().get("items", [])
            today_local_date = datetime.now().astimezone().date()

            summaries: List[TaskSummary] = []
            for tasklist in tasklists:
                tasks_response = (
                    service.tasks()
                    .list(tasklist=tasklist["id"], showCompleted=False, maxResults=100)
                    .execute()
                )
                for task in tasks_response.get("items", []):
                    due_raw = task.get("due")
                    if due_raw is None:
                        continue
                    try:
                        due_date = datetime.fromisoformat(due_raw.replace("Z", "+00:00")).astimezone().date()
                    except ValueError:
                        continue
                    if due_date != today_local_date:
                        continue
                    summaries.append(TaskSummary(title=task.get("title", "Untitled task"), due_label="today"))
            return summaries
        except HttpError as exc:
            with self._output_lock:
                print(f"[tasks] Google Tasks API error: {exc}")
            return []
        except Exception as exc:
            with self._output_lock:
                print(f"[tasks] Unexpected error fetching tasks: {exc}")
            return []

    # ---- NEW: write path (create + complete) ------------------------------------------------------

    def blocking_create_task(self, title: str, due_date: Optional[str] = None) -> str:
        """Creates a task. Returns a short human-readable result string
        (success or plain-language error), never raises.

        BUG FIX: when due_date is omitted, this now explicitly defaults to
        TODAY'S date from the system clock, rather than leaving the task
        with no due date at all (which was ending up displayed as an
        incorrect/inconsistent date elsewhere). This makes "task added
        today with no specified due date" deterministically due today,
        computed from our own reliable clock rather than depending on
        Claude's own date inference for this default case."""
        if not isinstance(title, str) or not title.strip():
            return "I need a title to create that task."

        creds = self._auth_manager.get_credentials()
        if creds is None:
            return "I can't add that task - Google Tasks isn't connected."

        if not due_date:
            due_date = datetime.now().astimezone().date().isoformat()

        try:
            service = build("tasks", "v1", credentials=creds)
            tasklists = service.tasklists().list(maxResults=1).execute().get("items", [])
            if not tasklists:
                return "You don't have any task lists set up in Google Tasks yet."
            tasklist_id = tasklists[0]["id"]

            body: dict = {"title": title.strip()}
            try:
                parsed_due = datetime.fromisoformat(due_date)
            except ValueError:
                return f"'{due_date}' isn't a date I recognize, so I didn't set a due date."
            if parsed_due.tzinfo is None:
                parsed_due = parsed_due.astimezone()
            body["due"] = parsed_due.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            due_label = f" due {parsed_due.strftime('%B %d')}"

            service.tasks().insert(tasklist=tasklist_id, body=body).execute()
            return f"Added '{title.strip()}' to your tasks{due_label}."
        except HttpError as exc:
            with self._output_lock:
                print(f"[tasks] Google Tasks API error while creating task: {exc}")
            return f"I ran into a problem adding that task: {exc}"
        except Exception as exc:
            with self._output_lock:
                print(f"[tasks] Unexpected error creating task: {exc}")
            return f"I ran into an unexpected problem adding that task: {exc}"

    def blocking_complete_task(self, title_query: str) -> str:
        """Fuzzy-matches title_query against incomplete tasks across all
        tasklists and marks the best match completed. Returns a short
        human-readable result string, never raises."""
        if not isinstance(title_query, str) or not title_query.strip():
            return "I need a task name to mark complete."

        creds = self._auth_manager.get_credentials()
        if creds is None:
            return "I can't do that - Google Tasks isn't connected."

        try:
            service = build("tasks", "v1", credentials=creds)
            tasklists = service.tasklists().list(maxResults=20).execute().get("items", [])

            best_match: Optional[tuple[str, str, str]] = None  # (tasklist_id, task_id, title)
            best_ratio = 0.0
            for tasklist in tasklists:
                tasks_response = (
                    service.tasks().list(tasklist=tasklist["id"], showCompleted=False, maxResults=100).execute()
                )
                for task in tasks_response.get("items", []):
                    candidate_title = task.get("title", "")
                    ratio = difflib.SequenceMatcher(None, candidate_title.lower(), title_query.lower()).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = (tasklist["id"], task["id"], candidate_title)

            if best_match is None or best_ratio < 0.4:
                return f"I couldn't find a task matching '{title_query}'."

            tasklist_id, task_id, matched_title = best_match
            service.tasks().patch(tasklist=tasklist_id, task=task_id, body={"status": "completed"}).execute()
            return f"Marked '{matched_title}' as completed."
        except HttpError as exc:
            with self._output_lock:
                print(f"[tasks] Google Tasks API error while completing task: {exc}")
            return f"I ran into a problem completing that task: {exc}"
        except Exception as exc:
            with self._output_lock:
                print(f"[tasks] Unexpected error completing task: {exc}")
            return f"I ran into an unexpected problem completing that task: {exc}"

    def blocking_get_pending_tasks(self, limit: int = 10) -> List[TaskSummary]:
        """ALL incomplete tasks across tasklists, NOT date-filtered (unlike
        blocking_get_today_tasks above, which only returns tasks due
        exactly today and remains completely unchanged). Used by the new
        phone-distraction task suggestion, which should surface any
        pending task, not just ones due today."""
        creds = self._auth_manager.get_credentials()
        if creds is None:
            return []
        if not isinstance(limit, int) or limit <= 0:
            limit = 10
        try:
            service = build("tasks", "v1", credentials=creds)
            tasklists = service.tasklists().list(maxResults=20).execute().get("items", [])
            summaries: List[TaskSummary] = []
            for tasklist in tasklists:
                tasks_response = (
                    service.tasks().list(tasklist=tasklist["id"], showCompleted=False, maxResults=100).execute()
                )
                for task in tasks_response.get("items", []):
                    due_raw = task.get("due")
                    due_label: Optional[str] = None
                    if due_raw:
                        try:
                            due_label = datetime.fromisoformat(due_raw.replace("Z", "+00:00")).astimezone().strftime(
                                "%b %d"
                            )
                        except ValueError:
                            due_label = None
                    summaries.append(TaskSummary(title=task.get("title", "Untitled task"), due_label=due_label))
                    if len(summaries) >= limit:
                        return summaries
            return summaries
        except HttpError as exc:
            with self._output_lock:
                print(f"[tasks] Google Tasks API error while fetching pending tasks: {exc}")
            return []
        except Exception as exc:
            with self._output_lock:
                print(f"[tasks] Unexpected error fetching pending tasks: {exc}")
            return []

    # ---- NEW: used ONLY by the phone-distraction callout. Deliberately a
    # SEPARATE method from blocking_get_pending_tasks above (which stays
    # completely untouched) - this one sorts by most-recently-modified
    # first, since Google Tasks doesn't expose a true "created at"
    # timestamp; "updated" is the closest available proxy for recency. ------------------------------------------------------

    def blocking_get_recent_pending_tasks(self, limit: int = 3) -> List[TaskSummary]:
        """Returns up to `limit` pending (incomplete) tasks across all
        tasklists, most-recently-updated first. Never raises."""
        creds = self._auth_manager.get_credentials()
        if creds is None:
            return []
        if not isinstance(limit, int) or limit <= 0:
            limit = 3

        try:
            service = build("tasks", "v1", credentials=creds)
            tasklists = service.tasklists().list(maxResults=20).execute().get("items", [])

            candidates: List[tuple] = []  # (updated_raw, title, due_label)
            for tasklist in tasklists:
                tasks_response = (
                    service.tasks()
                    .list(tasklist=tasklist["id"], showCompleted=False, showHidden=False, maxResults=100)
                    .execute()
                )
                for task in tasks_response.get("items", []):
                    updated_raw = task.get("updated", "")  # RFC3339 - lexicographically sortable
                    due_raw = task.get("due")
                    due_label: Optional[str] = None
                    if due_raw:
                        try:
                            due_label = datetime.fromisoformat(due_raw.replace("Z", "+00:00")).astimezone().strftime(
                                "%b %d"
                            )
                        except ValueError:
                            due_label = None
                    candidates.append((updated_raw, task.get("title", "Untitled task"), due_label))

            candidates.sort(key=lambda entry: entry[0], reverse=True)  # most recently updated first
            return [TaskSummary(title=title, due_label=due_label) for _, title, due_label in candidates[:limit]]
        except HttpError as exc:
            with self._output_lock:
                print(f"[tasks] Google Tasks API error while fetching recent tasks: {exc}")
            return []
        except Exception as exc:
            with self._output_lock:
                print(f"[tasks] Unexpected error fetching recent tasks: {exc}")
            return []

    # ---- NEW: read path for the conversational "what are all my tasks"
    # request. Deliberately a SEPARATE method from blocking_get_pending_tasks
    # above (which stays completely untouched, still used exactly as before
    # for the phone-distraction nudge) - this one is only used by the new
    # list_tasks tool. ------------------------------------------------------

    def blocking_list_all_tasks_summary(self, include_completed: bool = False, limit: int = 50) -> str:
        """Returns a spoken-friendly summary of the user's tasks across
        EVERY tasklist. Returns a plain string suitable to speak directly
        or use as a tool_result - never raises."""
        if not isinstance(limit, int) or limit <= 0:
            limit = 50
        limit = min(limit, 100)  # sane upper bound against a runaway task list

        creds = self._auth_manager.get_credentials()
        if creds is None:
            return "Google Tasks isn't connected, so I can't check your tasks."

        try:
            service = build("tasks", "v1", credentials=creds)
            tasklists = service.tasklists().list(maxResults=20).execute().get("items", [])
            if not tasklists:
                return "You don't have any task lists set up in Google Tasks yet."

            entries: List[tuple] = []  # (title, due_label, is_completed)
            for tasklist in tasklists:
                tasks_response = (
                    service.tasks()
                    .list(tasklist=tasklist["id"], showCompleted=True, showHidden=True, maxResults=100)
                    .execute()
                )
                for task in tasks_response.get("items", []):
                    is_completed = task.get("status") == "completed"
                    if is_completed and not include_completed:
                        continue

                    due_raw = task.get("due")
                    due_label: Optional[str] = None
                    if due_raw:
                        try:
                            due_label = datetime.fromisoformat(due_raw.replace("Z", "+00:00")).astimezone().strftime(
                                "%b %d"
                            )
                        except ValueError:
                            due_label = None

                    entries.append((task.get("title", "Untitled task"), due_label, is_completed))
                    if len(entries) >= limit:
                        break
                if len(entries) >= limit:
                    break

            if not entries:
                return "You have no tasks at all." if include_completed else "You have no pending tasks."

            listed_parts = []
            for title, due_label, is_completed in entries:
                part = title
                if due_label:
                    part += f" (due {due_label})"
                if include_completed and is_completed:
                    part += " (completed)"
                listed_parts.append(part)

            count_word = "task" if len(entries) == 1 else "tasks"
            return f"You have {len(entries)} {count_word}: {'; '.join(listed_parts)}."
        except HttpError as exc:
            with self._output_lock:
                print(f"[tasks] Google Tasks API error while listing all tasks: {exc}")
            return "I ran into a problem checking your tasks."
        except Exception as exc:
            with self._output_lock:
                print(f"[tasks] Unexpected error listing all tasks: {exc}")
            return "I ran into an unexpected problem checking your tasks."

    # ---- NEW: write path (delete + edit) ------------------------------------------------------

    def _blocking_find_best_matching_task(self, service, title_query: str) -> Optional[tuple]:
        """Fuzzy-matches title_query against tasks across ALL tasklists
        (both completed and pending - unlike complete_task's search, which
        only searches pending ones, deleting/editing should be able to
        target a completed task too). Returns (tasklist_id, task_id, title)
        or None if nothing matches well enough. Same 0.4 threshold as
        blocking_complete_task, for consistent matching behavior."""
        tasklists = service.tasklists().list(maxResults=20).execute().get("items", [])
        best_match = None
        best_ratio = 0.0
        for tasklist in tasklists:
            tasks_response = (
                service.tasks()
                .list(tasklist=tasklist["id"], showCompleted=True, showHidden=True, maxResults=100)
                .execute()
            )
            for task in tasks_response.get("items", []):
                candidate_title = task.get("title", "")
                ratio = difflib.SequenceMatcher(None, candidate_title.lower(), title_query.lower()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = (tasklist["id"], task["id"], candidate_title)
        return best_match if best_ratio >= 0.4 else None

    def blocking_delete_task(self, title_query: str) -> str:
        """Finds the best-matching task by fuzzy title match and
        permanently deletes it. Returns a short human-readable result
        string, never raises."""
        if not isinstance(title_query, str) or not title_query.strip():
            return "I need a task name to delete."

        creds = self._auth_manager.get_credentials()
        if creds is None:
            return "I can't do that - Google Tasks isn't connected."

        try:
            service = build("tasks", "v1", credentials=creds)
            best_match = self._blocking_find_best_matching_task(service, title_query)
            if best_match is None:
                return f"I couldn't find a task matching '{title_query}'."

            tasklist_id, task_id, matched_title = best_match
            service.tasks().delete(tasklist=tasklist_id, task=task_id).execute()
            return f"Deleted '{matched_title}' from your tasks."
        except HttpError as exc:
            with self._output_lock:
                print(f"[tasks] Google Tasks API error while deleting task: {exc}")
            return f"I ran into a problem deleting that task: {exc}"
        except Exception as exc:
            with self._output_lock:
                print(f"[tasks] Unexpected error deleting task: {exc}")
            return f"I ran into an unexpected problem deleting that task: {exc}"

    def blocking_edit_task(
        self,
        title_query: str,
        new_title: Optional[str] = None,
        new_due_date: Optional[str] = None,
        clear_due_date: bool = False,
    ) -> str:
        """Finds the best-matching task by fuzzy title match and applies
        only the fields provided. Google Tasks' 'due' field only carries
        meaningful DATE precision - Google's own apps ignore any time-of-day
        component - so a new due date is normalized to midnight UTC of that
        calendar day rather than pretending a specific time was set. Returns
        a short human-readable result string, never raises."""
        if not isinstance(title_query, str) or not title_query.strip():
            return "I need to know which task to edit."
        if new_title is None and new_due_date is None and not clear_due_date:
            return "You didn't tell me what to change about that task."

        creds = self._auth_manager.get_credentials()
        if creds is None:
            return "I can't do that - Google Tasks isn't connected."

        try:
            service = build("tasks", "v1", credentials=creds)
            best_match = self._blocking_find_best_matching_task(service, title_query)
            if best_match is None:
                return f"I couldn't find a task matching '{title_query}'."

            tasklist_id, task_id, matched_title = best_match
            patch_body: dict = {}
            changes_made: List[str] = []

            if new_title:
                patch_body["title"] = new_title.strip()
                changes_made.append(f"renamed to '{new_title.strip()}'")

            if clear_due_date:
                patch_body["due"] = None
                changes_made.append("due date removed")
            elif new_due_date:
                try:
                    parsed_due = date.fromisoformat(new_due_date)
                except ValueError:
                    return f"'{new_due_date}' isn't a date I recognize (expected YYYY-MM-DD)."
                patch_body["due"] = (
                    datetime(parsed_due.year, parsed_due.month, parsed_due.day, tzinfo=timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
                changes_made.append(f"due date changed to {parsed_due.strftime('%B %d')}")

            if not patch_body:
                return "I couldn't apply any changes to that task."

            service.tasks().patch(tasklist=tasklist_id, task=task_id, body=patch_body).execute()
            return f"Updated '{matched_title}': {'; '.join(changes_made)}."
        except HttpError as exc:
            with self._output_lock:
                print(f"[tasks] Google Tasks API error while editing task: {exc}")
            return f"I ran into a problem editing that task: {exc}"
        except Exception as exc:
            with self._output_lock:
                print(f"[tasks] Unexpected error editing task: {exc}")
            return f"I ran into an unexpected problem editing that task: {exc}"

    # ---- NEW: automatic daily rollover of overdue tasks ------------------------------------------------------

    def blocking_rollover_overdue_tasks(self) -> List[str]:
        """Finds every incomplete task whose due date is strictly before
        today and pushes its due date to TOMORROW (relative to right now).
        Repeats naturally on each future call as long as the task stays
        incomplete - each new day, an unfinished task's due date is
        'yesterday' relative to that new day, so it gets caught and pushed
        forward again automatically, with no separate tracking needed.

        NOTE: Google Tasks only supports DAY-level due-date granularity
        (not a specific time of day - Google's own apps ignore any
        time-of-day component, same limitation documented on
        blocking_edit_task), so 'push to 11:59 PM tomorrow' is implemented
        as 'push to tomorrow's date' - there is no way to make Google Tasks
        actually respect a specific time. Returns the titles of tasks that
        were rolled over. Never raises - logs and skips individual tasks
        that error rather than aborting the whole batch over one problem."""
        creds = self._auth_manager.get_credentials()
        if creds is None:
            return []

        rolled_over_titles: List[str] = []
        try:
            service = build("tasks", "v1", credentials=creds)
            tasklists = service.tasklists().list(maxResults=20).execute().get("items", [])

            today_local = datetime.now().astimezone().date()
            tomorrow_due_field = (
                datetime(today_local.year, today_local.month, today_local.day, tzinfo=timezone.utc)
                + timedelta(days=1)
            ).isoformat().replace("+00:00", "Z")

            for tasklist in tasklists:
                tasks_response = (
                    service.tasks()
                    .list(tasklist=tasklist["id"], showCompleted=False, showHidden=False, maxResults=100)
                    .execute()
                )
                for task in tasks_response.get("items", []):
                    due_raw = task.get("due")
                    if not due_raw:
                        continue
                    try:
                        due_date = datetime.fromisoformat(due_raw.replace("Z", "+00:00")).astimezone().date()
                    except ValueError:
                        continue
                    if due_date >= today_local:
                        continue  # not overdue yet - leave it alone

                    try:
                        service.tasks().patch(
                            tasklist=tasklist["id"], task=task["id"], body={"due": tomorrow_due_field}
                        ).execute()
                        rolled_over_titles.append(task.get("title", "Untitled task"))
                    except HttpError as exc:
                        with self._output_lock:
                            print(f"[tasks] Failed to roll over '{task.get('title', 'a task')}': {exc}")
        except HttpError as exc:
            with self._output_lock:
                print(f"[tasks] Google Tasks API error while checking for overdue tasks: {exc}")
        except Exception as exc:
            with self._output_lock:
                print(f"[tasks] Unexpected error while checking for overdue tasks: {exc}")

        return rolled_over_titles


class WeatherService:
    def __init__(self, output_lock: threading.Lock) -> None:
        self._output_lock = output_lock

    def blocking_get_today_forecast(self, latitude: float, longitude: float) -> Optional[WeatherSummary]:
        try:
            response = requests.get(
                OPEN_METEO_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "daily": "temperature_2m_max,temperature_2m_min,weathercode",
                    "temperature_unit": "celsius",
                    "timezone": "auto",
                    "forecast_days": 1,
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            daily = data["daily"]
            high_c = float(daily["temperature_2m_max"][0])
            low_c = float(daily["temperature_2m_min"][0])
            code = int(daily["weathercode"][0])
            condition = WEATHER_CODE_DESCRIPTIONS.get(code, "unusual conditions")
            return WeatherSummary(
                condition=condition,
                high_c=high_c,
                low_c=low_c,
                high_f=high_c * 9 / 5 + 32,
                low_f=low_c * 9 / 5 + 32,
            )
        except requests.RequestException as exc:
            with self._output_lock:
                print(f"[weather] Weather API request failed: {exc}")
            return None
        except (KeyError, IndexError, ValueError) as exc:
            with self._output_lock:
                print(f"[weather] Unexpected weather response shape: {exc}")
            return None


class ConversationManager:
    """Thread-safe running Claude conversation history, web-search enabled.

    NEW: optionally calendar/task-enabled too. calendar_service and
    tasks_service both default to None, so any existing call site that
    constructs this class without them (there is exactly one, in
    async_main, which this change also updates) behaves EXACTLY as before -
    the tool list still contains only WEB_SEARCH_TOOL, and since that tool
    executes server-side, blocking_ask's new tool-execution loop below
    never actually iterates more than once in that case, matching the
    original single-call behavior precisely."""

    MAX_TOOL_ITERATIONS = 5

    def __init__(
        self,
        client: Anthropic,
        model: str,
        max_tokens: int,
        location: Optional[str] = None,
        calendar_service: Optional["GoogleCalendarService"] = None,
        tasks_service: Optional["GoogleTasksService"] = None,
        calibration_store: Optional[VisionCalibrationStore] = None,
        vision_tracker: Optional["VisionBehaviorTracker"] = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._history: list[dict] = []
        self._lock = threading.Lock()
        self._calendar_service = calendar_service
        self._tasks_service = tasks_service
        # NEW, both optional/defaulted to None: record_vision_feedback is
        # only added to the tool list when BOTH are provided (need a store
        # to write into AND a tracker to read the last observation from).
        self._calibration_store = calibration_store
        self._vision_tracker = vision_tracker
        self._system_prompt = self._build_system_prompt(
            location,
            calendar_available=calendar_service is not None,
            tasks_available=tasks_service is not None,
            vision_feedback_available=calibration_store is not None and vision_tracker is not None,
        )

    @staticmethod
    def _build_system_prompt(
        location: Optional[str], calendar_available: bool, tasks_available: bool, vision_feedback_available: bool
    ) -> str:
        base = (
            "You are Reachy, a friendly desktop robot assistant speaking answers "
            "aloud through a speaker. Keep replies concise and conversational - "
            "1 to 3 sentences - since they will be spoken, not read. You have "
            "access to a web search tool; use it whenever a question needs "
            "current or location-specific information rather than relying on "
            "your own knowledge alone."
        )
        if location:
            base += f" The user's current location is {location}; assume that for 'near me' style questions."
        else:
            base += " You don't know the user's location; ask them to specify their city if needed."

        # Only claim these capabilities when the corresponding tools are
        # actually in the tool list (see _build_tools) - never advertise a
        # capability that isn't really available this session.
        if calendar_available:
            base += (
                " You can READ the user's Google Calendar with list_calendar_events (use this "
                "any time they ask what's on their schedule, whether they're free, or about an "
                "existing event), ADD events with add_calendar_event, DELETE events with "
                "delete_calendar_event (when they ask to cancel/remove/delete one), and EDIT "
                "existing events with edit_calendar_event (when they ask to reschedule, rename, "
                "or otherwise change one). If they don't give a clear date/time for a new event, "
                "ask for one rather than guessing. For delete/edit, if it's unclear which event "
                "they mean, ask for clarification rather than guessing which one to modify."
            )
        if tasks_available:
            base += (
                " You can LIST ALL of the user's Google Tasks with list_tasks (use this any time "
                "they ask what tasks/to-dos they have, not just about one specific task), add "
                "tasks with add_task, mark tasks completed with complete_task, DELETE tasks with "
                "delete_task (when they ask to remove/delete one entirely), and EDIT tasks with "
                "edit_task (when they ask to rename a task or change its due date - remember "
                "Google Tasks only supports a due date, not a specific time)."
            )
        if vision_feedback_available:
            base += (
                " Your camera is continuously describing what it sees, and the user may comment "
                "on whether a recent description was accurate ('that's right', 'no, I'm actually "
                "doing X', 'that's wrong, I wasn't on my phone'). Whenever they're clearly giving "
                "feedback like this on a recent observation, use record_vision_feedback to log it "
                "- this is how Reachy's vision gets calibrated from real corrections over time."
            )
        return base

    def _build_tools(self) -> list[dict]:
        tools = [WEB_SEARCH_TOOL]
        if self._calendar_service is not None:
            tools.append(LIST_CALENDAR_EVENTS_TOOL)
            tools.append(ADD_CALENDAR_EVENT_TOOL)
            tools.append(DELETE_CALENDAR_EVENT_TOOL)
            tools.append(EDIT_CALENDAR_EVENT_TOOL)
        if self._tasks_service is not None:
            tools.append(LIST_TASKS_TOOL)
            tools.append(ADD_TASK_TOOL)
            tools.append(COMPLETE_TASK_TOOL)
            tools.append(DELETE_TASK_TOOL)
            tools.append(EDIT_TASK_TOOL)
        if self._calibration_store is not None and self._vision_tracker is not None:
            tools.append(RECORD_VISION_FEEDBACK_TOOL)
        return tools

    def blocking_ask(self, user_text: str, vision_context: Optional[str] = None) -> str:
        # BUG FIX: Claude was never told the actual current date/time
        # anywhere in the prompt, so when interpreting relative expressions
        # like "today" or "tomorrow" (e.g. while filling in start_datetime
        # for add_calendar_event), it had to guess from its own training
        # data - which is exactly how "today" got created as the 11th when
        # it was actually the 10th. Computed FRESH on every call (not once
        # at startup) so this stays correct even across a session that runs
        # for multiple days without restarting.
        now_local = datetime.now().astimezone()
        current_datetime_line = (
            f"For reference, the current date and time is {now_local.strftime('%A, %B %d, %Y, %I:%M %p')} "
            f"(local time). Treat this as ground truth for any relative date/time the user mentions - "
            f"'today', 'tomorrow', 'this weekend', 'in an hour', etc. Do not guess or assume a different date."
        )
        system_prompt = f"{self._system_prompt}\n\n{current_datetime_line}"

        if vision_context:
            system_prompt = (
                f"{system_prompt}\n\nWhat Reachy's camera currently sees: {vision_context}. "
                "If the user asks what they're doing or how long they've been doing something, "
                "answer using this observation."
            )

        with self._lock:
            self._history.append({"role": "user", "content": user_text})
            messages_snapshot = list(self._history)

        tools = self._build_tools()
        current_messages = messages_snapshot

        # NOTE ON HISTORY: only the ORIGINAL user_text and the FINAL reply
        # text get persisted into self._history (below) - intermediate
        # tool_use/tool_result exchanges live only in current_messages,
        # local to this one call, and are discarded once it returns. This
        # keeps the stored conversation history exactly the same simple
        # shape it always was (plain user/assistant text turns), which is
        # what every other part of this file already assumes.
        for _ in range(self.MAX_TOOL_ITERATIONS):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system_prompt,
                    messages=current_messages,
                    tools=tools,
                )
            except Exception:
                with self._lock:
                    if self._history and self._history[-1]["role"] == "user":
                        self._history.pop()
                raise

            if response.stop_reason != "tool_use":
                reply_text = self._extract_text(response) or "Sorry, I wasn't able to come up with an answer for that."
                with self._lock:
                    self._history.append({"role": "assistant", "content": reply_text})
                return reply_text

            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                result_text = self._execute_tool(block.name, block.input)
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_text})

            current_messages = current_messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ]

        # Exceeded MAX_TOOL_ITERATIONS - fail safe rather than looping forever.
        fallback_text = (
            "I tried to take a few actions but I'm having trouble finishing up - "
            "could you check your calendar or tasks directly?"
        )
        with self._lock:
            self._history.append({"role": "assistant", "content": fallback_text})
        return fallback_text

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Executes one client-side tool call and returns a short
        human-readable result string. Never raises - any failure becomes an
        explanatory string instead, so a bad/malformed tool call from the
        model can't crash the conversation turn."""
        try:
            if tool_name == "list_calendar_events":
                return self._execute_list_calendar_events(tool_input)
            if tool_name == "add_calendar_event":
                return self._execute_add_calendar_event(tool_input)
            if tool_name == "delete_calendar_event":
                return self._execute_delete_calendar_event(tool_input)
            if tool_name == "edit_calendar_event":
                return self._execute_edit_calendar_event(tool_input)
            if tool_name == "list_tasks":
                return self._execute_list_tasks(tool_input)
            if tool_name == "add_task":
                return self._execute_add_task(tool_input)
            if tool_name == "complete_task":
                return self._execute_complete_task(tool_input)
            if tool_name == "delete_task":
                return self._execute_delete_task(tool_input)
            if tool_name == "edit_task":
                return self._execute_edit_task(tool_input)
            if tool_name == "record_vision_feedback":
                return self._execute_record_vision_feedback(tool_input)
            return f"Unknown tool: {tool_name}"
        except Exception as exc:
            return f"That action failed unexpectedly: {exc}"

    def _execute_list_calendar_events(self, tool_input: dict) -> str:
        if self._calendar_service is None:
            return "Calendar isn't connected."

        date_raw = tool_input.get("date")
        target_date = None
        date_label = "today"
        if date_raw:
            date_str = str(date_raw).strip()
            try:
                target_date = date.fromisoformat(date_str)
                date_label = target_date.strftime("%A, %B %d")
            except ValueError:
                return f"'{date_str}' isn't a date I recognize (expected YYYY-MM-DD)."

        try:
            events = self._calendar_service.blocking_get_events_for_date(target_date)
        except GoogleCalendarUnavailableError:
            # FIX: this used to be indistinguishable from "checked, found
            # nothing" - a missing/broken Google connection silently
            # produced an empty list, which then got reported as "you're
            # free". Now it's caught explicitly and reported honestly.
            return (
                "I can't actually check your calendar right now - it looks like the Google Calendar "
                "connection isn't working. I'm not able to tell you whether you're free."
            )

        if not events:
            return f"There's nothing on the calendar for {date_label}."

        # FIX: now includes duration when known, instead of only title+start
        # time - this is what previously made Reachy unable to answer any
        # question about how long an event runs.
        listed = "; ".join(
            f"{event.title} at {event.start_label}" + (f" (lasts {event.duration_label})" if event.duration_label else "")
            for event in events
        )
        return f"Calendar for {date_label}: {listed}."

    def _execute_add_calendar_event(self, tool_input: dict) -> str:
        if self._calendar_service is None:
            return "Calendar isn't connected."

        title = str(tool_input.get("title", "")).strip()
        start_raw = str(tool_input.get("start_datetime", "")).strip()
        if not title or not start_raw:
            return "Missing a title or start time for that event."

        try:
            start_dt = datetime.fromisoformat(start_raw)
        except ValueError:
            return f"'{start_raw}' isn't a date/time I recognize."
        if start_dt.tzinfo is None:
            start_dt = start_dt.astimezone()

        duration_raw = tool_input.get("duration_minutes", 60)
        try:
            duration_minutes = int(duration_raw)
        except (TypeError, ValueError):
            duration_minutes = 60
        if duration_minutes <= 0 or duration_minutes > 24 * 60:
            duration_minutes = 60

        location = tool_input.get("location")
        location = str(location).strip() if location else None

        return self._calendar_service.blocking_create_event(title, start_dt, duration_minutes, location)

    def _execute_delete_calendar_event(self, tool_input: dict) -> str:
        if self._calendar_service is None:
            return "Calendar isn't connected."

        title_query = str(tool_input.get("title_query", "")).strip()
        if not title_query:
            return "I need an event name to delete."

        date_hint = tool_input.get("date_hint")
        date_hint = str(date_hint).strip() if date_hint else None

        return self._calendar_service.blocking_delete_event(title_query, date_hint)

    def _execute_edit_calendar_event(self, tool_input: dict) -> str:
        if self._calendar_service is None:
            return "Calendar isn't connected."

        title_query = str(tool_input.get("title_query", "")).strip()
        if not title_query:
            return "I need to know which event to edit."

        date_hint = tool_input.get("date_hint")
        date_hint = str(date_hint).strip() if date_hint else None

        new_title = tool_input.get("new_title")
        new_title = str(new_title).strip() if new_title else None

        new_location = tool_input.get("new_location")
        new_location = str(new_location).strip() if new_location else None

        new_start_datetime = None
        new_start_raw = tool_input.get("new_start_datetime")
        if new_start_raw:
            try:
                new_start_datetime = datetime.fromisoformat(str(new_start_raw).strip())
            except ValueError:
                return f"'{new_start_raw}' isn't a date/time I recognize."

        new_duration_minutes = None
        duration_raw = tool_input.get("new_duration_minutes")
        if duration_raw is not None:
            try:
                candidate_duration = int(duration_raw)
                if 0 < candidate_duration <= 24 * 60:
                    new_duration_minutes = candidate_duration
            except (TypeError, ValueError):
                pass

        return self._calendar_service.blocking_edit_event(
            title_query,
            new_title=new_title,
            new_start_datetime=new_start_datetime,
            new_duration_minutes=new_duration_minutes,
            new_location=new_location,
            date_hint=date_hint,
        )

    def _execute_list_tasks(self, tool_input: dict) -> str:
        if self._tasks_service is None:
            return "Tasks isn't connected."
        include_completed_raw = tool_input.get("include_completed", False)
        include_completed = include_completed_raw is True  # strict: only a literal True opts in
        return self._tasks_service.blocking_list_all_tasks_summary(include_completed=include_completed)

    def _execute_add_task(self, tool_input: dict) -> str:
        if self._tasks_service is None:
            return "Tasks isn't connected."
        title = str(tool_input.get("title", "")).strip()
        if not title:
            return "Missing a title for that task."
        due_date = tool_input.get("due_date")
        due_date = str(due_date).strip() if due_date else None
        return self._tasks_service.blocking_create_task(title, due_date)

    def _execute_complete_task(self, tool_input: dict) -> str:
        if self._tasks_service is None:
            return "Tasks isn't connected."
        title_query = str(tool_input.get("title_query", "")).strip()
        if not title_query:
            return "I need a task name to mark complete."
        return self._tasks_service.blocking_complete_task(title_query)

    def _execute_delete_task(self, tool_input: dict) -> str:
        if self._tasks_service is None:
            return "Tasks isn't connected."
        title_query = str(tool_input.get("title_query", "")).strip()
        if not title_query:
            return "I need a task name to delete."
        return self._tasks_service.blocking_delete_task(title_query)

    def _execute_edit_task(self, tool_input: dict) -> str:
        if self._tasks_service is None:
            return "Tasks isn't connected."

        title_query = str(tool_input.get("title_query", "")).strip()
        if not title_query:
            return "I need to know which task to edit."

        new_title = tool_input.get("new_title")
        new_title = str(new_title).strip() if new_title else None

        new_due_date = tool_input.get("new_due_date")
        new_due_date = str(new_due_date).strip() if new_due_date else None

        clear_due_date = tool_input.get("clear_due_date", False) is True

        return self._tasks_service.blocking_edit_task(
            title_query,
            new_title=new_title,
            new_due_date=new_due_date,
            clear_due_date=clear_due_date,
        )

    def _execute_record_vision_feedback(self, tool_input: dict) -> str:
        if self._calibration_store is None or self._vision_tracker is None:
            return "Vision feedback isn't set up."

        last_observation = self._vision_tracker.get_last_raw_observation()
        if last_observation is None:
            return "I don't have a recent observation to give feedback on yet."

        was_correct = tool_input.get("was_correct")
        if not isinstance(was_correct, bool):
            return "I need to know whether that observation was correct or not."

        corrected_description = tool_input.get("corrected_description")
        corrected_description = str(corrected_description).strip() if corrected_description else None

        corrected_phone = tool_input.get("corrected_phone_visible")
        corrected_phone = corrected_phone if isinstance(corrected_phone, bool) else None

        corrected_drinking = tool_input.get("corrected_drinking")
        corrected_drinking = corrected_drinking if isinstance(corrected_drinking, bool) else None

        corrected_desk = tool_input.get("corrected_at_desk_working")
        corrected_desk = corrected_desk if isinstance(corrected_desk, bool) else None

        recorded = self._calibration_store.add_example(
            observed_description=last_observation["description"],
            was_correct=was_correct,
            corrected_description=corrected_description,
            corrected_phone_visible=corrected_phone,
            corrected_drinking=corrected_drinking,
            corrected_at_desk_working=corrected_desk,
        )
        if not recorded:
            return "I wasn't able to log that feedback - the last observation looked invalid."

        if was_correct:
            return "Thanks, good to know that was accurate - I'll keep recognizing scenes like that the same way."
        return "Got it, thanks for the correction - I'll factor that in going forward."

    @staticmethod
    def _extract_text(response) -> str:
        segments = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return " ".join(segment.strip() for segment in segments if segment.strip())


class AssistantServices:
    """Aggregates Claude conversation, Google Calendar, Google Tasks, and
    weather behind async wrappers around the shared executor."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        executor: concurrent.futures.ThreadPoolExecutor,
        output_lock: threading.Lock,
        conversation: ConversationManager,
        calendar_service: GoogleCalendarService,
        tasks_service: GoogleTasksService,
        weather_service: WeatherService,
        weather_latitude: float,
        weather_longitude: float,
        weather_location_label: str,
    ) -> None:
        self._loop = loop
        self._executor = executor
        self._output_lock = output_lock
        self._conversation = conversation
        self._calendar_service = calendar_service
        self._tasks_service = tasks_service
        self._weather_service = weather_service
        self._weather_latitude = weather_latitude
        self._weather_longitude = weather_longitude
        self._weather_location_label = weather_location_label

    async def ask_claude(self, user_text: str, vision_context: Optional[str]) -> str:
        return await self._loop.run_in_executor(
            self._executor, self._conversation.blocking_ask, user_text, vision_context
        )

    async def get_pending_task_suggestion(self) -> Optional[str]:
        """NEW: returns the title of one pending task (if any exist), for
        the phone-distraction callout to suggest. Returns None if Tasks
        isn't connected or there are simply no pending tasks - callers
        must treat None as 'say nothing extra', not an error."""
        if self._tasks_service is None:
            return None
        tasks = await self._loop.run_in_executor(self._executor, self._tasks_service.blocking_get_pending_tasks, 1)
        if not tasks:
            return None
        return tasks[0].title

    async def get_phone_callout_context(self) -> tuple[List[str], Optional[str]]:
        """NEW: returns (top_3_recently_added_pending_task_titles,
        next_event_today_label_or_None) for the richer, kinder phone
        callout. Both halves degrade gracefully to empty/None if the
        corresponding service isn't connected or there's nothing to report
        - never raises, never blocks on one half if the other is slow
        (fetched concurrently)."""
        tasks_future = None
        if self._tasks_service is not None:
            tasks_future = self._loop.run_in_executor(
                self._executor, self._tasks_service.blocking_get_recent_pending_tasks, 3
            )

        event_future = None
        if self._calendar_service is not None:
            now_local = datetime.now().astimezone()
            midnight = (now_local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            minutes_until_midnight = max(1.0, (midnight - now_local).total_seconds() / 60.0)
            event_future = self._loop.run_in_executor(
                self._executor,
                self._calendar_service.blocking_get_upcoming_events_detailed,
                minutes_until_midnight,
                0.0,
            )

        task_summaries = await tasks_future if tasks_future is not None else []
        events = await event_future if event_future is not None else []

        task_titles = [summary.title for summary in task_summaries]

        next_event_label: Optional[str] = None
        if events:
            earliest = min(events, key=lambda event: event.start_datetime)
            local_start = earliest.start_datetime.astimezone()
            time_label = local_start.strftime("%I:%M %p").lstrip("0")
            next_event_label = f"{earliest.title} at {time_label}"

        return task_titles, next_event_label

    async def build_morning_briefing(self) -> str:
        """Fetches calendar events, tasks, and weather concurrently, then
        composes one spoken briefing string."""
        events_future = self._loop.run_in_executor(
            self._executor, self._calendar_service.blocking_get_today_events
        )
        tasks_future = self._loop.run_in_executor(
            self._executor, self._tasks_service.blocking_get_today_tasks
        )
        weather_future = self._loop.run_in_executor(
            self._executor,
            self._weather_service.blocking_get_today_forecast,
            self._weather_latitude,
            self._weather_longitude,
        )
        events, tasks, weather = await asyncio.gather(events_future, tasks_future, weather_future)
        return self._compose_briefing(events, tasks, weather)

    def _compose_briefing(
        self,
        events: List[CalendarEventSummary],
        tasks: List[TaskSummary],
        weather: Optional[WeatherSummary],
    ) -> str:
        parts = ["Good morning!"]

        if weather is not None:
            parts.append(
                f"In {self._weather_location_label} today, expect {weather.condition} with a high of "
                f"{weather.high_f:.0f} and a low of {weather.low_f:.0f} degrees."
            )
        else:
            parts.append("I couldn't pull today's weather.")

        if events:
            if len(events) == 1:
                parts.append(f"You have one event today: {events[0].title} at {events[0].start_label}.")
            else:
                listed = "; ".join(f"{event.title} at {event.start_label}" for event in events[:4])
                remainder = len(events) - min(4, len(events))
                suffix = f", plus {remainder} more" if remainder > 0 else ""
                parts.append(f"You have {len(events)} events today: {listed}{suffix}.")
        else:
            parts.append("Your calendar is clear today.")

        if tasks:
            if len(tasks) == 1:
                parts.append(f"You have one task due today: {tasks[0].title}.")
            else:
                listed = ", ".join(task.title for task in tasks[:4])
                remainder = len(tasks) - min(4, len(tasks))
                suffix = f", plus {remainder} more" if remainder > 0 else ""
                parts.append(f"You have {len(tasks)} tasks due today: {listed}{suffix}.")
        else:
            parts.append("No tasks are due today.")

        return " ".join(parts)


# =============================================================================
# NEW: CalendarReminderScheduler
#    Polls upcoming Google Calendar events and speaks a creative reminder
#    30 minutes before, and again the moment each event starts. Entirely
#    additive - a new background task alongside (not replacing) the vision
#    loop, audio loop, and watchdog loop already run by MainExecutionLoop.
# =============================================================================

class CalendarReminderScheduler:
    def __init__(
        self,
        calendar_service: GoogleCalendarService,
        hardware: "ReachyHardwareManager",
        output_lock: threading.Lock,
        loop: asyncio.AbstractEventLoop,
        executor: concurrent.futures.ThreadPoolExecutor,
        stop_event: threading.Event,
        poll_interval_seconds: float = DEFAULT_REMINDER_POLL_INTERVAL_SECONDS,
        random_source: Optional[random.Random] = None,
    ) -> None:
        self._calendar_service = calendar_service
        self._hardware = hardware
        self._output_lock = output_lock
        self._loop = loop
        self._executor = executor
        self._stop_event = stop_event
        self._poll_interval_seconds = poll_interval_seconds
        self._random = random_source or random.Random()

        self._fired_lock = threading.Lock()
        self._fired_reminders: set[tuple[str, str]] = set()  # (event_id, reminder_key)
        self._event_start_cache: dict[str, datetime] = {}  # event_id -> start_datetime, for stale-entry pruning

        # Same "keep a strong reference so asyncio can't garbage-collect a
        # fire-and-forget task mid-flight" pattern already used by
        # MainExecutionLoop._spawn_background - duplicated here rather than
        # sharing MainExecutionLoop's private set, since this class is
        # deliberately independent and shouldn't need a reference to it.
        self._background_tasks: set[asyncio.Task] = set()

    def _spawn_background(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._poll_once()
            except Exception as exc:
                with self._output_lock:
                    print(f"[reminders] Unexpected error during reminder poll: {exc}")
            await asyncio.sleep(self._poll_interval_seconds)

    async def _poll_once(self) -> None:
        events = await self._loop.run_in_executor(
            self._executor,
            self._calendar_service.blocking_get_upcoming_events_detailed,
            REMINDER_ADVANCE_WINDOW_MINUTES + 5.0,
            REMINDER_LOOKBACK_BUFFER_MINUTES,
        )
        now = datetime.now(timezone.utc)
        self._prune_stale_entries(now)

        for event in events:
            self._event_start_cache[event.event_id] = event.start_datetime
            minutes_until = (event.start_datetime - now).total_seconds() / 60.0

            # BUG FIX: two mutually-exclusive windows (30-min tier excludes
            # the final 5 minutes, which the 5-min tier owns instead) so
            # the two reminders can't both claim the same moment or produce
            # misleading wording (e.g. "30 minutes" when it's really 4).
            if REMINDER_FIVE_MINUTE_WINDOW_MINUTES < minutes_until <= REMINDER_ADVANCE_WINDOW_MINUTES:
                if self._mark_and_check_should_fire(event.event_id, REMINDER_THIRTY_MINUTE_KEY):
                    message = self._build_thirty_minute_message(event)
                    with self._output_lock:
                        print(f"[reminders] 30-min reminder: {event.title}")
                    self._spawn_background(self._hardware.speak(message))

            if 0.0 < minutes_until <= REMINDER_FIVE_MINUTE_WINDOW_MINUTES:
                if self._mark_and_check_should_fire(event.event_id, REMINDER_FIVE_MINUTE_KEY):
                    message = self._build_five_minute_message(event)
                    with self._output_lock:
                        print(f"[reminders] 5-min reminder: {event.title}")
                    self._spawn_background(self._hardware.speak(message))

    def _mark_and_check_should_fire(self, event_id: str, reminder_key: str) -> bool:
        """Thread-safe, idempotent: returns True the first time this
        (event, reminder type) pair is seen, False every time after -
        guaranteeing each reminder fires exactly once per event."""
        key = (event_id, reminder_key)
        with self._fired_lock:
            if key in self._fired_reminders:
                return False
            self._fired_reminders.add(key)
            return True

    def _prune_stale_entries(self, now: datetime) -> None:
        """Bounds memory growth over a long-running session: fired-reminder
        tracking for events more than a day in the past gets dropped."""
        cutoff = now - timedelta(hours=STALE_REMINDER_RETENTION_HOURS)
        stale_ids = [event_id for event_id, start in self._event_start_cache.items() if start < cutoff]
        if not stale_ids:
            return
        with self._fired_lock:
            self._fired_reminders = {
                (event_id, key) for (event_id, key) in self._fired_reminders if event_id not in stale_ids
            }
        for event_id in stale_ids:
            del self._event_start_cache[event_id]

    def _build_thirty_minute_message(self, event: CalendarEventDetail) -> str:
        if event.location:
            template = self._random.choice(THIRTY_MINUTE_REMINDER_TEMPLATES_WITH_LOCATION)
            return template.format(title=event.title, location=event.location)
        template = self._random.choice(THIRTY_MINUTE_REMINDER_TEMPLATES_NO_LOCATION)
        return template.format(title=event.title)

    def _build_five_minute_message(self, event: CalendarEventDetail) -> str:
        if event.location:
            template = self._random.choice(FIVE_MINUTE_REMINDER_TEMPLATES_WITH_LOCATION)
            return template.format(title=event.title, location=event.location)
        template = self._random.choice(FIVE_MINUTE_REMINDER_TEMPLATES_NO_LOCATION)
        return template.format(title=event.title)


# =============================================================================
# 4. MainExecutionLoop
#    The asyncio event bus. Owns the vision loop, the audio/wake-word loop
#    (fed by a background listener thread via an asyncio.Queue), the
#    follow-up window, and dispatches morning briefings / conversation
#    replies as independent background tasks.
# =============================================================================

class FollowUpWindow:
    def __init__(self, window_seconds: float) -> None:
        self._window_seconds = window_seconds
        self._lock = threading.Lock()
        self._expires_at: Optional[float] = None

    def open(self) -> None:
        with self._lock:
            self._expires_at = time.monotonic() + self._window_seconds

    def is_open(self) -> bool:
        with self._lock:
            return self._expires_at is not None and time.monotonic() < self._expires_at

    def close(self) -> None:
        with self._lock:
            self._expires_at = None


class TaskRolloverScheduler:
    """NEW: periodically finds incomplete tasks whose due date has already
    passed and pushes their due date to TOMORROW, repeating automatically
    on each subsequent check until the task is marked complete - directly
    implements 'if a task isn't done by 11:59 PM on its due day, push it
    to the next day, and keep doing that until it's done'. Runs an
    immediate check on startup (catches anything that went overdue while
    the assistant wasn't running) in addition to the regular interval."""

    def __init__(
        self,
        tasks_service: "GoogleTasksService",
        output_lock: threading.Lock,
        loop: asyncio.AbstractEventLoop,
        executor: concurrent.futures.ThreadPoolExecutor,
        stop_event: threading.Event,
        poll_interval_seconds: float = DEFAULT_TASK_ROLLOVER_POLL_INTERVAL_SECONDS,
    ) -> None:
        self._tasks_service = tasks_service
        self._output_lock = output_lock
        self._loop = loop
        self._executor = executor
        self._stop_event = stop_event
        self._poll_interval_seconds = poll_interval_seconds

    async def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._poll_once()
            except Exception as exc:
                with self._output_lock:
                    print(f"[tasks] Unexpected error during task rollover check: {exc}")
            await asyncio.sleep(self._poll_interval_seconds)

    async def _poll_once(self) -> None:
        rolled_over_titles = await self._loop.run_in_executor(
            self._executor, self._tasks_service.blocking_rollover_overdue_tasks
        )
        if rolled_over_titles:
            with self._output_lock:
                print(f"[tasks] Rolled {len(rolled_over_titles)} overdue task(s) to tomorrow: {', '.join(rolled_over_titles)}")


class MainExecutionLoop:
    def __init__(
        self,
        hardware: ReachyHardwareManager,
        tracker: VisionBehaviorTracker,
        services: AssistantServices,
        describer: ClaudeVisionDescriber,
        executor: concurrent.futures.ThreadPoolExecutor,
        output_lock: threading.Lock,
        vision_interval_seconds: float,
        announce_new_activities: bool,
        save_frame_path: Optional[Path],
        calendar_service: Optional[GoogleCalendarService] = None,
        reminder_poll_interval_seconds: float = DEFAULT_REMINDER_POLL_INTERVAL_SECONDS,
        tasks_service: Optional["GoogleTasksService"] = None,
        task_rollover_poll_interval_seconds: float = DEFAULT_TASK_ROLLOVER_POLL_INTERVAL_SECONDS,
        external_stop_event: Optional[threading.Event] = None,
    ) -> None:
        self._hardware = hardware
        self._tracker = tracker
        self._services = services
        self._describer = describer
        self._executor = executor
        self._output_lock = output_lock
        self._vision_interval_seconds = vision_interval_seconds
        self._announce_new_activities = announce_new_activities
        self._save_frame_path = save_frame_path
        # NEW, both optional/defaulted: if calendar_service is omitted
        # (None), no reminder scheduler is constructed and run() below
        # simply doesn't add a reminder task to the gather() call - every
        # other existing behavior is completely unaffected.
        self._calendar_service = calendar_service
        self._reminder_poll_interval_seconds = reminder_poll_interval_seconds
        self._reminder_scheduler: Optional[CalendarReminderScheduler] = None
        # NEW, same pattern: optional task-rollover scheduler, only
        # constructed when a tasks_service is actually supplied.
        self._tasks_service = tasks_service
        self._task_rollover_poll_interval_seconds = task_rollover_poll_interval_seconds
        self._task_rollover_scheduler: Optional[TaskRolloverScheduler] = None

        self._caption_confirmer = CaptionConfirmer()
        self._follow_up_window = FollowUpWindow(FOLLOW_UP_WINDOW_SECONDS)
        self._transcript_queue: asyncio.Queue[str] = asyncio.Queue()
        # NEW, optional: when provided (by the GUI-mode entry point), this
        # SAME Event object is shared with the desktop feedback window, so
        # closing that window also stops the vision/voice loops, and an
        # internal shutdown (Ctrl+C) also closes the window. Defaults to a
        # fresh, private Event exactly as before when not provided.
        self._stop_event = external_stop_event if external_stop_event is not None else threading.Event()
        self._listener_thread: Optional[threading.Thread] = None
        self._listener_loop: Optional[asyncio.AbstractEventLoop] = None
        self._microphone_source_cm = None
        self._background_tasks: set[asyncio.Task] = set()

        # NEW: conversation-busy tracking so vision-triggered callouts
        # (phone/hydration/work) never interrupt an in-progress voice
        # exchange - including its follow-up window. See
        # _is_conversation_busy / _enqueue_or_speak_vision_message /
        # _vision_message_flush_loop below.
        self._is_actively_listening = False
        self._response_in_flight = False
        self._pending_vision_messages: List[str] = []
        self._pending_vision_messages_lock = threading.Lock()

    # ---- lifecycle ------------------------------------------------------

    async def run(self, microphone_source_cm) -> None:
        loop = asyncio.get_running_loop()
        self._listener_loop = loop
        self._microphone_source_cm = microphone_source_cm

        self._start_listener_thread()

        vision_task = asyncio.create_task(self._vision_loop(), name="VisionLoop")
        transcript_task = asyncio.create_task(self._transcript_dispatch_loop(), name="TranscriptDispatch")
        watchdog_task = asyncio.create_task(self._listener_watchdog_loop(), name="ListenerWatchdog")
        flush_task = asyncio.create_task(self._vision_message_flush_loop(), name="VisionMessageFlush")
        background_tasks = [vision_task, transcript_task, watchdog_task, flush_task]

        # NEW: purely additive fourth task, only created when a calendar
        # service was actually supplied. The three tasks above are
        # constructed and gathered exactly as before regardless.
        if self._calendar_service is not None:
            self._reminder_scheduler = CalendarReminderScheduler(
                self._calendar_service,
                self._hardware,
                self._output_lock,
                loop,
                self._executor,
                self._stop_event,
                poll_interval_seconds=self._reminder_poll_interval_seconds,
            )
            reminder_task = asyncio.create_task(self._reminder_scheduler.run(), name="CalendarReminders")
            background_tasks.append(reminder_task)

        # NEW: same additive pattern for the task-rollover scheduler - only
        # created when a tasks_service was actually supplied.
        if self._tasks_service is not None:
            self._task_rollover_scheduler = TaskRolloverScheduler(
                self._tasks_service,
                self._output_lock,
                loop,
                self._executor,
                self._stop_event,
                poll_interval_seconds=self._task_rollover_poll_interval_seconds,
            )
            rollover_task = asyncio.create_task(self._task_rollover_scheduler.run(), name="TaskRollover")
            background_tasks.append(rollover_task)

        try:
            await asyncio.gather(*background_tasks)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def stop_event(self) -> threading.Event:
        """NEW: public accessor so external components (the vision feedback
        GUI, which runs on its own thread outside this class) can watch for
        shutdown without reaching into a private attribute."""
        return self._stop_event

    def _spawn_background(self, coro) -> None:
        """Fires a coroutine as an independent background task and keeps a
        strong reference to it (asyncio otherwise allows fire-and-forget
        tasks to be garbage-collected mid-flight), removing the reference
        once it completes."""
        task = asyncio.get_running_loop().create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _start_listener_thread(self) -> None:
        self._listener_thread = threading.Thread(
            target=self._blocking_listener_loop,
            args=(self._listener_loop, self._microphone_source_cm),
            daemon=True,
            name="AudioListener",
        )
        self._listener_thread.start()

    async def _listener_watchdog_loop(self) -> None:
        """The listener thread is now wrapped in a catch-all in
        _blocking_listener_loop so it should never die - but hardware
        threads are exactly the place to not fully trust that assumption.
        If it somehow exits anyway (e.g. the microphone context manager
        itself raises on re-entry), this notices within 5s and restarts it
        rather than leaving the robot silently deaf to its wake word for
        the rest of the session."""
        while not self._stop_event.is_set():
            await asyncio.sleep(5.0)
            if self._stop_event.is_set():
                return
            if self._listener_thread is not None and not self._listener_thread.is_alive():
                with self._output_lock:
                    print("[audio] Listener thread was not running - restarting it now.")
                self._start_listener_thread()

    # ---- audio: background thread feeds asyncio.Queue ------------------------------------------------------

    def _blocking_listener_loop(self, loop: asyncio.AbstractEventLoop, microphone_source_cm) -> None:
        try:
            with microphone_source_cm as source:
                with self._output_lock:
                    print("Ready. Listening passively...\n")
                while not self._stop_event.is_set():
                    try:
                        status, transcript = self._hardware.blocking_listen_once(source)
                    except Exception as exc:
                        # This is the actual fix for "wake word stopped being
                        # heard": previously any exception here (a stray
                        # PyAudio error, an unexpected recognize_google
                        # failure type, etc.) that wasn't one of the
                        # specific types blocking_listen_once() already
                        # catches would propagate up and silently kill this
                        # daemon thread - nothing would ever listen again
                        # for the rest of the session, with no error shown.
                        # Catching broadly here and continuing means one bad
                        # cycle costs at most one lost listen attempt.
                        with self._output_lock:
                            print(f"[audio] Recovered from unexpected listener error: {exc}")
                        time.sleep(0.5)
                        continue

                    if status == ListenStatus.TIMEOUT:
                        with self._output_lock:
                            print("(still listening...)")
                        continue
                    if status == ListenStatus.ERROR:
                        continue
                    if status == ListenStatus.UNCLEAR:
                        loop.call_soon_threadsafe(self._transcript_queue.put_nowait, "\x00UNCLEAR\x00")
                        continue
                    if transcript:
                        loop.call_soon_threadsafe(self._transcript_queue.put_nowait, transcript)
        except Exception as exc:
            # Only reachable for a failure in opening/re-entering the
            # microphone context manager itself (everything inside the
            # while loop is already caught above). Logged loudly so the
            # watchdog's restart attempt is visible rather than mysterious.
            with self._output_lock:
                print(f"[audio] Listener thread exiting due to an error opening the microphone: {exc}")

    async def _transcript_dispatch_loop(self) -> None:
        is_actively_listening = False
        pending_partial_text = ""

        while not self._stop_event.is_set():
            transcript = await self._transcript_queue.get()

            if transcript == "\x00UNCLEAR\x00":
                if is_actively_listening:
                    self._spawn_background(self._hardware.speak("Sorry, I didn't catch that. Could you repeat that?"))
                continue

            if not is_actively_listening and self._follow_up_window.is_open():
                self._follow_up_window.close()
                is_actively_listening = True
                pending_partial_text = ""
                with self._output_lock:
                    print("(treating this as your answer to the follow-up - no wake word needed)")

            if not is_actively_listening:
                is_actively_listening, pending_partial_text = await self._handle_passive_transcript(transcript)
            else:
                is_actively_listening, pending_partial_text = await self._handle_active_transcript(
                    transcript, pending_partial_text
                )

            # NEW: mirror the local state into the instance attribute the
            # vision loop reads, so it knows a conversation is in progress.
            self._is_actively_listening = is_actively_listening

    async def _handle_passive_transcript(self, transcript: str) -> tuple[bool, str]:
        matched, remainder = detect_wake_word(transcript)
        if not matched:
            # Closes the third diagnostic gap: the transcript WAS heard
            # clearly (you'll see it above in the "[heard]:" line), but
            # detect_wake_word() didn't find a "hey"-type word immediately
            # followed by something close to "reachy" in it. If you're
            # consistently seeing this right after saying "hey reachy," the
            # mic/transcription pipeline is working fine - the phrasing
            # itself (or how Google is transcribing it) isn't matching.
            with self._output_lock:
                print(f'(transcript did not match the wake word pattern: "{transcript}")')
            return False, ""

        with self._output_lock:
            print(f'Wake word detected: "{transcript}"')

        if not remainder:
            with self._output_lock:
                print("Listening for your request...")
            return True, ""

        if is_good_morning_command(remainder):
            self._spawn_background(self._run_morning_briefing())
            return False, ""

        if is_sentence_incomplete(remainder):
            with self._output_lock:
                print("(sounds cut off - waiting for you to finish)")
            self._spawn_background(self._hardware.speak("Sounds like you got cut off - go ahead and finish your thought."))
            return True, remainder

        self._spawn_background(self._respond_to_request(remainder))
        return False, ""

    async def _handle_active_transcript(self, transcript: str, pending_partial_text: str) -> tuple[bool, str]:
        matched, remainder = detect_wake_word(transcript)
        clean_text = remainder if (matched and remainder) else transcript
        full_text = f"{pending_partial_text} {clean_text}".strip() if pending_partial_text else clean_text

        if is_good_morning_command(full_text):
            self._spawn_background(self._run_morning_briefing())
            return False, ""

        if is_sentence_incomplete(full_text):
            with self._output_lock:
                print("(still sounds cut off - waiting for you to finish)")
            self._spawn_background(self._hardware.speak("Go ahead, finish your thought."))
            return True, full_text

        self._spawn_background(self._respond_to_request(full_text))
        with self._output_lock:
            print("\nBack to passive listening...\n")
        return False, ""

    async def _respond_to_request(self, request_text: str) -> None:
        # NEW: marks the conversation as busy for the full duration of this
        # call (Claude request + speaking the reply), so vision callouts
        # queue instead of interrupting. Always cleared via finally, even
        # on an exception, so a failed request can't leave the flag stuck.
        self._response_in_flight = True
        try:
            with self._output_lock:
                print(f'[you]: "{request_text}"')

            vision_context = describe_current_vision_state(self._tracker)
            if vision_context:
                with self._output_lock:
                    print(f"(vision context for this turn: {vision_context})")

            try:
                reply_text = await self._services.ask_claude(request_text, vision_context)
            except Exception as exc:
                with self._output_lock:
                    print(f"[claude] Request failed: {exc}")
                await self._hardware.speak("Sorry, something went wrong on my end. Could you try that again?")
                return

            with self._output_lock:
                print(f"[reachy]: {reply_text}")
            await self._hardware.speak(reply_text)

            if reply_text.strip().endswith("?"):
                self._follow_up_window.open()
                with self._output_lock:
                    print(f"(follow-up window open for {FOLLOW_UP_WINDOW_SECONDS:.0f}s - no wake word needed)")
        finally:
            self._response_in_flight = False

    async def _run_morning_briefing(self) -> None:
        with self._output_lock:
            print("(good morning command detected - building briefing)")
        try:
            briefing_text = await self._services.build_morning_briefing()
        except Exception as exc:
            with self._output_lock:
                print(f"[briefing] Failed to build morning briefing: {exc}")
            await self._hardware.speak("Good morning! I ran into a problem pulling your briefing together.")
            return

        with self._output_lock:
            print(f"[reachy briefing]: {briefing_text}")

        # Speak the briefing and perform the greeting gesture concurrently -
        # neither blocks the other, and neither blocks vision or audio.
        await asyncio.gather(
            self._hardware.speak(briefing_text),
            self._hardware.run_gesture("greet"),
        )

    # ---- vision loop ------------------------------------------------------

    async def _vision_loop(self) -> None:
        while not self._stop_event.is_set():
            cycle_start = time.monotonic()
            try:
                await self._run_one_vision_cycle()
            except TimeoutError as exc:
                with self._output_lock:
                    print(f"(camera timeout, skipping this cycle: {exc})")
            except (ValueError, RuntimeError) as exc:
                with self._output_lock:
                    print(f"(vision error, skipping this cycle: {exc})")

            elapsed = time.monotonic() - cycle_start
            remaining = max(0.0, self._vision_interval_seconds - elapsed)
            await asyncio.sleep(remaining)

    async def _run_one_vision_cycle(self) -> None:
        frame = await self._hardware.capture_downsampled_frame()

        if self._save_frame_path is not None:
            Image.fromarray(frame[:, :, ::-1]).save(self._save_frame_path)

        loop = asyncio.get_running_loop()
        observation = await loop.run_in_executor(self._executor, self._describer.describe, frame)
        raw_label = observation.activity_summary if observation.person_present else NO_PERSON_LABEL

        timestamp = datetime.now().strftime("%H:%M:%S")
        with self._output_lock:
            print(
                f"[{timestamp}] (raw): {raw_label} "
                f"[phone={observation.phone_visible} drink={observation.drinking} work={observation.at_desk_working}]"
            )

        # NEW: remembers this exact reading so a voice correction/confirmation
        # right after ("that was wrong, I'm actually...") can be tagged to it.
        self._tracker.set_last_raw_observation(
            raw_label, observation.phone_visible, observation.drinking, observation.at_desk_working
        )

        # The activity_summary text still gets a light confirmation pass
        # before it drives the generic duration log / Claude conversation
        # context, since free-text wording can still vary slightly cycle to
        # cycle even when the underlying scene hasn't changed. The
        # phone/drink/work booleans below are NOT gated on this - Claude
        # already returns those as reasoned-about booleans per frame, and
        # the tracker's own time-based debouncing (15s hydration cooldown,
        # 3s sustained phone, 60s sustained work) is what provides
        # stability for those signals.
        confirmed_label = self._caption_confirmer.observe(raw_label)
        if confirmed_label is not None:
            is_new_activity = self._tracker.record_observation(confirmed_label)
            snapshot = self._tracker.get_current_task_snapshot()
            if snapshot is not None:
                _, elapsed, _ = snapshot
                with self._output_lock:
                    print(f"[{timestamp}] Reachy sees (confirmed): {confirmed_label} (elapsed {format_duration(elapsed)})")

            if is_new_activity and self._announce_new_activities:
                self._spawn_background(self._hardware.speak(confirmed_label))

        await self._handle_phone(observation.phone_visible)
        await self._handle_hydration(observation.drinking, raw_label)
        await self._handle_work(observation.at_desk_working)

    def _is_conversation_busy(self) -> bool:
        """NEW: True while ANY part of a voice exchange is in progress -
        from wake-word detection (or mid-answer to a follow-up), through
        an active Claude request, through speaking the reply, and for the
        duration of any open follow-up window afterward. Vision callouts
        check this before speaking and queue instead if busy."""
        return self._is_actively_listening or self._response_in_flight or self._follow_up_window.is_open()

    def _enqueue_or_speak_vision_message(self, message: str) -> None:
        """NEW: speaks immediately if no conversation is in progress,
        otherwise queues the message to be spoken once
        _vision_message_flush_loop notices the conversation has ended -
        this is what stops phone/hydration/work callouts from talking over
        or immediately after an in-progress voice exchange."""
        if self._is_conversation_busy():
            with self._pending_vision_messages_lock:
                self._pending_vision_messages.append(message)
            with self._output_lock:
                print(f"(deferring vision callout until conversation is done: {truncate_for_display(message, 60)})")
            return
        self._spawn_background(self._hardware.speak(message))

    async def _vision_message_flush_loop(self) -> None:
        """NEW: independent background loop that speaks queued vision
        callouts as soon as the conversation is genuinely idle. Checks
        roughly once a second - frequent enough to feel responsive, cheap
        enough to not matter."""
        while not self._stop_event.is_set():
            await asyncio.sleep(1.0)
            if self._is_conversation_busy():
                continue
            with self._pending_vision_messages_lock:
                if not self._pending_vision_messages:
                    continue
                message = self._pending_vision_messages.pop(0)
            self._spawn_background(self._hardware.speak(message))

    async def _handle_phone(self, is_phone_now: bool) -> None:
        alert_tier = self._tracker.observe_phone(is_phone_now)
        if alert_tier <= 0:
            return

        if alert_tier == 1:
            # FIRST alert in a new distraction period - gentle tone,
            # suggest ONE task (not the whole list - keeps it light) and
            # today's next event if there is one.
            task_titles, next_event_label = await self._services.get_phone_callout_context()
            message_parts = [PHONE_CALLOUT_MESSAGE]
            if task_titles:
                message_parts.append(f"If you have a moment, you could work on: {task_titles[0]}.")
            if next_event_label:
                message_parts.append(f"Also, you've got {next_event_label} coming up today.")
            self._enqueue_or_speak_vision_message(" ".join(message_parts))
        else:
            # Escalating within the SAME ongoing distraction period -
            # firmer but still kind, deliberately NOT re-reading the task
            # list every time (that repetition was the actual source of
            # annoyance, not the reminder itself).
            message = random.choice(PHONE_ESCALATION_MESSAGES)
            self._enqueue_or_speak_vision_message(message)

    async def _handle_hydration(self, is_drinking_now: bool, description: str) -> None:
        new_count = self._tracker.observe_drinking(is_drinking_now, description)
        if new_count is None:
            return

        with self._output_lock:
            print(f"(hydration event logged - today's count: {new_count})")

        if new_count % HYDRATION_MILESTONE_INTERVAL == 0:
            self._enqueue_or_speak_vision_message(HYDRATION_MILESTONE_MESSAGE)
            self._spawn_background(self._hardware.run_gesture("wave_antennas"))

    async def _handle_work(self, is_working_now: bool) -> None:
        should_praise = self._tracker.observe_work(is_working_now)
        if should_praise:
            self._enqueue_or_speak_vision_message(WORK_MILESTONE_MESSAGE)
            self._spawn_background(self._hardware.run_gesture("dance"))


# =============================================================================
# Entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reachy Mini asynchronous personal assistant + behavior watch")
    parser.add_argument("--silence", type=float, default=DEFAULT_SILENCE_SECONDS,
                         help=f"Seconds of quiet before a phrase is considered finished (default: {DEFAULT_SILENCE_SECONDS}).")
    parser.add_argument("--device-index", type=int, default=None,
                         help="Explicit PyAudio microphone device index, if the system default input "
                              "device isn't the right one. Defaults to the system default.")
    parser.add_argument("--location", type=str, default=None,
                         help="City/area label (e.g. 'Santa Clara, CA') used for web search and weather display.")
    parser.add_argument("--latitude", type=float, default=37.3541, help="Latitude for weather (default: Santa Clara, CA).")
    parser.add_argument("--longitude", type=float, default=-121.9552, help="Longitude for weather (default: Santa Clara, CA).")
    parser.add_argument("--interval", type=float, default=DEFAULT_CAPTURE_INTERVAL_SECONDS,
                         help=f"Seconds between camera samples (default: {DEFAULT_CAPTURE_INTERVAL_SECONDS}). "
                              "Each sample is a Claude API call - lower values are more responsive but cost "
                              "and rate-limit usage scale with frequency.")
    parser.add_argument("--vision-speak", action="store_true",
                         help="Announce every newly-observed activity aloud.")
    parser.add_argument("--save-frame", type=str, default=None,
                         help="Path to overwrite with the latest captured frame each vision cycle.")
    parser.add_argument("--reminder-poll-interval", type=float, default=DEFAULT_REMINDER_POLL_INTERVAL_SECONDS,
                         help=f"Seconds between checks for upcoming calendar events to remind about "
                              f"(default: {DEFAULT_REMINDER_POLL_INTERVAL_SECONDS}).")
    parser.add_argument("--no-gui-feedback", action="store_true",
                         help="Disable the desktop vision-feedback window (confirm/correct via voice instead).")
    return parser.parse_args()


async def async_main(
    args: argparse.Namespace,
    gui_handoff: Optional[dict] = None,
    gui_ready_event: Optional[threading.Event] = None,
    external_stop_event: Optional[threading.Event] = None,
) -> None:
    """gui_handoff/gui_ready_event/external_stop_event are all optional and
    None by default - when omitted, this behaves EXACTLY as before (no GUI
    coordination overhead). They're populated by _run_with_gui_on_main_thread
    when the desktop feedback window is enabled, so the GUI (built on the
    real main thread) can receive the tracker/calibration_store this
    function constructs, and share one shutdown signal with it."""
    if args.silence <= 0:
        print("Error: --silence must be a positive number of seconds.")
        sys.exit(1)
    if args.interval <= 0:
        print("Error: --interval must be a positive number of seconds.")
        sys.exit(1)
    if args.reminder_poll_interval <= 0:
        print("Error: --reminder-poll-interval must be a positive number of seconds.")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            'ANTHROPIC_API_KEY is not set. Run:\n  $env:ANTHROPIC_API_KEY = "your-key-here"\nthen try again.'
        )

    output_lock = threading.Lock()
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=EXECUTOR_MAX_WORKERS, thread_name_prefix="reachy-io")

    # BUG FIX: explicit timeout, defense-in-depth alongside the global
    # socket timeout above - a hung Claude API call was a likely
    # contributor to shutdown never completing.
    client = Anthropic(api_key=api_key, timeout=30.0)

    auth_manager = GoogleAuthManager(output_lock)
    calendar_service = GoogleCalendarService(auth_manager, output_lock)
    tasks_service = GoogleTasksService(auth_manager, output_lock)
    weather_service = WeatherService(output_lock)

    # NEW: constructed before ConversationManager/ClaudeVisionDescriber so
    # both can be wired up to it. tracker is also moved up from its
    # previous location (further below) for the same reason - Conversation
    # Manager needs it to look up the most recent raw observation when the
    # user gives voice feedback on it.
    calibration_store = VisionCalibrationStore(VISION_CALIBRATION_PATH)
    tracker = VisionBehaviorTracker()

    # NEW: hand these off to the main thread for the GUI, if this run is
    # using it. Both params are None in the normal (no-GUI) path, in which
    # case this is a no-op.
    if gui_handoff is not None:
        gui_handoff["tracker"] = tracker
        gui_handoff["calibration_store"] = calibration_store
    if gui_ready_event is not None:
        gui_ready_event.set()

    # NOTE: conversation is now constructed AFTER calendar_service/
    # tasks_service (previously it was built first, before either existed).
    # This reordering is required so Claude can be given the calendar/task
    # tools - it is the only change to this block's ordering; nothing about
    # what conversation does with plain text-only turns changes.
    conversation = ConversationManager(
        client,
        CLAUDE_MODEL,
        CLAUDE_MAX_TOKENS,
        location=args.location,
        calendar_service=calendar_service,
        tasks_service=tasks_service,
        calibration_store=calibration_store,
        vision_tracker=tracker,
    )

    weather_location_label = args.location or f"({args.latitude:.2f}, {args.longitude:.2f})"

    print("Connecting to Reachy Mini...")
    with ReachyMini(media_backend="default") as mini:
        hardware = ReachyHardwareManager(mini, loop, executor, output_lock)
        hardware.configure_recognizer(args.silence)

        try:
            microphone = hardware.open_microphone(device_index=args.device_index)
        except RuntimeError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

        describer = ClaudeVisionDescriber(client, CLAUDE_VISION_MODEL, output_lock, calibration_store=calibration_store)

        services = AssistantServices(
            loop,
            executor,
            output_lock,
            conversation,
            calendar_service,
            tasks_service,
            weather_service,
            args.latitude,
            args.longitude,
            weather_location_label,
        )

        save_frame_path = Path(args.save_frame) if args.save_frame else None
        main_loop = MainExecutionLoop(
            hardware,
            tracker,
            services,
            describer,
            executor,
            output_lock,
            vision_interval_seconds=args.interval,
            announce_new_activities=args.vision_speak,
            save_frame_path=save_frame_path,
            calendar_service=calendar_service,
            reminder_poll_interval_seconds=args.reminder_poll_interval,
            tasks_service=tasks_service,
            external_stop_event=external_stop_event,
        )

        # NOTE: the desktop feedback window (if enabled) is now built and
        # run separately, on the real main thread, by
        # _run_with_gui_on_main_thread - not here. This function only needs
        # to hand off tracker/calibration_store (done above) and share its
        # stop_event (via external_stop_event, passed into MainExecutionLoop
        # just above) - see the module-level comment on VisionFeedbackGUI
        # for why building it on a background thread previously failed.

        print(
            f"Vision watch: sampling every {args.interval:.1f}s via Claude vision "
            f"(downsampled to {VISION_DOWNSAMPLE_MAX_DIMENSION}px, model={CLAUDE_VISION_MODEL})."
        )
        print(f'Voice assistant: say "hey reachy" to start talking, or "hey reachy, good morning" for your briefing.')
        print(
            f"Calendar reminders: checking every {args.reminder_poll_interval:.0f}s for events starting within "
            f"{REMINDER_ADVANCE_WINDOW_MINUTES:.0f} minutes."
        )
        print('You can also say things like "add a task to buy groceries" or "schedule a meeting tomorrow at 3pm".')
        if args.location:
            print(f"Location context: {args.location}")
        print("Ctrl+C to stop everything and see the vision session summary.\n")

        try:
            await main_loop.run(microphone)
        except KeyboardInterrupt:
            pass
        finally:
            main_loop.stop()
            summary = tracker.finalize_and_get_summary()
            print_session_summary(summary)
            print(
                f"Hydration: {tracker.get_todays_hydration_count()} drink instance(s) logged today, "
                f"{tracker.total_hydration_events()} total this session."
            )
            executor.shutdown(wait=False, cancel_futures=True)


def _print_fallback_summary(handoff: dict) -> None:
    """BUG FIX (part of the shutdown-hang fix): prints the session summary
    directly from whatever thread calls this - normally the main thread -
    using the tracker reference handed off at startup. Used when the
    backend thread doesn't finish within the shutdown timeout, so the user
    reliably sees their summary instead of waiting indefinitely for a
    backend that may be stuck on a hung network call. Safe to call even if
    the backend thread is still concurrently running - the tracker's own
    internal locking protects against data corruption, though the exact
    snapshot may be a moment behind whatever the backend was doing right
    at that instant."""
    tracker = handoff.get("tracker")
    if tracker is None:
        return
    try:
        summary = tracker.finalize_and_get_summary()
        print_session_summary(summary)
        print(
            f"Hydration: {tracker.get_todays_hydration_count()} drink instance(s) logged today, "
            f"{tracker.total_hydration_events()} total this session."
        )
    except Exception as exc:
        print(f"[shutdown] Could not print the fallback summary: {exc}")


def _wait_for_backend_shutdown(backend_thread: threading.Thread, handoff: dict, timeout: float = 20.0) -> None:
    """BUG FIX: replaces the previous bare/unbounded backend_thread.join()
    (which caused shutdown to hang indefinitely "most of the time" - a
    single stalled network call anywhere in the backend, with no timeout
    of its own, could block it from ever finishing). Waits up to `timeout`
    seconds for the backend's OWN clean shutdown (which normally prints
    the summary itself, inside async_main's finally block). If it doesn't
    finish in time, prints the summary directly from the CALLING thread
    instead (via _print_fallback_summary), so the user is never left
    waiting forever with no summary at all - bounded wait, guaranteed
    result, either way."""
    print("Shutting down - waiting for Reachy's background tasks to finish (this prints the session summary)...")
    backend_thread.join(timeout=timeout)
    if backend_thread.is_alive():
        print(
            f"[shutdown] Background tasks are taking longer than {timeout:.0f}s to finish (likely a slow network "
            "call) - showing your session summary now anyway."
        )
        _print_fallback_summary(handoff)
        print("(Reachy will keep finishing up in the background; it's fine to close this window now.)")
    print("\nStopped.")


def _run_with_gui_on_main_thread(args: argparse.Namespace) -> None:
    """Runs the asyncio backend (voice, vision, calendar, everything) on a
    background thread, and the Tkinter feedback window's mainloop on the
    REAL main thread - see VisionFeedbackGUI's docstring for why this is
    the correct architecture.

    BUG FIX (reliable typed shutdown): the earlier version only offered a
    typed 'open'/'quit' prompt AFTER the feedback window had been closed -
    there was no way to type a command to end the session at any other
    time, and Ctrl+C proved fragile across several rounds of Tkinter/
    Windows-specific signal-delivery quirks. This now runs ONE dedicated
    stdin-watcher thread for the entire program lifetime that accepts
    'end'/'quit'/'exit' at ANY point (window open, closed, or mid-startup)
    as the primary, reliable way to stop and see the summary - Ctrl+C
    remains as a backup. Deliberately only ONE thread ever calls input():
    two different threads both reading stdin concurrently would race
    unpredictably over who receives each typed line, which is why the old
    'open' prompt no longer does its own separate input() call - it now
    just waits on an Event that this single watcher thread sets."""
    handoff: dict = {"tracker": None, "calibration_store": None, "error": None}
    ready_event = threading.Event()
    shutdown_event = threading.Event()
    active_gui: dict = {"window": None}
    reopen_signal = threading.Event()
    reopen_requested: dict = {"flag": False}

    def run_async_backend() -> None:
        try:
            asyncio.run(
                async_main(args, gui_handoff=handoff, gui_ready_event=ready_event, external_stop_event=shutdown_event)
            )
        except Exception as exc:
            handoff["error"] = exc
        finally:
            shutdown_event.set()
            ready_event.set()  # unblock the main thread if it was still waiting on startup
            reopen_signal.set()  # unblock the reopen-wait loop below if it's waiting

    def request_shutdown_and_close_window() -> None:
        shutdown_event.set()
        gui_wrapper = active_gui.get("window")
        if gui_wrapper is not None and gui_wrapper._root is not None:
            try:
                gui_wrapper._root.after(0, gui_wrapper._root.destroy)
            except Exception:
                pass
        reopen_signal.set()

    def stdin_watcher() -> None:
        """The ONLY thread that ever reads stdin. Runs for the entire
        program lifetime, independent of whether the GUI window is
        currently open or closed."""
        while not shutdown_event.is_set():
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                request_shutdown_and_close_window()
                break
            normalized = line.strip().lower()
            if normalized in ("end", "quit", "exit"):
                request_shutdown_and_close_window()
                break
            if normalized == "open":
                reopen_requested["flag"] = True
                reopen_signal.set()

    def handle_sigint(signum, frame) -> None:
        # BUG FIX: a custom signal.signal() handler REPLACES Python's
        # default SIGINT behavior entirely - it must explicitly raise
        # KeyboardInterrupt itself, or whatever is currently blocking
        # (mainloop(), input(), anything) never actually gets interrupted.
        request_shutdown_and_close_window()
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGINT, handle_sigint)

    backend_thread = threading.Thread(target=run_async_backend, daemon=True, name="AsyncioBackend")
    backend_thread.start()

    stdin_thread = threading.Thread(target=stdin_watcher, daemon=True, name="StdinWatcher")
    stdin_thread.start()

    print("Type 'end' and press Enter at any time to stop and see the session summary (or press Ctrl+C).\n")

    try:
        if not ready_event.wait(timeout=60):
            print("[gui] Timed out waiting for the assistant to finish starting up - feedback window not shown.")
            shutdown_event.set()
            _wait_for_backend_shutdown(backend_thread, handoff, timeout=15.0)
            return

        if handoff["tracker"] is None or handoff["calibration_store"] is None:
            if handoff["error"] is not None:
                print(f"[gui] Assistant failed to start: {handoff['error']}")
            _wait_for_backend_shutdown(backend_thread, handoff, timeout=15.0)
            return

        tracker = handoff["tracker"]
        calibration_store = handoff["calibration_store"]

        while not shutdown_event.is_set():
            print("Vision feedback window opened - use it to confirm/correct observations without speaking.")
            gui = VisionFeedbackGUI(tracker, calibration_store, shutdown_event)
            active_gui["window"] = gui
            try:
                gui.run_on_main_thread()
            except KeyboardInterrupt:
                shutdown_event.set()
                break
            finally:
                active_gui["window"] = None

            if shutdown_event.is_set():
                # The backend (or the stdin watcher's 'end'/'quit') already
                # triggered shutdown - nothing left to reopen against.
                break

            # The window was closed voluntarily (X button). Reachy is still
            # running normally in the background - vision, voice, everything.
            print("\nCorrective captioning window closed - Reachy keeps running normally in the background.")
            print("(Regular vision captioning is unaffected either way.)")
            print("Type 'open' to reopen it, or 'end'/'quit' to stop everything (or Ctrl+C).\n")

            reopen_signal.clear()
            reopen_requested["flag"] = False
            reopen_signal.wait()  # woken by the stdin watcher (open/end/quit) or backend shutdown

            if shutdown_event.is_set():
                break
            # If woken for any other reason without an explicit 'open',
            # loop back defensively rather than exiting unexpectedly.
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    _wait_for_backend_shutdown(backend_thread, handoff)


def _run_without_gui(args: argparse.Namespace) -> None:
    """Same reliable-shutdown mechanism as the GUI path.

    BUG FIX: previously ran asyncio directly (blocking) on the main
    thread, which meant if the backend ever got stuck (e.g. a hung network
    call), there was NO free thread available to print a fallback summary
    - the whole process would just hang. Restructured to mirror the GUI
    path's architecture: asyncio runs on a background thread, freeing the
    main thread to apply the same bounded-wait-with-fallback-summary
    behavior via _wait_for_backend_shutdown."""
    handoff: dict = {"tracker": None, "calibration_store": None, "error": None}
    ready_event = threading.Event()
    shutdown_event = threading.Event()

    def run_async_backend() -> None:
        try:
            asyncio.run(
                async_main(args, gui_handoff=handoff, gui_ready_event=ready_event, external_stop_event=shutdown_event)
            )
        except Exception as exc:
            handoff["error"] = exc
        finally:
            shutdown_event.set()
            ready_event.set()

    def stdin_watcher() -> None:
        while not shutdown_event.is_set():
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                shutdown_event.set()
                break
            if line.strip().lower() in ("end", "quit", "exit"):
                shutdown_event.set()
                break

    def handle_sigint(signum, frame) -> None:
        shutdown_event.set()
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGINT, handle_sigint)

    backend_thread = threading.Thread(target=run_async_backend, daemon=True, name="AsyncioBackend")
    backend_thread.start()

    stdin_thread = threading.Thread(target=stdin_watcher, daemon=True, name="StdinWatcher")
    stdin_thread.start()

    print("Type 'end' and press Enter at any time to stop and see the session summary (or press Ctrl+C).\n")

    try:
        shutdown_event.wait()  # main thread free to wait here - not stuck deep inside asyncio anymore
    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    _wait_for_backend_shutdown(backend_thread, handoff)


def main() -> None:

    enable_windows_ansi_support()
    args = parse_args()

    use_gui = (not args.no_gui_feedback) and _TKINTER_AVAILABLE
    if not args.no_gui_feedback and not _TKINTER_AVAILABLE:
        print("[gui] tkinter not available - desktop feedback window disabled. Use voice feedback instead.")

    if not use_gui:
        _run_without_gui(args)
        return

    _run_with_gui_on_main_thread(args)


if __name__ == "__main__":
    main()