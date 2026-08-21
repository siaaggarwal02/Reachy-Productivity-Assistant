Reachy — A Physical AI Accountability Partner

Reachy is a desktop robot (running on the Reachy Mini platform) that watches your workspace and speaks up — turning knowing what you need to do into actually doing it.

Phone reminders don't work because they live on the exact device causing the distraction in the first place. Reachy doesn't. It sits on your desk, sees what you're actually doing through camera-based AI vision, and responds to real behavior in real time — not a notification you can swipe away and forget about thirty seconds later.

The Problem

Staying focused has never been harder. Apps like TikTok and Instagram are engineered to shrink attention spans down to the length of a reel, and for students juggling coursework, jobs, internship applications, research, sports, and a social life on top of it all, that constant pull adds up fast. Most of these students know exactly what they need to do — the problem isn't a lack of intention, it's that their attention is being pulled in five directions at once. Deadlines slip, backlogs grow, and the growing pile of undone work only makes it harder to start.

Why Reachy Is Different

Every productivity app in existence lives on your phone or your laptop — the same devices doing the distracting. Reachy is deliberately not there. It's a physical presence in the room, watching through its camera and listening through its microphone, so a nudge to put the phone down actually requires you to put the phone down, not just tap "dismiss."

It's also not trying to be your drill sergeant. When you hit a focused stretch of work, Reachy dances. When you stay hydrated, it celebrates. Accountability here is built to feel supportive, not punitive — the goal is to make follow-through easier, not to make you feel worse about not following through yet.

What Reachy Actually Does
Watches, and knows the difference between looking and doing. Reachy's vision pipeline (Claude AI vision analyzing frames from Reachy's camera) tracks whether you're actively holding and using your phone — not just whether one happens to be visible on the desk — and speaks up if that goes on too long. The first nudge is gentle and points you toward something specific on your task list; if it keeps happening, the tone firms up without becoming repetitive or naggy.
Tracks hydration honestly. It only logs a drink when you're actually drinking — not when a water bottle happens to be sitting nearby — and won't double-count sips a few seconds apart as two separate instances.
Celebrates focus. Sustained, uninterrupted work at your desk earns recurring praise (and a little dance) roughly every 30 minutes, not just once at the start.
Manages your calendar and tasks by voice. Ask Reachy to add, edit, delete, or list events and tasks — no app-switching required. It reminds you 30 minutes and 5 minutes before anything starts, and automatically rolls incomplete tasks forward to the next day until they're actually done.
Answers questions on the spot, the same way you'd talk to Alexa or Google Home — weather, quick lookups, "what's on my calendar today," whatever comes up.
Learns your specific setup over time. A built-in correction system (voice or a small desktop confirm/correct window) lets you tell Reachy when it got something wrong, and it factors that feedback into future observations — without ever forcing you to use it.
How It Works

Reachy's "brain" isn't onboard the robot — Reachy Mini itself is camera, microphone, speaker, and motors, nothing more. All of the actual understanding is powered by Claude (Anthropic):

Vision: every few seconds, a frame from Reachy's camera is sent to Claude, which returns a structured read on what's happening — is a person present, are they on their phone, are they drinking, are they working — plus a plain-language description.
Voice: speech is transcribed, matched against a wake word ("hey Reachy"), and handed to Claude for a real conversational response — including tool access to Google Calendar and Google Tasks, so it can actually take action, not just talk about it.
Everything runs locally on your machine except the actual API calls to Anthropic and Google — your data isn't being routed through some third-party service in between.
Tech Stack
Layer	What's used
Robot platform	Reachy Mini (Pollen Robotics / Hugging Face)
Vision & conversation	Claude (Anthropic API)
Speech-to-text	Google Speech Recognition
Text-to-speech	pyttsx3
Calendar & tasks	Google Calendar API, Google Tasks API
Desktop feedback UI	Tkinter
Core language	Python 3.12, asyncio
Getting Started
Prerequisites
A Reachy Mini, assembled and connected via USB
uv for Python environment management
Python 3.12 (installed automatically by uv in the steps below)
An Anthropic API key
(Optional, for calendar/task features) a Google Cloud project with the Calendar and Tasks APIs enabled
Setup
powershell
# Install uv and Python 3.12
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv python install 3.12 --default

# Create and activate a virtual environment
uv venv reachy_mini_env --python 3.12
reachy_mini_env\Scripts\activate

# Install the Reachy Mini SDK (includes the daemon)
uv pip install "reachy-mini"

# Install the remaining dependencies
uv pip install numpy pyttsx3 requests SpeechRecognition anthropic pillow scipy `
    google-api-python-client google-auth-httplib2 google-auth-oauthlib pyaudio

Set your API key (per terminal session):

powershell
$env:ANTHROPIC_API_KEY = "your-key-here"

(Optional) If you want calendar/task features, follow Google's OAuth setup to generate a credentials.json, and place it in the same folder as the script. Everything else works fine without it — those features just stay quietly disabled.

Running Reachy

Reachy Mini needs two terminals running at once:

Terminal 1 — the daemon (leave this running the whole session):

powershell
reachy_mini_env\Scripts\activate
reachy-mini-daemon

Terminal 2 — Reachy itself:

powershell
reachy_mini_env\Scripts\activate
$env:ANTHROPIC_API_KEY = "your-key-here"
python reachyRun.py --location "Your City, ST"

Say "hey Reachy" to start talking, or "hey Reachy, good morning" for a spoken morning briefing. To end a session and see a full summary of everything it observed, type end and press Enter (or Ctrl+C as a backup).

A Note on Privacy

Everything Reachy observes stays local — vision analysis is sent to Claude for interpretation the same way any AI vision request works, but nothing is stored or shared beyond what's needed to run the assistant. credentials.json, token.json, and the vision-calibration history file all contain personal account access or usage data and are intentionally excluded from version control (see .gitignore).

Built on Reachy Mini by Pollen Robotics & Hugging Face, powered by Claude from Anthropic.
