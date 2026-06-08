# MailPilot

MailPilot is a small Gmail-to-Slack approval assistant.

It watches a Gmail inbox for forwarded messages, extracts private instructions from
the forwarding text, asks Claude to draft a reply, and posts the draft to Slack for
human approval. Nothing is sent automatically unless an allowed Slack user approves
it.

## Flow

1. Forward an email to the Gmail account used by MailPilot.
2. Add a private instruction block above the forwarded message:

```text
### ANWEISUNG AN ASSISTENZ
Ziel: Freundlich absagen, aber Kontakt offenhalten.
Stil: Kurz, warm und professionell.
Modus: Freigabe in Slack.
### ENDE ANWEISUNG

[forwarded original email]
```

3. MailPilot separates the private instructions from the original email.
4. MailPilot checks whether the forwarding sender is allowed.
5. Claude creates a reply draft without exposing the private instructions.
6. The draft is posted into a Slack thread.
7. The allowed Slack user approves, rejects, creates a Gmail draft, or asks for a revision.

## Slack Approval

React to the Slack thread with:

```text
:white_check_mark: send
:memo: create Gmail draft
:x: reject
```

You can also reply in the thread with:

```text
senden
entwurf
ablehnen
```

Any other thread reply is treated as a refinement instruction, for example:

```text
Kuerzer, etwas waermer, und bitte Ende Juni vorschlagen.
```

## Installation

```bash
git clone git@github.com:flodido/mailpilot.git
cd mailpilot
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp config.example.json config.json
```

Then edit `config.json`.

## Configuration

Required local files:

- `config.json`: local configuration, copied from `config.example.json`
- `credentials.json`: Google OAuth Desktop App credentials
- `token.json`: created after the first Gmail OAuth flow
- `state.json`: runtime state, created automatically

These files are ignored by git because this repository is public.

Important config keys:

- `working_directory`: directory used when calling Claude
- `slack_channel_id`: Slack channel where drafts are posted
- `slack_allowed_user_id`: Slack user allowed to approve or refine drafts
- `slack_bot_token_env`: environment variable containing the Slack bot token
- `claude_command`: path to the Claude CLI
- `allowed_forwarders`: email addresses allowed to forward messages to MailPilot

## Gmail OAuth

1. Create an OAuth Desktop App credential in Google Cloud.
2. Save it as `credentials.json` in the repo directory.
3. Run:

```bash
python email_assistant.py --auth-only
```

For headless machines:

```bash
python email_assistant.py --auth-only --no-browser
```

## Run Once

```bash
python email_assistant.py --once
```

## LaunchAgent

On macOS, install the polling LaunchAgent after a successful manual test:

```bash
./install-launchagent.sh
```

The agent runs every two minutes.
