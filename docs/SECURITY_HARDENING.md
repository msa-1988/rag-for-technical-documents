# Security Hardening

This project is designed to be safe by default for local use.

## Current security posture

The safer defaults are:
- Streamlit binds to `127.0.0.1`
- `enableCORS` remains enabled
- `enableXsrfProtection` remains enabled
- the Ollama endpoint stays on `http://localhost:11434`
- personal PDFs are not tracked in Git
- any public demo can be password-gated

## Main risks this hardening is addressing

1. exposing the app publicly without access control
2. leaving a public tunnel open after a demo
3. leaking internal error details to strangers
4. accidentally publishing personal documents
5. letting one-click demo scripts bypass security checks

## What changed

### 1. Localhost-only default

File:
- `.streamlit/config.toml`

This makes the app bind to:
- `127.0.0.1`

That means:
- no public internet exposure by default
- no LAN-wide exposure by default

### 2. Password gate for public sharing

File:
- `app/streamlit_app.py`

Behavior:
- if `APP_ACCESS_PASSWORD` is set, the app requires it before use
- sessions expire automatically after inactivity
- the session can be manually locked again

### 3. Safer public demo script

File:
- `scripts/start_secure_public_demo.sh`

Behavior:
- refuses to start if `.env` is missing
- refuses to start if `APP_ACCESS_PASSWORD` is not set properly
- keeps the app on `127.0.0.1`
- uses a temporary tunnel only when explicitly requested

### 4. Less revealing errors

Behavior:
- detailed internal error messages are hidden by default
- if needed for debugging, set:

```bash
APP_SHOW_DETAILED_ERRORS=true
```

### 5. Personal data protection

Behavior:
- `data/input/` is ignored by Git
- README and public docs describe only the public paper workflow

## How to use the project safely

### Local only

Recommended default:

```bash
./scripts/run_safe_local.sh
```

### Temporary public demo

Only if needed:

1. set a real `APP_ACCESS_PASSWORD` in `.env`
2. start the app with:

```bash
./scripts/start_secure_public_demo.sh
```

3. stop it as soon as the demo is over

## What not to do

Do not:
- leave a public tunnel running in the background
- share a tunnel URL without a password
- commit `.env`
- place personal documents in tracked repo paths
- disable CORS or XSRF protections for convenience

## Best next security upgrade

If you ever want a more stable public deployment, use:
- a hosted inference layer
- a proper reverse proxy or access-control layer
- named Cloudflare tunnels with access rules instead of quick tunnels
