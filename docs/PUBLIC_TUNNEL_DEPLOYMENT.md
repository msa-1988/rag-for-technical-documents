# Public Tunnel Deployment

This project is currently deployed publicly through a `Cloudflare quick tunnel`.

## Why this method was used

The app depends on:
- a local Ollama runtime
- local models
- local GPU-backed inference

That makes it excellent for development and learning, but it also means:
- it is not directly deployable to a static host
- it is not directly deployable to Streamlit Community Cloud without a hosted inference layer

So the fastest honest public deployment is:

`run the Streamlit app locally and expose it through a temporary public tunnel`

## How it works

1. start Streamlit locally
2. point Cloudflare Tunnel to the local port
3. share the generated `trycloudflare.com` URL

## Important limitation

This URL is:
- public
- useful for demos
- temporary

It stays live only while:
- the Streamlit app is running
- the tunnel process is running
- the local machine stays online

## Recommended next deployment step

For a stronger portfolio deployment later, create:

1. a hosted inference layer
2. a hosted web app
3. a provider abstraction so local and hosted modes both work

## Good interview framing

`For the first public deployment, I used a temporary Cloudflare tunnel so I could demo the real local-GPU version of the app without prematurely redesigning the inference architecture.`
