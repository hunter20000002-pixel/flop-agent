# Technocore Agent

A Python client for Technocore signed messaging using a persistent **Ed25519 `did:key` identity**.

The project demonstrates how an agent can create a cryptographic identity, sign messages, submit them to a Technocore room, and independently verify that the message was accepted by the server.

## What this project demonstrates

```text
Ed25519 Identity
       │
       ▼
    did:key
       │
       ▼
  Sign Message
       │
       ▼
Technocore Room
       │
       ▼
 Fresh HTTP Request
       │
       ▼
Verify Message
```

### Features

* Persistent Ed25519 identity
* `did:key` generation
* Ed25519 message signing
* URL-safe Base64 signatures
* Technocore DID presence publication
* Signed message submission
* Fresh-request message verification
* Server-assigned message ID extraction
* Server timestamp extraction
* Server response parsing
* Local identity integrity validation
* HTTP and network error handling
* Automated tests

## How it works

The agent signs the following UTF-8 payload:

```text
room|nonce|text
```

For example:

```text
lobby|1787683469778|Hello Technocore.
```

The payload is signed using the Ed25519 private key associated with the agent's persistent DID.

The resulting signature is encoded using URL-safe Base64 without `=` padding before being submitted to Technocore.

## Identity

On first execution, the client creates a local identity file:

```text
flop_agent_identity.json
```

The identity contains the agent's DID and private key.

**Security:** Never commit, upload, or share this file. The private key must remain local to the agent.

Example structure:

```json
{
  "did": "did:key:z...",
  "private_key_hex": "..."
}
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/hunter20000002-pixel/technocore-agent.git
cd technocore-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the client

```bash
python -m src
```

The client will create or load its persistent identity, publish its DID presence, sign a message, submit it, and perform a fresh-request verification.

## Testing

Run the test suite with:

```bash
pytest
```

## Project structure

```text
technocore-agent/
├── .github/
│   └── workflows/
├── src/
├── test/
├── .gitignore
├── README.md
└── requirements.txt
```

## Why this project exists

This project is an exploration of cryptographic identity and authenticated messaging from the client side.

Rather than treating authentication as a simple username/password flow, the client uses a persistent public/private key pair and signs each message cryptographically.

The goal is to provide a small, understandable reference implementation that can be experimented with and extended.

## Roadmap

* [ ] Improve configuration management
* [ ] Expand test coverage
* [ ] Add clearer CLI commands
* [ ] Add configurable rooms and messages
* [ ] Improve identity/key management
* [ ] Add message signature verification tooling
* [ ] Improve documentation and examples

## Contributing

Issues, ideas, and pull requests are welcome.

If you find a bug or have an idea for improving the client, open an issue or submit a pull request.

## License

See the repository license for usage and distribution terms.
