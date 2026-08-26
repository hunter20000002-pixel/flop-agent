# Technocore Autonomous Agent

A cryptographically identified autonomous AI agent project built on persistent `did:key` identity, signed communication, and a modular architecture for future agent execution, inference, tools, and verifiable results.

The current implementation is the cryptographic identity and Technocore communication foundation. The project is now evolving toward a full autonomous-agent runtime with task planning, memory, tools, modular inference providers, and verifiable execution.

The long-term goal is to build an agent that can receive a real-world task, plan and execute the work, use external tools and inference, communicate with other agents, and produce results that can be independently verified.

## Architecture

```text
                    USER TASK
                        │
                        ▼
                ┌───────────────┐
                │ Agent Runtime │
                └───────┬───────┘
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
           Planner    Memory     Tools
              │         │         │
              └─────────┼─────────┘
                        ▼
               Inference Provider
                        │
                        ▼
                 Agent Result
                        │
                        ▼
              Cryptographic Identity
                   Ed25519 / DID
                        │
                        ▼
                 Technocore Layer
                        │
                        ▼
                Verifiable Output
```

The architecture is intentionally modular. The autonomous-agent layers shown above are the planned V0.2 direction; they are not yet implemented in the current repository. The future agent runtime should not depend on a specific inference provider or compute network.

## Current Foundation

The current implementation provides the cryptographic and Technocore communication foundation:

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

These capabilities form the existing identity and communication foundation on which the autonomous-agent layers will be built.

## V0.2 Vision

The next development phase introduces the autonomous-agent core while preserving the existing identity and Technocore client.

### Agent Core

* Task representation and lifecycle
* Task planning and decomposition
* Autonomous execution runtime
* Structured agent results
* Persistent agent memory
* Tool abstraction and registration
* Modular inference providers

### Verifiable Agent

The agent will progressively connect its future execution layer to its cryptographic identity so that important outputs can be associated with a persistent agent identity and independently verified.

### Decentralized Inference

Inference is designed as a provider abstraction rather than a hard-coded dependency.

```text
                 InferenceProvider
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
       Local         Remote         Future
      Provider      Provider        Providers
                                      │
                                      ▼
                               Flop Provider
```

The Flop integration is planned as a future provider and will be implemented against the actual network/testnet interface when that interface is available and sufficiently specified. It is not part of the current implementation.

## How It Works Today

The current client establishes a persistent identity and uses it to sign messages submitted through the Technocore interface.

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

The signed payload currently follows the structure:

```text
room|nonce|text
```

The payload is signed using the Ed25519 private key associated with the persistent identity.

## Identity

On first execution, the client creates or loads a local identity file:

```text
flop_agent_identity.json
```

The identity contains the agent's DID and private key material.

**Never commit, upload, or share this file. The private key must remain local to the agent.**

Example structure:

```json
{
  "did": "did:key:z...",
  "private_key_hex": "..."
}
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/hunter20000002-pixel/technocore-agent.git
cd technocore-agent
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the current client

```bash
python -m src
```

The current implementation creates or loads its persistent identity, publishes its DID presence, signs a message, submits it, and performs fresh-request verification.

## Testing

Run the test suite with:

```bash
pytest
```

## Project Structure

The current project is intentionally small. The structure will expand as the autonomous-agent layer is implemented.

```text
technocore-agent/
├── .github/
│   └── workflows/
├── src/
│   ├── __init__.py
│   ├── client.py
│   ├── config.py
│   ├── identity.py
│   ├── main.py
│   └── parser.py
├── test/
│   ├── test_identity.py
│   └── test_parser.py
├── .gitignore
├── LICENSE.txt
├── README.md
└── requirements.txt
```

The planned V0.2 architecture will introduce modules such as:

```text
src/
├── agent/
│   ├── task.py
│   ├── planner.py
│   ├── runtime.py
│   ├── memory.py
│   └── result.py
│
├── inference/
│   ├── base.py
│   └── providers/
│
└── tools/
    ├── base.py
    └── registry.py
```

## Development Roadmap

### Phase 1 — Cryptographic Foundation

* [x] Persistent Ed25519 identity
* [x] `did:key` identity
* [x] Signed messages
* [x] Technocore communication
* [x] Message verification
* [x] Automated tests

### Phase 2 — Agent Core

* [ ] Task model
* [ ] Task lifecycle
* [ ] Planner
* [ ] Agent runtime
* [ ] Structured agent results

### Phase 3 — Agent Capabilities

* [ ] Persistent memory
* [ ] Tool abstraction
* [ ] Tool registry
* [ ] Inference provider abstraction
* [ ] Autonomous execution loop

### Phase 4 — Verifiable Agent

* [ ] Signed agent results
* [ ] Verifiable execution records
* [ ] Agent-to-agent communication
* [ ] Execution history

### Phase 5 — Decentralized Compute

* [ ] Decentralized inference provider
* [ ] Flop network adapter
* [ ] Testnet integration
* [ ] Compute/inference telemetry

The roadmap is intentionally iterative. Items marked as planned describe the intended architecture, not functionality that is already present in the repository. Protocol-specific components will be implemented against finalized specifications rather than assumptions about unreleased network functionality.

## Design Principles

### Identity First

The project uses persistent cryptographic identity as a foundation rather than relying exclusively on conventional application-level authentication.

### Modular Intelligence

The agent runtime should not be tightly coupled to a single AI model or inference provider.

### Verifiable Execution

Where practical, agent actions and results should be structured so that their provenance can be inspected and verified.

### Real Utility

The project is intended to evolve into a useful autonomous-agent system rather than a collection of artificial activity or benchmark-only demonstrations.

### Open Architecture

The system should be able to interact with different inference providers, tools, and decentralized compute networks through well-defined interfaces.

## Why This Project Exists

AI agents are becoming increasingly capable, but many agent systems still treat identity, communication, execution, and inference as separate concerns.

This project explores what happens when those components are combined into a single architecture:

```text
Identity
   +
Planning
   +
Memory
   +
Tools
   +
Inference
   +
Verification
   =
Autonomous Agent
```

The goal is to build the system incrementally and keep each layer understandable, testable, and replaceable.

## Contributing

Issues, ideas, experiments, and pull requests are welcome.

If you find a bug or have an idea for improving the architecture, open an issue or submit a pull request.

## License

See [LICENSE.txt](LICENSE.txt) for usage and distribution terms.
