# FLOP Agent

A cryptographically identified, network-agnostic autonomous AI agent being built to participate in the Flop Network's decentralized inference ecosystem, with a modular architecture for agent execution, inference, tools, and verifiable results.

The current implementation provides persistent Ed25519 identity, `did:key` identity, signed communication, and a Technocore client. Technocore is an existing independent communication/identity foundation used by this project; it is not a component of the Flop Network.

The project is now evolving toward a full autonomous-agent runtime designed to use Flop as a first-class decentralized compute and inference provider while remaining independent of any single inference network.

The long-term goal is to build an agent that can receive a real-world task, plan and execute the work, use tools and decentralized inference, communicate with other agents, and produce results that can be independently verified.

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
                 Inference Layer
                        │
                        ▼
                Verifiable Output
```

The architecture is intentionally modular. The core V0.2 agent layers are now implemented as provider-independent building blocks. Persistent memory, concrete inference providers, and Flop integration remain future work. The agent runtime is designed not to depend on a specific inference provider or compute network, with Flop intended as the first-class decentralized inference/compute integration.

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

These capabilities form the identity and communication foundation on which the autonomous-agent layers are built.

### Autonomous-Agent Foundation

* Task model and lifecycle
* Execution plan model
* Task planner
* Agent runtime
* Structured execution results
* Tool abstraction
* Tool registry
* Inference provider abstraction
* Automated tests covering the implemented components

These components establish the core interfaces needed to add concrete tools, inference providers, memory, and verifiable execution without coupling the agent to a single network.

## V0.2 Vision

The next development phase expands the autonomous-agent core while preserving the existing cryptographic identity and communication foundation. The core is network-agnostic, with Flop implemented as a dedicated inference provider rather than hard-coded into the agent runtime.

### Agent Core — Implemented Foundation

* [x] Task representation and lifecycle
* [x] Task planning and decomposition foundation
* [x] Autonomous execution runtime foundation
* [x] Structured execution results
* [ ] Persistent agent memory
* [x] Tool abstraction and registration
* [x] Modular inference provider abstraction

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

Flop is the intended first major decentralized compute/inference provider for this project. The Flop integration is planned and will be implemented against the actual network/testnet interface when it is available and sufficiently specified. It is not part of the current implementation.

The same agent runtime should eventually be able to switch between providers based on task requirements such as capability, latency, cost, availability, or privacy.

## How It Works Today

The current implementation establishes a persistent cryptographic identity and uses the existing Technocore interface for signed communication. This is the project's current foundation; it should not be interpreted as an existing integration between Technocore and the Flop Network.

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
git clone https://github.com/hunter20000002-pixel/flop-agent.git
cd flop-agent
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

The current project remains intentionally small, but the V0.2 agent foundation is now represented in the source tree.

```text
flop-agent/
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

The implemented V0.2 architecture now includes:

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

* [x] Task model
* [x] Task lifecycle
* [x] Planner
* [x] Agent runtime foundation
* [x] Structured agent results

### Phase 3 — Agent Capabilities

* [ ] Persistent memory
* [x] Tool abstraction
* [x] Tool registry
* [x] Inference provider abstraction
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

The roadmap is intentionally iterative. Checked items describe functionality already implemented in the repository; unchecked items describe planned work. Protocol-specific components will be implemented against finalized specifications rather than assumptions about unreleased network functionality.

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
