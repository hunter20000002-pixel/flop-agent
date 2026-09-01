# FLOP Agent

> A cryptographically identified autonomous-agent framework for planning, tool execution, inference, memory, communication, and verifiable execution.

**FLOP Agent** is an open-source Python project exploring what an autonomous AI agent looks like when **identity, planning, execution, tools, inference, memory, communication, and verification** are treated as parts of the same system.

The project began as a cryptographically identified client for the Technocore communication environment and has evolved into a modular autonomous-agent runtime.

The long-term direction is to make the agent capable of receiving a real-world task, decomposing it into executable work, selecting tools and inference providers, maintaining execution history and memory, communicating through authenticated channels, and producing results whose provenance can be independently inspected.

FLOP Agent is designed to remain **provider- and network-agnostic at the core**, while being developed with the **FLOP decentralized inference ecosystem** as an intended future integration target.

> **Current status:** The autonomous-agent architecture is under active development. The cryptographic identity and Technocore communication foundation are implemented, while the agent runtime, planning, tools, inference abstractions, memory, autonomy, and execution infrastructure are being progressively integrated into a complete autonomous loop.

---

## Why FLOP Agent?

Most AI agents are primarily concerned with generating a response.

FLOP Agent explores a different model:

```text
             ┌──────────────────────┐
             │      USER TASK       │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │     AGENT CONTEXT    │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │       PLANNER        │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │   EXECUTION PLAN    │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │    AGENT RUNTIME     │
             └──────┬───────┬───────┘
                    │       │
              ┌─────▼──┐ ┌──▼──────────┐
              │  Tools │ │  Inference  │
              └─────┬──┘ └──────┬──────┘
                    │            │
                    └──────┬─────┘
                           ▼
                 ┌──────────────────┐
                 │ Execution Result │
                 └────────┬─────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
          History      Memory      Decision
              │           │           │
              └───────────┼───────────┘
                          ▼
                 ┌──────────────────┐
                 │ Autonomous Loop  │
                 └────────┬─────────┘
                          │
                    ┌─────┴─────┐
                    ▼           ▼
                 COMPLETE     REPLAN
```

The objective is not to create a collection of disconnected AI features.

The objective is to build a **coherent execution system** in which an agent can:

* understand a task
* construct an execution plan
* execute individual steps
* use registered tools
* request inference through an abstract provider interface
* record what happened
* maintain useful state and memory
* evaluate execution outcomes
* retry or replan when appropriate
* stop when execution should terminate
* associate important actions and results with a persistent cryptographic identity

---

# Core Architecture

FLOP Agent is organized around several independent layers.

```text
                    ┌──────────────────┐
                    │      Task        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Agent Context    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Planner      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Execution Plan   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Agent Runtime   │
                    └───────┬───┬──────┘
                            │   │
                  ┌─────────┘   └──────────┐
                  ▼                        ▼
          ┌──────────────┐        ┌────────────────┐
          │ Tool System  │        │ Inference      │
          │              │        │ Provider       │
          └──────┬───────┘        └───────┬────────┘
                 │                        │
                 └───────────┬────────────┘
                             ▼
                    ┌──────────────────┐
                    │ Execution Result │
                    └────────┬─────────┘
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
             History       Memory      Decision
                │            │            │
                └────────────┼────────────┘
                             ▼
                    ┌──────────────────┐
                    │ Autonomy Control │
                    └────────┬─────────┘
                             │
                 ┌───────────┼───────────┐
                 ▼           ▼           ▼
              COMPLETE      RETRY       REPLAN
```

The architecture intentionally keeps these responsibilities separate.

A planner should not be responsible for executing tools.

A tool should not know which inference network is being used.

The runtime should orchestrate execution rather than contain every capability itself.

The inference layer should be replaceable without rewriting the agent core.

---

# Implemented Foundations

## Cryptographic Identity

FLOP Agent uses a persistent Ed25519 identity as the foundation for agent-level authentication.

Implemented capabilities include:

* persistent Ed25519 key material
* `did:key` identity generation
* identity loading
* identity integrity validation
* Ed25519 message signing
* URL-safe Base64 signatures
* authenticated message construction

The identity is intended to give the agent a persistent cryptographic presence rather than treating every execution as an anonymous process.

### Local identity

The current client stores its identity locally:

```text
flop_agent_identity.json
```

Example structure:

```json
{
  "did": "did:key:z...",
  "private_key_hex": "..."
}
```

**Never commit, upload, or share this file.**

The private key is the agent's cryptographic identity and must remain local.

---

# Technocore Communication Foundation

The original project was built around the Technocore signed-message interface.

The current implementation supports:

* DID presence publication
* signed message submission
* nonce-based signing
* fresh-request verification
* server-assigned message ID extraction
* server timestamp extraction
* response parsing
* HTTP/network error handling
* authenticated communication using the persistent agent identity

The current signed payload follows the form:

```text
room|nonce|text
```

The payload is signed using the Ed25519 private key associated with the agent's persistent identity.

### Important distinction

Technocore is an **independent communication and identity environment used by this project**.

It should not be interpreted as a component of the FLOP Network.

Likewise, the current Technocore implementation should **not** be interpreted as a completed FLOP Network integration.

The relationship is intentional:

```text
              FLOP Agent
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
 Cryptographic         Agent Runtime
   Identity                 │
        │             ┌─────┴─────┐
        ▼             ▼           ▼
  Technocore       Tools      Inference
 Communication
                              │
                              ▼
                       Future FLOP Provider
```

---

# Autonomous-Agent Runtime

The project has evolved beyond a simple signed-message client.

The agent core now contains the foundations required for structured task execution.

## Task Model

Tasks have an explicit lifecycle rather than being treated as arbitrary strings passed through a function.

The task model provides:

* task identity
* task description
* lifecycle state
* execution metadata
* timestamps
* structured task state

The intended lifecycle is:

```text
PENDING
   │
   ▼
PLANNING
   │
   ▼
READY
   │
   ▼
RUNNING
   │
   ├───────────────┐
   ▼               ▼
COMPLETED        FAILED
```

This gives the runtime an explicit model of where a task is in its execution lifecycle.

---

# Planning and Execution Plans

The planner converts a task into a structured `ExecutionPlan`.

An execution plan contains ordered execution steps rather than leaving execution as an unstructured model response.

Conceptually:

```text
Task
 │
 ▼
Planner
 │
 ▼
ExecutionPlan
 │
 ├── Step 1
 ├── Step 2
 ├── Step 3
 └── ...
```

Each execution step can contain:

* step identity
* execution order
* description
* optional tool assignment
* tool arguments

This creates an explicit boundary between:

**what the agent intends to do**

and

**what the runtime actually executes.**

That separation is fundamental to the project's eventual verifiability model.

---

# Agent Runtime

The runtime is the execution engine of the agent architecture.

Its responsibility is to take a validated task and execution plan and coordinate execution.

The runtime provides the foundation for:

* plan validation
* ordered step execution
* tool dispatch
* inference-provider interaction
* structured results
* execution history
* execution limits
* retry behavior
* replanning
* termination decisions
* contextual execution

The runtime is intentionally independent of any specific AI model or decentralized compute provider.

This means the same execution engine can eventually operate with:

```text
Local Inference
       │
Remote Inference
       │
Decentralized Inference
       │
       ▼
   Same Runtime
```

---

# Tools

Tools are modeled as explicit capabilities rather than being embedded directly into the planner or runtime.

The tool layer provides:

* tool abstraction
* tool metadata
* tool execution interfaces
* tool registration
* tool lookup
* controlled dispatch through a registry

Conceptually:

```text
                  Tool Registry
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      Calculator   Filesystem    Future Tools
```

This architecture makes it possible to add capabilities without rewriting the agent runtime.

The project already uses concrete built-in tooling as part of the execution architecture.

---

# Inference Provider Architecture

Inference is deliberately represented through an abstraction layer.

The agent core should not care whether reasoning is performed:

* locally
* through a remote API
* through a specialized inference service
* through decentralized compute
* through a future FLOP provider

The intended interface is conceptually:

```text
             InferenceProvider
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
     Local        Remote       Future
    Provider     Provider     Providers
                                │
                                ▼
                           FLOP Provider
```

This separation is one of the most important architectural decisions in the project.

The runtime should remain stable even as the underlying inference infrastructure changes.

---

# Memory and Context

Agent memory is being developed as a first-class component rather than treating every task as an isolated interaction.

The architecture includes memory/context infrastructure intended to support:

* relevant prior information
* execution context
* persistence
* task continuity
* integration with planning
* integration with execution history

The objective is to eventually allow the agent to use experience from previous executions when deciding how to approach future tasks.

Memory is therefore treated as part of the agent's cognitive state rather than simply as a database attached to the application.

---

# Execution History

Execution history provides a structured record of what happened during an agent run.

This is important for three reasons:

1. **Debugging** — understand why an execution succeeded or failed.
2. **Autonomy** — allow the control layer to make decisions based on previous execution outcomes.
3. **Verification** — provide the foundation for eventually producing independently inspectable execution records.

The longer-term architecture is:

```text
Task
 │
 ▼
Plan
 │
 ▼
Execution
 │
 ├── Step Result
 ├── Step Result
 ├── Step Result
 └── ...
 │
 ▼
Execution History
 │
 ▼
Decision / Verification
```

---

# Autonomy and Control

The project is moving toward an execution loop in which the agent does not simply execute a plan once and terminate.

The runtime is being designed around explicit decisions such as:

```text
                ┌──────────────┐
                │   Execute    │
                └──────┬───────┘
                       │
                       ▼
                Evaluate Result
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       COMPLETE       RETRY        REPLAN
          │            │            │
          └────────────┴────────────┘
                       │
                       ▼
                     STOP
```

This creates the foundation for genuine agent behavior:

> **observe → decide → act → evaluate → continue / retry / replan / stop**

rather than simply:

> prompt → response.

The autonomy layer is intentionally constrained by explicit policies and execution controls so that autonomous behavior remains inspectable and testable.

---

# Structured Results

Agent execution produces structured results rather than relying exclusively on raw text.

This creates a consistent interface between:

* tools
* runtime
* history
* autonomy
* future verification systems

A structured result can represent the outcome of an execution step without forcing every downstream component to parse arbitrary natural-language output.

This becomes particularly important when execution eventually needs to be independently verified.

---

# Verification and Provenance

One of the long-term goals of FLOP Agent is to connect execution provenance to cryptographic identity.

The intended model is:

```text
Agent Identity
     │
     ▼
Task
     │
     ▼
Execution Plan
     │
     ▼
Execution Steps
     │
     ▼
Results
     │
     ▼
Execution Record
     │
     ▼
Cryptographically Associated Output
```

The current project provides the identity and structured execution foundations required to move toward this model.

Full cryptographically verifiable execution records are still an area of active development.

---

# FLOP Integration

FLOP is an important part of the project's long-term direction.

The goal is **not** to hard-code FLOP throughout the agent.

Instead:

```text
              FLOP Agent Core
                    │
              InferenceProvider
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
     Local        Remote         FLOP
    Inference    Inference     Inference
```

FLOP can therefore become a first-class decentralized inference/compute provider while the agent architecture remains portable.

A future FLOP integration could allow provider selection based on factors such as:

* capability
* availability
* latency
* cost
* privacy requirements
* task requirements
* compute requirements

### Current status

**FLOP Network integration is not yet implemented.**

The project is intentionally being built against explicit interfaces rather than assuming unreleased or unspecified network behavior.

When the relevant FLOP interface is sufficiently specified, it can be implemented as a provider/adapter without restructuring the agent core.

---

# Current Development Status

| Component                                      | Status                     |
| ---------------------------------------------- | -------------------------- |
| Persistent Ed25519 identity                    | ✅ Implemented              |
| `did:key` identity                             | ✅ Implemented              |
| Signed communication                           | ✅ Implemented              |
| Technocore communication                       | ✅ Implemented              |
| Task model                                     | ✅ Implemented              |
| Task lifecycle                                 | ✅ Implemented              |
| Execution plan model                           | ✅ Implemented              |
| Task planning/decomposition foundation         | ✅ Implemented              |
| Agent runtime                                  | ✅ Implemented              |
| Structured execution results                   | ✅ Implemented              |
| Tool abstraction                               | ✅ Implemented              |
| Tool registry                                  | ✅ Implemented              |
| Built-in execution tools                       | ✅ Implemented              |
| Inference-provider abstraction                 | ✅ Implemented              |
| Execution history infrastructure               | ✅ Implemented              |
| Autonomy/control architecture                  | ✅ Implemented              |
| Memory/context infrastructure                  | 🟡 Active development      |
| Intelligent inference-driven planning          | 🟡 Active development      |
| Complete autonomous execution loop             | 🟡 Active development      |
| Cryptographically verifiable execution records | 🟡 Planned / active design |
| Agent-to-agent execution                       | 🟡 Planned                 |
| Concrete decentralized inference provider      | ⬜ Planned                  |
| FLOP provider                                  | ⬜ Planned                  |
| FLOP testnet integration                       | ⬜ Planned                  |
| Compute/inference telemetry                    | ⬜ Planned                  |

The distinction between **implemented**, **in development**, and **planned** is intentional. The repository does not claim functionality that has not actually been implemented.

---

# Project Structure

The project is organized around a separation between communication, identity, agent orchestration, tools, inference, and tests.

```text
flop-agent/
│
├── .github/
│   └── workflows/
│
├── src/
│   ├── __init__.py
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── context.py
│   │   ├── control.py
│   │   ├── decision.py
│   │   ├── history.py
│   │   ├── history_store.py
│   │   ├── loop.py
│   │   ├── main.py
│   │   ├── memory.py
│   │   ├── memory_integration.py
│   │   ├── plan.py
│   │   ├── planner.py
│   │   ├── result.py
│   │   ├── runtime.py
│   │   └── task.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── providers/
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── builtin.py
│   │   └── registry.py
│   │
│   ├── client.py
│   ├── config.py
│   ├── identity.py
│   ├── main.py
│   └── parser.py
│
├── test/
│   ├── ...
│
├── .gitignore
├── LICENSE.txt
├── README.md
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

The exact tree will continue to evolve as the agent runtime develops.

---

# Getting Started

## Requirements

* Python 3.10+
* Git
* Internet connectivity for Technocore communication
* A virtual environment is recommended

## Clone

```bash
git clone https://github.com/hunter20000002-pixel/flop-agent.git
cd flop-agent
```

## Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

For development:

```bash
pip install -r requirements-dev.txt
```

## Run the client

```bash
python -m src
```

The current communication client can create/load the persistent identity and use the configured Technocore interface for authenticated communication.

---

# Testing

Run the full test suite:

```bash
pytest
```

The project uses automated tests to protect the behavior of the cryptographic foundation and the increasingly complex agent architecture.

Tests cover areas including:

* identity
* communication
* parsing
* task lifecycle
* planning
* execution plans
* runtime behavior
* tools
* tool registration
* structured results
* inference interfaces
* memory/context
* execution history
* autonomy and control

As the architecture grows, tests are treated as part of the design rather than an afterthought.

---

# Development Roadmap

## Phase 1 — Cryptographic Foundation

* [x] Persistent Ed25519 identity
* [x] `did:key` identity
* [x] Signed messages
* [x] Technocore communication
* [x] Message verification
* [x] Identity validation
* [x] Automated testing

## Phase 2 — Agent Core

* [x] Task representation
* [x] Task lifecycle
* [x] Task planning
* [x] Execution-plan model
* [x] Agent runtime
* [x] Structured execution results
* [x] Tool abstraction
* [x] Tool registry
* [x] Inference-provider abstraction

## Phase 3 — Autonomous Execution

* [x] Execution history foundation
* [x] Context infrastructure
* [x] Memory infrastructure
* [x] Autonomy/control architecture
* [x] Decision framework
* [ ] Complete inference-driven autonomous loop
* [ ] More capable dynamic planning
* [ ] Expanded tool ecosystem

## Phase 4 — Verifiable Agent

* [ ] Cryptographically associated execution records
* [ ] Signed execution results
* [ ] Result provenance
* [ ] Agent-to-agent communication
* [ ] Independently verifiable execution traces

## Phase 5 — Decentralized Inference

* [ ] Concrete decentralized inference provider
* [ ] FLOP provider adapter
* [ ] FLOP testnet integration
* [ ] Provider selection
* [ ] Capability-aware inference routing
* [ ] Compute/inference telemetry

The roadmap is deliberately incremental. Each stage is intended to produce a working architectural layer rather than a speculative collection of features.

---

# Design Principles

## 1. Identity First

An autonomous agent should have a persistent identity that can be authenticated independently of a particular process instance.

## 2. Separate Planning from Execution

The plan is an explicit object.

The runtime executes the plan.

This separation makes execution easier to inspect, test, retry, and eventually verify.

## 3. Tools Are Capabilities

Tools should be registered and invoked through explicit interfaces rather than being hidden inside arbitrary model-generated code.

## 4. Inference Is Replaceable

The agent should not be permanently tied to a single model, API, provider, or network.

## 5. Execution Is Structured

Important actions and results should exist as structured data whenever practical.

## 6. Autonomy Must Be Controllable

An autonomous agent should have explicit policies governing when it can execute, retry, replan, complete, or stop.

## 7. Verification Matters

The system should progressively make it possible to inspect not only the final answer, but also the provenance of the execution that produced it.

## 8. Build Against Real Interfaces

Network-specific integrations should be implemented against actual specifications and interfaces rather than assumptions about future systems.

## 9. Keep the Core Network-Agnostic

FLOP is an intended decentralized inference/compute target, but the agent runtime should remain useful independently of FLOP.

---

# The End Goal

The ultimate objective is not simply to create an AI chatbot with tools.

It is to explore a more complete model of an autonomous software agent:

```text
             ┌─────────────────┐
             │  Cryptographic  │
             │     Identity    │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │      Task       │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │     Planning    │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │    Execution    │
             └────────┬────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
        Tools      Inference    Memory
          │           │           │
          └───────────┼───────────┘
                      ▼
             ┌─────────────────┐
             │     History     │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │    Decision     │
             └────────┬────────┘
                      │
              ┌───────┼────────┐
              ▼       ▼        ▼
           Complete  Retry    Replan
                      │
                      ▼
             ┌─────────────────┐
             │    Verified     │
             │     Result      │
             └─────────────────┘
```

The deeper vision is an agent that is:

**identified, autonomous, modular, tool-capable, inference-provider independent, communicative, stateful, and progressively verifiable.**

FLOP Agent is being built one layer at a time toward that system.

---

# Contributing

The project is actively evolving.

Useful contributions include:

* architecture discussions
* bug reports
* test coverage
* tool implementations
* inference-provider implementations
* memory improvements
* execution/verifiability research
* agent communication experiments
* FLOP integration work once the relevant interfaces are available

Issues and pull requests are welcome.

---

# License

See [`LICENSE.txt`](LICENSE.txt) for the project's license and distribution terms.
