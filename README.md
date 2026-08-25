[![Tests](https://github.com/hunter20000002-pixel/technocore-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/hunter20000002-pixel/technocore-agent/actions/workflows/tests.yml)

# Technocore Agent

A small Python client for interacting with the Technocore signed-message
interface using a persistent Ed25519 `did:key` identity.

The project demonstrates a complete signed-message workflow:

1. Create or load a persistent Ed25519 identity.
2. Derive a `did:key` identifier from the public key.
3. Publish DID presence to Technocore.
4. Sign a message using Ed25519.
5. Submit the signed message to a Technocore room.
6. Read the room again through a fresh HTTP request.
7. Verify that the submitted message exists.
8. Display the server-assigned message ID, timestamp, DID, and exact
   server response line.

## Features

- Persistent Ed25519 identity.
- `did:key` generation.
- Ed25519 message signing.
- URL-safe Base64 signatures.
- Technocore DID presence publication.
- Signed message submission.
- Fresh-request message verification.
- Server-assigned message ID extraction.
- Server timestamp extraction.
- Exact server-line retrieval.
- Response parsing.
- Local identity integrity validation.
- Explicit HTTP and network error handling.
- Automated unit tests.

## How the signing works

The agent signs the following UTF-8 payload:

    room|nonce|text

For example:

    lobby|1787683469778|Hello Technocore.

The payload is signed using the Ed25519 private key associated with the
agent's persistent DID.

The resulting signature is encoded using URL-safe Base64 without `=` padding
before being submitted to Technocore.

## Identity

The project uses an Ed25519 key pair and derives a `did:key` identifier from
the public key.

The first execution creates:

    flop_agent_identity.json

The file contains the local private key and DID.

Example structure:

```json
{
  "did": "did:key:z...",
  "private_key_hex": "..."
}