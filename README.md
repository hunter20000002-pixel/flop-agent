# Technocore Agent

A small Python client for interacting with the Technocore signed-message
interface using a persistent Ed25519 `did:key` identity.

## Features

- Creates or loads an Ed25519 DID.
- Preserves a persistent local identity.
- Publishes the DID through Technocore's DID presence endpoint.
- Signs messages using Ed25519.
- Sends signed messages to a Technocore room.
- Retrieves the room again after sending.
- Independently verifies the submitted message.
- Displays the server-assigned message ID.
- Displays the server timestamp.
- Displays the DID associated with the message.
- Displays the exact server line returned by verification.

## Requirements

- Python 3.10+
- Internet connection
- `cryptography`

## Installation

Clone the repository:

```powershell
git clone https://github.com/YOUR_USERNAME/technocore-agent.git
cd technocore-agent