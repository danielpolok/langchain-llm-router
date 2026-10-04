# Security policy

## Supported versions

Security fixes go into the latest release. Until 1.0, that is the newest `0.x` minor version.

## Reporting a vulnerability

Please report it privately through
[GitHub's vulnerability reporting](https://github.com/danielpolok/langchain-model-router/security/advisories/new),
not in a public issue. Include the package version, a minimal example that shows the problem, and
what an attacker could do with it. You will get a reply within a week, and a fix or a plan once
the report is confirmed.

## What is in scope

`langchain-model-router` runs inside your application. It holds no credentials, opens no network
connections of its own and calls only the chat models and embeddings you give it. Reports about
the routes' providers belong with those providers, and reports about LangChain itself with
[LangChain](https://github.com/langchain-ai/langchain/security).

A strategy that routes on what a message says is a policy, not a security boundary. A user who
controls the message can word it to reach a different route. If a request must never reach a
particular model, for example because of where data may go, enforce that outside the strategy,
or force the route for those requests.
