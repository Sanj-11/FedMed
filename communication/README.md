FedMed — Distributed Communication
Overview

This module handles communication between the central federated learning server and hospital clients in the FedMed project.

Responsibilities
Implement gRPC-based communication between the server and hospital nodes.
Secure communication using TLS certificates.
Handle communication failures, timeouts, and disconnected clients.
Support reliable exchange of federated learning instructions and model updates.
Integrate with the Flower federated learning framework.
Planned Components
Server-client communication interfaces
gRPC service definitions
TLS configuration
Error handling and timeout management
Communication testing
Privacy Considerations

Raw patient data must remain at the hospital clients. Communication should transfer only the information required by the federated learning workflow, subject to the team's privacy and security design.

Status

Initial module documentation. Implementation will follow the agreed team architecture.