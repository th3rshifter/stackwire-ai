# Networking

## TCP and UDP
TCP is connection-oriented and provides ordering, retransmission and flow control. UDP is datagram-oriented and leaves reliability to the application. Use TCP for HTTP, databases and SSH; use UDP when latency matters or the protocol handles reliability itself, for example DNS, QUIC or streaming.

## TLS and mTLS
TLS authenticates the server and encrypts the connection. mTLS authenticates both client and server with certificates, so it is common in service mesh, internal APIs and zero-trust style networks.

## Debug flow
A practical network debug path is: name resolution, route, local listener, firewall/security group, upstream health, application logs. Avoid jumping directly to Kubernetes or proxy config unless the symptom points there.
