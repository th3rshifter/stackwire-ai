# Networking

## TCP and UDP
TCP is connection-oriented and provides ordering, retransmission and flow control. UDP is datagram-oriented and leaves reliability to the application. Use TCP for HTTP, databases and SSH; use UDP when latency matters or the protocol handles reliability itself, for example DNS, QUIC or streaming.

## TLS and mTLS
TLS authenticates the server and encrypts the connection. mTLS authenticates both client and server with certificates, so it is common in service mesh, internal APIs and zero-trust style networks.

## Debug flow
A practical network debug path is: name resolution, route, local listener, firewall/security group, upstream health, application logs. Avoid jumping directly to Kubernetes or proxy config unless the symptom points there.

## Comparing requests between servers
When the same request behaves differently on two servers, compare one controlled request end to end: method, scheme, host, path, query string, request body, headers, cookies/auth, source IP, DNS target, TLS/SNI, status code, response body and latency.

Use a correlation id if the application supports it. Otherwise compare proxy access logs and application logs by timestamp, client IP and request path. Check whether both servers run the same app version, environment variables, feature flags, upstream target, routing/proxy config and firewall/security group rules.

Useful commands include `curl -sv`, `dig`, `ss -lntp`, proxy access logs and application logs.
