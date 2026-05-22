# Docker

## Image and container
An image is an immutable build artifact made of layers. A container is a runtime process created from an image, isolated by namespaces and cgroups, with optional volumes and networks.

## Dockerfile
Dockerfile instructions describe build-time layers, except `CMD` and `ENTRYPOINT`, which define the default runtime command. Keep build cache stable by copying dependency files before frequently changing source files.

## Compose
Compose describes multiple containers, networks and volumes for local or simple deployments. It is not the same as Kubernetes orchestration, but it can model dependencies and service wiring.
