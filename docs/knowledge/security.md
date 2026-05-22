# Security

## Authentication and authorization
Authentication proves identity. Authorization decides whether that identity may perform an action. RBAC policies should be scoped to least privilege and tested against real service accounts or users.

## Secrets and certificates
Secrets need storage, rotation, audit and access boundaries. Certificates add identity and encryption but require lifecycle management: issuing, renewal, revocation and trust roots.

## Admission and policy
Admission webhooks and policy engines can reject Kubernetes objects before they are persisted. If a policy blocks deploys, check webhook availability, policy logs and admission events.
