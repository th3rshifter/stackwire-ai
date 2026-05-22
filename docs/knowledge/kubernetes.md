# Kubernetes

## Architecture
Kubernetes is a container orchestration platform built around declarative state. A user or controller writes the desired state into the Kubernetes API. The control plane stores that state and controllers reconcile the real cluster toward it.

## Control plane and data plane
Control plane is the management layer. It receives API requests, stores cluster state, schedules workloads and runs reconciliation loops.

Data plane is the execution layer. It runs Pods, containers and the networking path on worker nodes.

| Component | Plane | Purpose |
| --- | --- | --- |
| kube-apiserver | Control plane | Main API entry point for `kubectl`, controllers and internal components |
| etcd | Control plane | Distributed key-value store for cluster state |
| kube-scheduler | Control plane | Chooses a suitable Node for a new Pod |
| kube-controller-manager | Control plane | Runs controllers that reconcile actual state to desired state |
| Worker Node | Data plane | Machine where Pods and containers actually run |
| kubelet | Data plane | Node agent that creates Pods through the container runtime and reports status to the API server |
| container runtime | Data plane | Runs containers, usually `containerd` or CRI-O |
| kube-proxy | Data plane | Implements Service networking with iptables, IPVS or eBPF-based alternatives |
| CNI plugin | Data plane | Provides Pod networking, IP allocation, routing and sometimes NetworkPolicy |

Short model: the control plane decides and stores desired state, while the data plane runs workloads and moves traffic.

## Core objects

| Object | Purpose |
| --- | --- |
| Pod | Smallest runnable unit. One or more containers sharing IP, network namespace and volumes |
| Deployment | Manages stateless replicas through ReplicaSet, rolling update and rollback |
| ReplicaSet | Keeps the desired number of matching Pods running |
| StatefulSet | Runs stateful workloads with stable Pod identity, ordinal names and usually stable PVCs |
| DaemonSet | Runs one Pod on every matching Node, often for agents, logs, monitoring or networking |
| Job | Runs a task to completion |
| CronJob | Creates Jobs on a schedule |
| Service | Stable virtual IP and DNS name for a changing set of Pods |
| Ingress | HTTP/HTTPS routing rules by host, path and TLS to backend Services |
| ConfigMap | Non-sensitive configuration as environment variables or mounted files |
| Secret | Sensitive configuration. Base64 is encoding, not encryption |
| PersistentVolume | Storage resource available to the cluster |
| PersistentVolumeClaim | Request for storage by a workload |
| StorageClass | Defines dynamic volume provisioning behavior |
| Namespace | Logical isolation boundary for namespaced resources |
| ServiceAccount | Identity used by Pods when calling the Kubernetes API |
| RBAC | Authorization model for Kubernetes API access |
| HPA | Horizontal Pod Autoscaler based on CPU, memory or custom metrics |

## Workload controllers
Deployment, StatefulSet, DaemonSet and Job are Kubernetes API objects. The user writes desired state into the API server. It is stored in etcd, and controllers reconcile real cluster state toward that desired state.

### Deployment
Deployment is used for stateless workloads. It manages ReplicaSets and replaceable Pods.

Key points:

- Pods are interchangeable.
- Pod names and identities are not stable.
- Rollout and rollback are managed through ReplicaSets.
- Good fit for web applications, APIs and workers without stable identity requirements.

Useful commands:

```bash
kubectl rollout status deployment/<name>
kubectl rollout history deployment/<name>
kubectl rollout undo deployment/<name>
kubectl scale deployment/<name> --replicas=3
```

### StatefulSet
StatefulSet is used for workloads that need stable identity or stable storage.

Key points:

- Stable Pod names with ordinal indexes, for example `db-0`, `db-1`, `db-2`.
- Stable network identity, usually through a Headless Service.
- Stable storage, usually one PVC per Pod.
- Ordered rollout and ordered termination.
- Good fit for databases, queues and clustered systems that care about identity.

Main difference: Deployment treats Pods as replaceable replicas. StatefulSet gives each Pod stable identity and usually stable storage.

## Services
Service gives stable access to changing Pods. Pods can be recreated and receive new IPs, but Service keeps a stable DNS name and virtual IP.

| Type | Purpose |
| --- | --- |
| ClusterIP | Internal-only access inside the cluster |
| NodePort | Opens a port on every Node |
| LoadBalancer | Requests an external load balancer from the cloud or provider integration |
| Headless Service | No virtual IP. Used for direct Pod DNS records, often with StatefulSet |

Useful checks:

```bash
kubectl get svc
kubectl get endpoints <service>
kubectl get endpointslice
```

If a Service has no endpoints, traffic will not reach Pods. Common causes are wrong labels, wrong selectors, Pods not Ready or backend Pods not existing.

## Ingress and Gateway
Ingress is an API object for HTTP/HTTPS routing rules: host, path, TLS and backend Service. It does not proxy traffic by itself.

Ingress Controller watches Ingress objects and configures the real data-plane proxy such as Nginx, Envoy, HAProxy or Traefik.

### Ingress model
Control plane:

- Ingress object.
- Kubernetes API.
- Ingress Controller watching API changes.

Data plane:

- Real proxy process, for example Nginx, Envoy, HAProxy or Traefik.
- It accepts HTTP/HTTPS traffic and routes it to Kubernetes Services.

### Gateway API
Gateway API separates infrastructure entry points from application routing rules.

| Object | Purpose |
| --- | --- |
| GatewayClass | Type of Gateway implementation |
| Gateway | Listener, port, protocol and TLS entry point |
| HTTPRoute | HTTP routing rules to backend Services |

Gateway API is more expressive than classic Ingress and is often better for shared platform routing, multi-team ownership and advanced traffic management.

### Istio IngressGateway
Do not confuse Gateway API with Istio IngressGateway.

Istio IngressGateway is usually an Envoy-based gateway in a service mesh. It receives external traffic and applies Istio routing, policy and mTLS behavior through Istio configuration.

## Pod troubleshooting
For Pod issues, start from status, events and logs. Do not start with `kubectl exec` if the container is constantly crashing because it may not stay alive long enough.

Useful commands:

```bash
kubectl get pod <pod> -o wide
kubectl describe pod <pod>
kubectl logs <pod> -c <container>
kubectl logs <pod> -c <container> --previous
kubectl get events --sort-by=.lastTimestamp
```

### Common Pod states

| Status | Meaning | First checks |
| --- | --- | --- |
| CrashLoopBackOff | Container starts and exits repeatedly | `kubectl logs --previous`, `kubectl describe pod`, exit code, command, args, probes |
| ImagePullBackOff | kubelet cannot pull the image | image name, tag, registry access, imagePullSecret |
| ErrImagePull | Initial image pull failure | same checks as ImagePullBackOff |
| Pending | Pod cannot be scheduled or volume is not ready | events, node resources, taints, tolerations, nodeSelector, PVC |
| OOMKilled | Container exceeded memory limit | `kubectl describe pod`, memory limits, application memory usage |
| Terminating stuck | Pod deletion is blocked | finalizers, volume detach, node unreachable |
| ContainerCreating | Container creation is stuck | image pull, CNI, volume mount, Secrets, ConfigMaps |

### CrashLoopBackOff checklist
Typical causes:

- wrong command or args;
- missing Secret or ConfigMap;
- application exits because a dependency is unavailable;
- failed liveness or startup probe;
- permission issue;
- incompatible image or runtime config;
- resource limit, especially memory.

Useful checks:

```bash
kubectl describe pod <pod>
kubectl logs <pod> --previous
kubectl get events --sort-by=.lastTimestamp
kubectl get pod <pod> -o yaml
```

## Probes
Kubernetes probes help decide whether a container is alive, ready for traffic or successfully started.

| Probe | Purpose |
| --- | --- |
| livenessProbe | Restarts the container if it is stuck or unhealthy |
| readinessProbe | Controls whether the Pod is included in Service endpoints |
| startupProbe | Gives slow-starting applications more time before liveness checks begin |

Common pitfall: using livenessProbe too aggressively can cause endless restarts under temporary load.

## Resources and scheduling
Requests and limits affect scheduling and runtime behavior.

| Field | Meaning |
| --- | --- |
| requests.cpu / requests.memory | Used by the scheduler to place the Pod on a Node |
| limits.cpu | CPU throttling limit |
| limits.memory | Hard memory limit. Exceeding it can cause OOMKilled |

Scheduling can also be affected by:

- nodeSelector;
- node affinity;
- pod affinity and anti-affinity;
- taints and tolerations;
- topology spread constraints;
- resource pressure on Nodes.

Useful commands:

```bash
kubectl describe pod <pod>
kubectl describe node <node>
kubectl top pod
kubectl top node
```

## ConfigMap and Secret
ConfigMap stores non-sensitive configuration. Secret stores sensitive data, but Kubernetes Secret values are base64-encoded by default. Base64 is not encryption.

Common production practices:

- avoid putting secrets into images or plain manifests;
- enable encryption at rest for Kubernetes Secrets;
- use External Secrets Operator, Vault, SOPS or cloud secret managers;
- restart or reload workloads when configuration changes, depending on how config is consumed.

## RBAC and ServiceAccount
Pods access the Kubernetes API using ServiceAccount credentials. RBAC controls what that identity can do.

| Object | Purpose |
| --- | --- |
| ServiceAccount | Identity used by a Pod |
| Role | Permissions inside one Namespace |
| ClusterRole | Cluster-wide or reusable permissions |
| RoleBinding | Binds Role or ClusterRole to a subject in one Namespace |
| ClusterRoleBinding | Cluster-wide binding |

Principle: give the minimum permissions required for the workload.

## Useful kubectl commands

```bash
kubectl get nodes -o wide
kubectl get pods -A -o wide
kubectl describe pod <pod>
kubectl logs <pod> --previous
kubectl get events --sort-by=.lastTimestamp
kubectl get svc,ep,ingress -n <namespace>
kubectl rollout status deployment/<name>
kubectl rollout undo deployment/<name>
kubectl top pod
kubectl top node
```

## Production checklist
For Kubernetes workloads, check:

- resource requests and limits;
- readiness, liveness and startup probes;
- rolling update strategy;
- logs and metrics;
- Service selectors and endpoints;
- Secrets and ConfigMaps;
- RBAC permissions;
- PodDisruptionBudget for critical workloads;
- anti-affinity or topology spread for high availability;
- image tags and registry access;
- rollback path.
