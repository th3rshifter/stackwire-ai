from dataclasses import dataclass


@dataclass(frozen=True)
class DomainProfile:
    triggers: tuple[str, ...]
    required_concepts: tuple[str, ...]
    forbidden_concepts: tuple[str, ...]
    component_model: str
    dangerous_confusions: tuple[str, ...]


DOMAIN_PROFILES: dict[str, DomainProfile] = {
    "kubernetes": DomainProfile(
        triggers=(
            "kubernetes",
            "k8s",
            "kube",
            "kuber",
            "kubectl",
            "pod",
            "deployment",
            "statefulset",
            "daemonset",
            "replicaset",
            "service",
            "ingress",
            "gateway",
            "httproute",
            "configmap",
            "secret",
            "namespace",
            "pvc",
            "pv",
            "storageclass",
            "crashloopbackoff",
            "imagepullbackoff",
            "errimagepull",
            "imagepolicybackoff",
            "helm",
        ),
        required_concepts=("Kubernetes API object", "controller/reconciliation", "Pod", "desired state"),
        forbidden_concepts=("generic deployment process", "inode-first explanation", "Docker BuildKit cache"),
        component_model=(
            "control plane: kube-apiserver, etcd, scheduler, controller-manager; "
            "data plane: kubelet, container runtime, CNI/kube-proxy, Pods on worker nodes"
        ),
        dangerous_confusions=(
            "Deployment means Kubernetes workload controller, not generic deployment process, when question also mentions StatefulSet, Pod, Service, Kubernetes, kube, kuber.",
            "StatefulSet means Kubernetes workload controller with stable identity, ordinal names, stable network identity, Headless Service and stable PVC, not generic stateful service.",
            "Ingress means Kubernetes API object plus Ingress Controller. It is not itself the reverse proxy process. Controller/data plane proxy such as Nginx/Envoy/Traefik handles traffic.",
            "Gateway can mean Gateway API, Istio IngressGateway or generic network gateway. If context says Kubernetes, explain Gateway API vs Ingress. If context says Istio, explain Istio IngressGateway.",
            "ImagePolicyBackOff is suspicious. Standard Kubernetes statuses are ImagePullBackOff and ErrImagePull. Do not invent a controller mechanism. Say likely meant ImagePullBackOff unless explicitly about admission/image policy.",
        ),
    ),
    "docker": DomainProfile(
        triggers=("docker", "dockerfile", "container", "image", "compose", "buildkit", "registry", "layer"),
        required_concepts=("image", "container", "layer", "registry"),
        forbidden_concepts=("Kubernetes Ingress", "PromQL", "inode-first explanation"),
        component_model="build/runtime: Dockerfile/build context/layers/image registry; runtime: container process, namespaces, cgroups, volumes, networks",
        dangerous_confusions=(
            "Docker image is immutable build artifact; container is runtime process from image.",
            "Dockerfile instructions describe build steps; they are not shell script lines executed at container start except CMD/ENTRYPOINT.",
        ),
    ),
    "git": DomainProfile(
        triggers=("git", "commit", "branch", "merge", "rebase", "reflog", "stash", "checkout", "pull", "push", "remote"),
        required_concepts=("commit/snapshot", "branch/ref", "working tree/index", "history"),
        forbidden_concepts=("Kubernetes Ingress", "PromQL", "Docker BuildKit cache"),
        component_model="working tree, index/staging area, local repository objects, refs/branches/tags, remotes",
        dangerous_confusions=(
            "Branch is a movable ref to a commit, not a copy of the repository.",
            "Reflog is local history of ref movements and is useful for recovery after reset/rebase/checkout mistakes.",
        ),
    ),
    "linux_fs": DomainProfile(
        triggers=("df", "du", "inode", "lsof", "filesystem", "mount", "/var/log", "deleted file", "open file", "disk", "fsck"),
        required_concepts=("filesystem", "mount point", "open file/process", "space usage"),
        forbidden_concepts=("Kubernetes Ingress", "BuildKit", "PromQL"),
        component_model="VFS/filesystem, mount points, block device, inode metadata, open file descriptors held by processes",
        dangerous_confusions=(
            "df vs du common causes: deleted open files, different mount points, reserved blocks, overlay/container layers, permissions. Do not start with inode unless question mentions inode or df -i.",
        ),
    ),
    "linux_process": DomainProfile(
        triggers=("ps", "top", "htop", "kill", "signal", "strace", "d state", "zombie", "process", "pid", "oom"),
        required_concepts=("process", "kernel state", "signal", "parent/child process"),
        forbidden_concepts=("Ingress Controller", "PromQL", "Terraform state"),
        component_model="kernel scheduler/process table, process states, signals, parent reaping, /proc process metadata",
        dangerous_confusions=(
            "D state means uninterruptible sleep, usually blocked kernel I/O. Do not answer about netstat/ss unless question explicitly mentions sockets/ports.",
            "Zombie process is not fixed by kill -9 to zombie; parent must reap it or restart parent.",
        ),
    ),
    "linux_network": DomainProfile(
        triggers=("ss", "netstat", "tcpdump", "ip route", "ip addr", "iptables", "nftables", "dns", "dig", "curl", "port", "socket", "tcp", "udp"),
        required_concepts=("socket/connection", "port", "route", "DNS/firewall when relevant"),
        forbidden_concepts=("inode-first explanation", "Kubernetes controller", "Terraform state"),
        component_model="network interfaces, routing table, sockets, conntrack/firewall, DNS resolver, application process",
        dangerous_confusions=(
            "Network command questions should stay in sockets/routes/DNS/firewall unless the user explicitly asks about filesystem or Kubernetes.",
        ),
    ),
    "service_mesh": DomainProfile(
        triggers=("service mesh", "servish mesh", "istio", "linkerd", "consul connect", "envoy", "sidecar", "mtls", "virtualservice", "destinationrule"),
        required_concepts=("service mesh", "control plane", "data plane", "proxy/sidecar", "mTLS", "traffic policy", "telemetry"),
        forbidden_concepts=("Servish mesh", "generic StatefulSet explanation", "generic deployment process"),
        component_model=(
            "control plane: config, identity/certs, policy and discovery distribution; "
            "data plane: sidecar/proxy/Envoy/eBPF datapath that carries service-to-service traffic"
        ),
        dangerous_confusions=(
            "Service mesh is concept, not “Servish mesh”. Examples: Istio, Linkerd, Consul.",
            "Control plane distributes config, identity, policy and discovery.",
            "Data plane is proxy/sidecar/Envoy/eBPF datapath carrying traffic.",
        ),
    ),
    "ci_cd": DomainProfile(
        triggers=("ci/cd", "pipeline", "runner", "gitlab ci", "github actions", "jenkins", "job", "stage", "artifact", "cache"),
        required_concepts=("pipeline", "job/stage", "runner/agent", "artifact/cache"),
        forbidden_concepts=("inode-first explanation", "Kubernetes Ingress", "PromQL"),
        component_model="control/config: pipeline definition, jobs, stages, variables; execution: runner/agent, workspace, artifacts/cache, registry",
        dangerous_confusions=(
            "Pipeline is orchestration definition/execution graph; runner is executor that runs jobs.",
        ),
    ),
    "iac": DomainProfile(
        triggers=("terraform", "opentofu", "tfstate", ".tf", "provider", "resource", "ansible", "playbook", "inventory", "role"),
        required_concepts=("desired state/config", "provider/module or playbook/task", "state/idempotency"),
        forbidden_concepts=("Kubernetes Ingress", "PromQL", "inode-first explanation"),
        component_model="definition files, variables, provider/modules or inventory/playbooks, state/idempotent execution, target infrastructure",
        dangerous_confusions=(
            "Terraform state is source of mapping between config and real resources; Ansible idempotency comes from modules/tasks.",
        ),
    ),
    "observability": DomainProfile(
        triggers=("prometheus", "promql", "grafana", "alert", "alertmanager", "slo", "sli", "burn rate", "metric", "label", "лог", "trace", "opentelemetry"),
        required_concepts=("metric/log/trace", "label/tag", "query", "alert/SLO when relevant"),
        forbidden_concepts=("Kubernetes Ingress", "inode-first explanation", "generic deployment process"),
        component_model="sources/exporters/instrumentation, collector/scrape, storage/index, query language, dashboards and alerting",
        dangerous_confusions=(
            "In PromQL, metric name and label are different. A label alone is not a time series.",
            "Burn rate is normally SLO/error budget consumption ratio, not “speed of CPU decreasing”.",
            "If user says “по лейблу”, produce PromQL with label selector against a metric, not label as metric.",
        ),
    ),
    "database": DomainProfile(
        triggers=("postgres", "postgresql", "mysql", "redis", "mongodb", "clickhouse", "replication", "backup", "wal", "index", "query"),
        required_concepts=("storage engine/data", "connection/query", "replication/backup when relevant"),
        forbidden_concepts=("Kubernetes Ingress", "PromQL label as metric", "Docker BuildKit cache"),
        component_model="client connections, query executor, storage/cache/WAL or persistence, replication/backup, monitoring",
        dangerous_confusions=(
            "Database replication/backup are data-safety mechanisms, not generic horizontal scaling by itself.",
        ),
    ),
    "web_proxy": DomainProfile(
        triggers=("nginx", "haproxy", "apache", "reverse proxy", "load balancer", "upstream", "server block", "location", "tls termination"),
        required_concepts=("HTTP/reverse proxy", "server/location/upstream", "load balancing", "TLS termination"),
        forbidden_concepts=("Kubernetes StatefulSet", "PromQL", "inode-first explanation"),
        component_model="master/worker processes, event loop, listeners, server/location routing, upstream pools, TLS and logs",
        dangerous_confusions=(
            "Nginx has master/worker process model, event loop, server/location/upstream config blocks, reverse proxy/load balancing/TLS termination.",
        ),
    ),
    "security": DomainProfile(
        triggers=("tls", "mtls", "rbac", "vault", "secret", "certificate", "sast", "scan", "policy", "admission", "oauth", "jwt"),
        required_concepts=("identity/authentication", "authorization/policy", "secret/certificate lifecycle"),
        forbidden_concepts=("PromQL label as metric", "inode-first explanation", "generic Deployment"),
        component_model="identity, policy engine, secret/certificate storage, enforcement point, audit/logging",
        dangerous_confusions=(
            "Authentication proves identity; authorization decides allowed action.",
        ),
    ),
    "messaging": DomainProfile(
        triggers=("kafka", "rabbitmq", "queue", "topic", "partition", "consumer group", "offset", "exchange", "broker"),
        required_concepts=("broker", "topic/queue", "producer/consumer", "offset/ack when relevant"),
        forbidden_concepts=("Kubernetes Ingress", "inode-first explanation", "PromQL label as metric"),
        component_model="brokers, producers, consumers, topics/queues, partitions/exchanges, offsets/acks and retention",
        dangerous_confusions=(
            "Kafka topic/partition/offset model differs from RabbitMQ queue/exchange/routing-key model.",
        ),
    ),
    "generic_software": DomainProfile(
        triggers=("api", "service", "application", "library", "framework", "architecture", "algorithm", "protocol"),
        required_concepts=("definition", "mechanism", "tradeoff when relevant"),
        forbidden_concepts=(),
        component_model="components depend on the specific software system; identify input/output, runtime, storage and integration points",
        dangerous_confusions=(
            "If a term is also a platform object, use surrounding context before choosing generic software meaning.",
        ),
    ),
}


INFRASTRUCTURE_DOMAINS = frozenset(
    {
        "kubernetes",
        "docker",
        "git",
        "linux_fs",
        "linux_process",
        "linux_network",
        "service_mesh",
        "ci_cd",
        "iac",
        "observability",
        "database",
        "web_proxy",
        "security",
        "messaging",
    }
)
