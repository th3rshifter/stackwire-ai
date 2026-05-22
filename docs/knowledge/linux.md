# Linux

## Overview

Linux troubleshooting usually starts by separating the problem layer: process, CPU/RAM, disk, filesystem, permissions, network, service manager or logs. Use small targeted commands first, then go deeper only when the symptom points to that layer.

## Processes and resources

`ps`, `top`, `htop`, `vmstat`, `free` and `uptime` answer different questions: current process list, live resource usage, memory state, CPU/I/O pressure and load average.

| Command | What it shows |
| --- | --- |
| `top` / `htop` | CPU, RAM, swap and processes in real time |
| `ps aux \| grep <proc>` | Find a process by name |
| `free -m` | Memory: total, used, free, cache/buffers in MB |
| `vmstat 1 5` | CPU, memory and I/O stats every second, 5 samples |
| `uptime` | Load average for 1, 5 and 15 minutes |

### Process states

Linux process state matters during incidents:

- `R` - running or runnable on CPU.
- `S` - interruptible sleep, usually normal waiting.
- `D` - uninterruptible sleep, often blocked on disk, NFS or kernel I/O.
- `Z` - zombie. The process already exited, but its parent has not reaped it.
- `T` - stopped or traced.

Zombie processes need the parent process to reap them. `kill -9` does not remove a zombie directly.

Useful checks:

```bash
ps aux
top
vmstat 1 5
journalctl -p err -b
```

## Filesystems and disk usage

`df` reads filesystem-level usage, while `du` walks directory entries. They can disagree because of deleted files still held open by a process, different mount points, reserved blocks, overlay filesystems or permissions.

| Command | Purpose |
| --- | --- |
| `df -hT` | Human-readable filesystem usage and filesystem type |
| `df -i` | inode usage. If it reaches 100%, new files may not be created |
| `du -h --max-depth=1 /var/log \| sort -h` | Directory sizes under `/var/log`, sorted by size |
| `iostat -x 1` | Disk latency and utilization, including `await` and `%util` |
| `iotop` | Processes generating disk I/O |
| `dmesg -T \| tail -50` | Kernel messages, disk, driver or NFS errors |
| `lsblk` | Block devices and mount points |
| `mount \| column -t` | Mounted filesystems and mount options |
| `stat <path>` | Metadata for a file or directory |

Useful checks:

```bash
df -h
du -sh /path
lsof +L1
findmnt
```

### Common disk and inode symptoms

| Symptom | Diagnostic command | Typical fix |
| --- | --- | --- |
| `df -i` is 100%, files cannot be created | `df -i` | Remove many small files from temp, cache or logs |
| `df` shows used space, but `du` shows less | `lsof +L1` | Restart or reload the process holding deleted files |
| Processes stuck in D state | `ps aux \| awk '$8 ~ /D/ {print}'` | Check disk, NFS, driver or storage latency with `dmesg` and `iostat` |
| Filesystem is full | `du -h --max-depth=1 / \| sort -h` | Clean logs, cache, old artifacts, container layers or unused data |

## Permissions: DAC and MAC

### DAC: classic Unix permissions

DAC is discretionary access control. The owner and permissions on the file decide access.

```bash
chmod 644 file          # rw-r--r--, common for config files
chmod 755 dir           # rwxr-xr-x, common for executable dirs
chown user:group file   # change owner and group
chmod +x script.sh      # add executable bit
```

### MAC: SELinux and similar policy systems

MAC is mandatory access control. System policy can block access even for root if the security context does not match.

| Command | What it does |
| --- | --- |
| `getenforce` | SELinux mode: Enforcing, Permissive or Disabled |
| `ls -Z file` | Show security context: user, role, type, level |
| `chcon -t httpd_sys_content_t /var/www` | Temporarily change context type |
| `semanage fcontext -a -t TYPE '/path(/.*)?'` | Create persistent context rule |
| `restorecon -Rv /path` | Apply policy-defined contexts |
| `audit2allow -a` | Suggest policy rules from audit logs, use carefully |

## Networking

`ss` is the modern replacement for many `netstat` use cases. `ip route`, `ip addr`, `dig`, `curl` and `tcpdump` help separate DNS, routing, firewall and application problems.

| Command | Purpose |
| --- | --- |
| `ss -tulpn` | Listening TCP/UDP ports and owning processes |
| `netstat -tulpn` | Legacy alternative to `ss`, still common on old systems |
| `lsof -i :8080` | Process using a specific port |
| `tcpdump -i eth0 port 80` | Capture traffic on interface and port |
| `curl -v https://example.com` | Verbose HTTP/TLS request diagnostics |
| `curl -d 'k=v' -X POST http://url` | POST request with data |
| `ip route` / `ip addr` | Routing table and IP addresses |
| `dig example.com +short` | DNS lookup for A, AAAA, MX, TXT and other records |
| `traceroute example.com` / `mtr example.com` | Network path and packet loss diagnostics |

Useful checks:

```bash
ss -tulpn
ip addr
ip route
dig example.com +short
curl -v https://example.com
```

## cron and systemd

### cron

Cron uses five time fields followed by a command:

```cron
# minute hour day_of_month month day_of_week command
0 2 * * * /usr/bin/backup.sh      # every day at 02:00
*/5 * * * * /opt/check.sh         # every 5 minutes
```

Useful commands:

```bash
crontab -e          # edit current user's crontab
crontab -l          # list current user's cron jobs
ls -l /etc/cron.d/  # system cron files
```

### systemd

| Command | Action |
| --- | --- |
| `systemctl status <svc>` | Service state and recent log lines |
| `systemctl start <svc>` | Start a service |
| `systemctl stop <svc>` | Stop a service |
| `systemctl restart <svc>` | Restart a service |
| `systemctl enable --now <svc>` | Enable autostart and start immediately |
| `systemctl daemon-reload` | Reload unit files after changes |
| `journalctl -u <svc> -f` | Follow service logs in real time |
| `journalctl --since '1 hour ago'` | Logs for the last hour |
| `journalctl -p err -b` | Errors from the current boot |

## Bash basics

Use safe defaults for scripts that should fail fast.

```bash
#!/usr/bin/env bash
set -euo pipefail # exit on error, unset variable or failed pipeline

VAR="value"
echo "Hello, ${VAR}"

if [ -f /etc/hosts ]; then
  echo "hosts found"
fi

for item in a b c; do
  echo "$item"
done

check_service() {
  systemctl is-active --quiet "$1" && echo "OK" || echo "FAIL"
}

check_service nginx

command > out.txt 2>&1 # stdout and stderr to file
command 2>/dev/null    # suppress stderr
```

## Logging and diagnostics

| What to check | Command or location |
| --- | --- |
| systemd service logs | `journalctl -u nginx -f --no-pager` |
| system log | `/var/log/syslog` or `/var/log/messages` |
| auth and sudo actions | `/var/log/auth.log` or `/var/log/secure` |
| kernel and disk errors | `dmesg -T \| tail -50` |
| errors in a file | `grep -i error /var/log/nginx/error.log` |
| last lines of a log | `tail -100f /var/log/app.log` |
| archived gzip logs | `zcat /var/log/nginx/access.log.1.gz \| grep 500` |

## Practical troubleshooting flow

1. Define the symptom: service down, high latency, full disk, DNS failure, permission denied or process stuck.
2. Check service state and logs with `systemctl status` and `journalctl`.
3. Check resource pressure with `top`, `free`, `df`, `iostat` and `vmstat`.
4. Check network path with `ss`, `ip route`, `dig`, `curl` and `tcpdump`.
5. Fix the most likely operational cause first, then verify with the same command that exposed the problem.