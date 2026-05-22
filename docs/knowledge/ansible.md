# Ansible

## Core model

**Ansible** is an agentless IaC and configuration-management tool. It connects to managed hosts, usually over **SSH**, and converges them to the desired state using **modules**.

Main building blocks:

- **Inventory** - list of managed hosts and groups.
- **Playbook** - YAML file describing what should be done.
- **Play** - maps hosts to tasks, variables and privileges.
- **Task** - one desired action, usually through an Ansible module.
- **Module** - idempotent unit of work, for example `package`, `template`, `service`, `copy`.
- **Role** - reusable project layout with tasks, handlers, templates, files and defaults.
- **Collection** - package format for roles, modules and plugins.

## Common commands

| Command | Purpose |
| --- | --- |
| `ansible all -i inv.ini -m ping` | Check connectivity to all hosts from inventory. |
| `ansible-playbook -i inv.ini play.yml` | Run a playbook. |
| `ansible-playbook -i inv.ini play.yml --check --diff` | Dry-run with planned changes. |
| `ansible-playbook -i inv.ini play.yml --limit host1` | Run only against one host or group. |
| `ansible-playbook -i inv.ini play.yml -t nginx,deploy` | Run only tasks with selected tags. |
| `ansible-playbook -i inv.ini play.yml -e 'var=val'` | Pass extra variables with highest precedence. |
| `ansible-vault create secrets.yml` | Create encrypted secrets file. |
| `ansible-galaxy role install namespace.role` | Install a role from Ansible Galaxy. |

## Project and role structure

A common structure is to keep inventories, playbooks and roles separate:

```text
project/
├── ansible.cfg                 # defaults: inventory, forks, callbacks, interpreter
├── inventory/
│   ├── hosts.ini               # static inventory
│   ├── group_vars/
│   │   ├── all.yml             # variables for all hosts
│   │   └── webservers.yml      # variables for webservers group
│   └── host_vars/
│       └── server1.yml         # host-specific variables
├── playbooks/
│   └── deploy.yml              # entry playbook
└── roles/
    └── nginx/
        ├── tasks/main.yml      # main role tasks
        ├── handlers/main.yml   # service reload/restart handlers
        ├── templates/          # Jinja2 templates, usually *.j2
        ├── files/              # static files for copy module
        ├── defaults/main.yml   # low-priority role defaults
        ├── vars/main.yml       # high-priority role vars
        └── meta/main.yml       # role metadata and dependencies
```

## Variable precedence

Variable precedence is one of the most common Ansible pitfalls. A practical simplified order from lower to higher priority:

```text
defaults < group_vars < host_vars < play vars < role vars < extra vars (-e)
```

**Extra vars** passed with `-e` almost always win. Use them carefully because they can override values that normally come from inventory or roles.

## Playbook example

This playbook installs **Nginx**, renders virtual host configs from a **Jinja2** template and reloads the service only when configuration changes.

```yaml
---
- name: Install and configure Nginx
  hosts: webservers
  become: true # run tasks with privilege escalation

  vars:
    nginx_port: 80
    vhosts:
      - name: site1
        domain: site1.example.com
      - name: site2
        domain: site2.example.com

  tasks:
    - name: Install nginx package
      ansible.builtin.package:
        name: nginx
        state: present # idempotent: install only if missing

    - name: Render virtual host configs
      ansible.builtin.template:
        src: vhost.conf.j2
        dest: "/etc/nginx/sites-available/{{ item.name }}.conf"
        mode: "0644"
      loop: "{{ vhosts }}" # create one config per vhost
      notify: Reload nginx

    - name: Ensure nginx is enabled and running in prod
      ansible.builtin.service:
        name: nginx
        state: started
        enabled: true
      when: ansible_env.ENVIRONMENT | default('dev') == 'prod'

  handlers:
    - name: Reload nginx
      ansible.builtin.service:
        name: nginx
        state: reloaded # reload only when notified by changed template task
```

## Jinja2 template example

Example `roles/nginx/templates/vhost.conf.j2`:

```nginx
server {
    listen {{ nginx_port }};
    server_name {{ item.domain }};

    location / {
        proxy_pass http://127.0.0.1:8080; # backend app endpoint
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Idempotency

**Idempotency** means repeated runs should converge to the same result without unnecessary changes. Prefer Ansible modules over raw shell commands.

Good examples:

```yaml
- name: Ensure package is installed
  ansible.builtin.package:
    name: nginx
    state: present

- name: Ensure service is running
  ansible.builtin.service:
    name: nginx
    state: started
    enabled: true
```

Risky pattern:

```yaml
- name: Remove directory with shell
  ansible.builtin.shell: rm -rf /opt/app
```

Use modules such as `file`, `copy`, `template`, `package`, `service`, `user`, `group`, `lineinfile` and `blockinfile` when possible.

## Error handling and task control

| Pattern | Purpose |
| --- | --- |
| `register: result` | Save task result into a variable. |
| `failed_when: result.rc != 0` | Define custom failure condition. |
| `changed_when: false` | Mark read-only command as unchanged. |
| `ignore_errors: true` | Continue after failure. Use sparingly. |
| `block / rescue / always` | Try/catch/finally style error handling. |
| `when:` | Conditional task execution. |
| `loop:` | Iterate over a list. |
| `notify:` | Trigger a handler only if task changed something. |

Example:

```yaml
- name: Check nginx config and reload safely
  block:
    - name: Validate nginx config
      ansible.builtin.command: nginx -t
      register: nginx_test
      changed_when: false # validation does not change host state

    - name: Reload nginx
      ansible.builtin.service:
        name: nginx
        state: reloaded
      when: nginx_test.rc == 0

  rescue:
    - name: Show validation output
      ansible.builtin.debug:
        var: nginx_test.stderr

  always:
    - name: Print completion message
      ansible.builtin.debug:
        msg: "Nginx validation flow finished"
```

## Handlers

Handlers are tasks that run only when notified by a changed task. They are commonly used to reload or restart services after config changes.

```yaml
- name: Deploy nginx config
  ansible.builtin.template:
    src: nginx.conf.j2
    dest: /etc/nginx/nginx.conf
  notify: Restart nginx

handlers:
  - name: Restart nginx
    ansible.builtin.service:
      name: nginx
      state: restarted
```

## Ansible Vault

Use **Ansible Vault** for secrets that must be stored in Git, such as passwords, API tokens and private variables.

```bash
ansible-vault create secrets.yml
ansible-vault edit secrets.yml
ansible-playbook -i inv.ini play.yml --ask-vault-pass
```

Practical notes:

- Do not put plaintext secrets into `group_vars` or role defaults.
- Keep secret variable names stable, but encrypt their values.
- In CI/CD, prefer vault password files or external secret stores with restricted access.

## Troubleshooting

| Symptom | Checks | Typical fix |
| --- | --- | --- |
| Host unreachable | `ansible all -i inv.ini -m ping -vvv` | Check SSH user, key, inventory, firewall, Python interpreter. |
| Task always reports changed | Review module usage, `changed_when`, template content | Make task idempotent or set correct changed condition. |
| Variable has unexpected value | `ansible-inventory --list`, debug variable | Check precedence: group_vars, host_vars, role vars, extra vars. |
| Handler did not run | Check whether notifying task changed | Handler runs only after a changed task. |
| Template renders wrong value | Use `debug`, inspect vars and Jinja2 filters | Fix variable scope, defaults or inventory values. |

Useful debug commands:

```bash
ansible-inventory -i inventory/hosts.ini --list
ansible-playbook -i inventory/hosts.ini playbooks/deploy.yml --check --diff
ansible-playbook -i inventory/hosts.ini playbooks/deploy.yml -vvv
```

## Production notes

- Keep roles small and focused: one role should manage one component or responsibility.
- Use `--check --diff` before risky changes, but remember not all modules support perfect dry-run.
- Prefer immutable artifacts and explicit versions for packages, containers and templates.
- Use tags for controlled partial runs, but do not rely on tags as the only deployment safety mechanism.
- Keep inventory and variables readable; unclear variable hierarchy is a common source of Ansible incidents.
