# Orion security model

Two questions shape it: **what can the agent touch**, and **when it is
wrong, what is the worst that happens?** The answer is five rings, each in
its own file. The gate decides *whether* a call runs. The environment
decides *where* it runs and how far its damage can reach.

```
tool call ──► [1 schema validation]  orion/tools/base.py      bad args never reach code
          ──► [2 the gate]           orion/security/gate.py   mode × tool kind × rules
                 ├─ hard-deny command rules                    never, in any mode
                 ├─ protected paths                            never or ask, in any mode
                 ├─ mode dial                                  read-only / ask / accept-edits / auto
                 └─ approver                                   human yes / no / always
          ──► [3 path jail]          Environment.resolve()    every path lands inside root
          ──► [4 operations]         orion/env/*.py           where the tool actually runs
          ──► [5 OS isolation]       sandbox / container / remote   bounds what a miss can cost
```

Pattern rules (ring 2) are the outer ring, and they are known to leak.
Isolation (ring 5) is the boundary we rely on.

## The gate

Decision order for each call; the first layer that decides wins:

1. Unknown tool → allow (the loop answers it with an error).
2. Paths: every path argument is resolved through the environment (an
   escape is a deny), then checked against `deny`, `deny_write` (for
   writes), and `ask_write` (for writes).
3. Hard-deny command rules (exec tools).
4. The ask list (exec tools).
5. The mode table.
6. If the result is "ask": session approvals, then the approver. With no
   approver (headless), an ask becomes a deny. The gate fails closed.

### Mode table

| kind \ mode | `read-only` | `ask` (default) | `accept-edits` | `auto` (alias `yolo`) |
|---|---|---|---|---|
| read | allow | allow | allow | allow |
| write | **hidden** | ask | allow | allow |
| exec | only on `SandboxEnv(writable=False)`, else **hidden** | allowlist → allow, else ask | allowlist → allow, else ask | allow |
| network | hidden unless `--web`, then allow | ask | ask | allow if `--web` |

"Hidden" means the tool is not offered to the model at all. `auto` refuses
to start on an environment without isolation unless you explicitly pass
`allow_unisolated=True`.

### Protected paths (every mode)

| level | default globs | why |
|---|---|---|
| `deny` | `.env`, `.env.*` (not `.env.example`), `*.pem`, `*.key`, `id_rsa*`, `id_ed25519*`, `~/.ssh/**`, `~/.aws/**`, `~/.netrc`, `~/.orion/**` | secrets never enter the context, so they never reach the provider |
| `deny_write` | `.git/**`, `.orion/**` | git state changes go through `git`; the agent can never rewrite its own policy, logs or sessions |
| `ask_write` | `.github/workflows/**`, `Dockerfile`, `Makefile`, `package.json`, `pyproject.toml`, `requirements*.txt`, lockfiles | supply-chain and CI surfaces get a human look |

`grep` skips `deny` files, so their contents cannot arrive through a
search. An allowlisted read-only command (`cat`, `head`, `tail`, …) that
names a `deny` path is denied too, because it would otherwise auto-run.

### Policy files

`~/.orion/policy.json` (yours) and `<root>/.orion/policy.json` (the
project's) are merged onto the defaults. Keys: `deny`, `deny_write`,
`ask_write`, `deny_commands` (regex), `ask_commands` (regex),
`allow_commands` (read-only command prefixes). Merging only adds rules:
`!exceptions` in files are ignored, and `allow_commands` is honored only
from your own file, because a project file arrives with whatever
repository you cloned. A malformed file is an error, not a skip.

## Environments: what each one does and does not protect

| environment | protects | does **not** protect |
|---|---|---|
| `local` | The file tools are jailed to the workspace root (symlinks and `..` included). Commands get a scrubbed environment (no `*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*CREDENTIAL*` variables), no stdin, a timeout, and a process-group kill. | **Anything a shell command does.** `bash` runs as you and can read, write and delete anything your user can, and reach the network. Only the gate stands in the way, and the gate is pattern-based. |
| `sandbox` (macOS seatbelt / Linux bubblewrap) | Commands can write only inside the workspace and temp dirs, never to `.orion/` or `.git/hooks`. They cannot read `~/.ssh`, `~/.aws`, `~/.config/gcloud`, `~/.netrc`, `~/.orion`, or `<root>/.env*`, and have no network unless `--exec-network`. `writable=False` makes the workspace read-only too. | CPU, memory, disk usage and time (beyond the per-command timeout). Reads of other files on the machine (only the listed secret locations are hidden). Other secrets inside the workspace that are not named `.env*` (for example `*.pem`) are hidden from the file tools but readable by commands. The file tools themselves run in the trusted harness process, bounded by the jail, not the sandbox. `sandbox-exec` is deprecated by Apple, though it still works. |
| `docker` | Commands see only the container plus the bind-mounted workspace. No capabilities, `no-new-privileges`, no network by default, 512 pids, 2 GB memory, run as your uid. | Anything in the mounted workspace (it is writable, including `.env` files in it). Kernel exploits, since containers share the host kernel. With `--exec-network`, everything the container can reach. |
| `ssh` | Your laptop: nothing runs locally except `ssh`. | The remote host. It is only as safe as that host, and counts as isolated only with `disposable=True`. A production box is not a sandbox. |

## Known leaks in the pattern rules

The command rules split commands the way a shell does, including
`$(...)`, backticks, `<(...)`, `sh -c '...'` and `eval`, and track `cd`
out of the workspace, so `cd / && rm -rf .` is caught. They still cannot
see:

- **Variables and indirection**: `rm -rf $DIR` with `DIR=/` set earlier,
  or `x=rm; $x -rf /`. Unknown expansions are not hard-denied. A recursive
  `rm` still lands on the ask list, but in `auto` on an isolated
  environment only the sandbox stops it.
- **Code written, then run**: the agent can write `cleanup.sh` or a Python
  script that deletes things and then run it. The rules see `python3
  cleanup.py`, not what it does.
- **Aliases, functions, PATH tricks**, and programs that delete on their own
  (`git clean` is on the ask list; `make clean` is not).
- **Exfiltration through allowed channels**: with network on, any command
  can send any readable file anywhere. The `curl`/`wget` ask rule only
  catches the obvious spelling.
- **Glob and variable arguments to read-only commands** (`cat *.pem`,
  `cat $F`) are not treated as read-only, so they need approval instead of
  auto-running. Some false positives here are intentional.
- **web_fetch DNS rebinding**: the host is resolved for the public-address
  check and resolved again for the connection, so a hostile DNS server can
  answer differently the second time. Every redirect hop is checked, but
  the connection is not pinned to the IP that was checked.
- **Prompt injection**: web content is wrapped in `<untrusted_content>`
  and the system prompt says it is data, not instructions. That reduces
  the risk but does not eliminate it. An injected instruction that leads
  to an allowed call still runs, which is why `auto` requires isolation.

When the rules miss, the environment decides the cost. Choose the
environment for the worst case, not the expected one.
