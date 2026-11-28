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
A command line, read the way a shell reads it: which words are commands that run, and which are only data.

The first version of the hunts searched a whole command line with regular expressions, and so could not tell
`curl ... | sh` from `grep 'curl ... | sh' notes.md`, from a commit message that mentions it, or from a heredoc that
writes it into a test file. On a real coding session most of its command findings were text the agent wrote, not
commands it ran. This module gives the hunts the distinction:

- `parse` splits a line into simple commands, following pipes, lists, redirections, quotes, heredocs, line
  continuations and command and process substitution.
- Text is parsed as commands only where a shell would run it: the code of `sh -c`, `eval` and `ssh host`, a
  substitution, a heredoc or a pipe that feeds a shell, and a file the same line wrote and then ran. Everything else
  (quoted arguments, heredoc bodies written to files, search patterns) stays data and is never matched as a command.

It is not a shell. It expands nothing and evaluates nothing, it reads POSIX-style syntax well and PowerShell or
`cmd` syntax only as words and pipes, and anything it cannot read comes back as one command of whitespace-split
words, marked `uncertain`, so a caller never gets an exception from hostile text.
"""

import dataclasses
import functools
import re

MAX_PARSE = 200_000
"""The most characters of one command line that are read. A longer line is read to here and marked uncertain."""

MAX_DEPTH = 6
RESERVED_PREFIXES = frozenset(["do", "then", "else", "elif", "if", "while", "until", "!", "{"])

WINDOWS_LINE = re.compile(
    r"(?i)(?:^|[\s\"'=(])[a-z]:\\|\\\\[\w.$-]{1,64}\\|%[a-z_()]{2,32}%|\$env:|\bHK(?:CU|LM|CR|U|EY_[A-Z_]{4,24})\\|"
    r"^\s{0,8}(?:\S{0,200}[\\/])?(?:cmd|powershell|pwsh)(?:\.exe)?\s"
    r"|\b(?:get|set|remove|invoke|new|copy|move|start|add|out)"
    r"-[a-z]{3,20}\b")
"""A line written for Windows, where a backslash separates a path and escapes nothing."""
MAX_COMMANDS = 2000

SHELLS = frozenset(["sh", "bash", "zsh", "dash", "ksh", "ash", "fish"])
INTERPRETERS = frozenset([
    "python",
    "python2",
    "python3",
    "node",
    "nodejs",
    "deno",
    "bun",
    "perl",
    "ruby",
    "php",
    "pwsh",
    "powershell",
    "osascript",
    "lua"
])
WRAPPERS = {
    "sudo": frozenset(["-u", "-g", "-h", "-p", "-C", "-D", "-R", "-T", "-U", "--user", "--group"]),
    "doas": frozenset(["-u"]),
    "env": frozenset(["-u", "-C", "-S", "--unset", "--chdir"]),
    "nohup": frozenset(),
    "time": frozenset(["-f", "-o"]),
    "timeout": frozenset(["-s", "-k", "--signal", "--kill-after"]),
    "nice": frozenset(["-n", "--adjustment"]),
    "ionice": frozenset(["-c", "-n", "-p"]),
    "stdbuf": frozenset(["-i", "-o", "-e"]),
    "exec": frozenset(["-a"]),
    "command": frozenset(),
    "builtin": frozenset(),
    "xargs": frozenset(["-I", "-n", "-P", "-L", "-d", "-E", "-s", "--max-args", "--max-procs", "--delimiter"]),
    "watch": frozenset(["-n", "-d", "--interval"]),
    "caffeinate": frozenset(["-t", "-w"]),
    "setsid": frozenset(),
    "unbuffer": frozenset(),
}
"""Commands that run another command, with the options of each that take a value."""

INLINE_CODE_FLAGS = {
    "python": ("-c", ),
    "python2": ("-c", ),
    "python3": ("-c", ),
    "node": ("-e", "--eval", "-p"),
    "nodejs": ("-e", "--eval"),
    "deno": ("eval", ),
    "bun": ("-e", ),
    "perl": ("-e", "-E"),
    "ruby": ("-e", ),
    "php": ("-r", ),
    "pwsh": ("-c", "-command", "-Command"),
    "powershell": ("-c", "-command", "-Command"),
    "osascript": ("-e", ),
    "lua": ("-e", )
}


@dataclasses.dataclass
class Command:
    """One simple command: a program, its arguments, what it reads and writes, and how it came to run."""

    argv: list
    """The words, quotes removed, wrappers such as `sudo` and leading `VAR=value` assignments included."""
    raw: str
    """The text of this command as written."""
    pipeline: int = 0
    position: int = 0
    """Which pipeline of the line this is in, and where in it; position 0 reads no pipe."""
    piped_out: bool = False
    writes: list = dataclasses.field(default_factory=list)
    """Redirection targets written to, each `(path, append, descriptor-form)`."""
    reads: list = dataclasses.field(default_factory=list)
    """Redirection sources read from."""
    heredoc: str = ""
    herestring: str = ""
    via: str = ""
    """How this command comes to run when it is not written at the top level: `-c`, `eval`, `ssh`, `$(`, `<(`,
    `heredoc`, `pipe` or `script`. Empty at the top level."""
    parent: int = -1
    """Index of the command whose code or substitution this is."""
    whole_words: list = dataclasses.field(default_factory=list)
    """Indexes into `argv` of words that are one substitution and nothing else, each with the child command indexes:
    `[(argv index, kind, [children])]`."""
    assignments: dict = dataclasses.field(default_factory=dict)
    program: str = ""
    """The program that runs once wrappers and assignments are set aside: its base name, lower case, no `.exe`."""
    args: list = dataclasses.field(default_factory=list)
    """The words after `program`."""
    code: str = ""
    """Code this command hands to a language interpreter: the text after `python -c`, or a heredoc fed to one."""
    uncertain: bool = False

    @property
    def operands(self) -> list:
        """The arguments that are not options. A `--` ends the options."""
        found = []
        ended = False

        for word in self.args:
            if (ended or not word.startswith("-") or word == "-"):
                found.append(word)
            elif (word == "--"):
                ended = True

        return found

    def option(self, *names: str) -> bool:
        """Whether any of the named options is given, as a word, a `--name=value`, or inside a `-xyz` cluster."""
        for word in self.args:
            if (word == "--"):
                break

            for name in names:
                if (word == name or word.startswith(name + "=")):
                    return True

                if (len(name) == 2 and len(word) > 2 and word[0] == "-" and word[1] != "-" and name[1] in word[1:]
                        and word[1:].isalpha()):
                    return True

        return False

    def value(self, *names: str) -> list:
        """Every value given to the named options: the next word, or the text after `=`, or glued to a short name."""
        found = []

        for (index, word) in enumerate(self.args):
            for name in names:
                if (word == name and index + 1 < len(self.args)):
                    found.append(self.args[index + 1])
                elif (word.startswith(name + "=")):
                    found.append(word[len(name) + 1:])
                elif (len(name) == 2 and word.startswith(name) and len(word) > 2 and not word.startswith("--")):
                    found.append(word[2:])

        return found


class _Unreadable(Exception):
    """Syntax this reader does not follow; the caller falls back to plain words."""


def program_name(word: str) -> str:
    name = word.replace("\\", "/").rsplit("/", 1)[-1].lower()

    for suffix in (".exe", ".cmd", ".bat"):
        name = name.removesuffix(suffix)

    return name


def _matching(text: str, start: int, opener: str, closer: str) -> int:
    """Index just past the `closer` that balances the `opener` at `start - 1`, minding quotes."""
    depth = 1
    index = start
    quote = ""

    while index < len(text):
        character = text[index]

        if (quote):
            if (character == "\\" and quote == '"'):
                index += 1
            elif (character == quote):
                quote = ""
        elif (character in "'\""):
            quote = character
        elif (character == "\\"):
            index += 1
        elif (character == opener):
            depth += 1
        elif (character == closer):
            depth -= 1

            if (depth == 0):
                return index + 1

        index += 1

    raise _Unreadable("unbalanced")


class _Reader:
    """One pass over a line, collecting commands; nested code is read by a nested reader into the same list."""

    def __init__(self, commands: list, depth: int, via: str, parent: int, windows: bool = False):
        self.commands = commands
        self.depth = depth
        self.via = via
        self.parent = parent
        self.windows = windows

    def nested(self, text: str, via: str, parent: int) -> list:
        """Read `text` as code run by `parent`; returns the indexes of the commands it holds."""
        if (self.depth >= MAX_DEPTH or len(self.commands) >= MAX_COMMANDS or not text.strip()):
            return []

        before = len(self.commands)

        try:
            _Reader(self.commands, self.depth + 1, via, parent, self.windows).read(text)
        except _Unreadable:
            del self.commands[before:]
            self.commands.extend(_fallback_commands(text, via, parent))

        return list(range(before, len(self.commands)))

    def read(self, text: str) -> None:  # pylint: disable=too-many-branches,too-many-statements,too-many-locals
        length = len(text)
        index = 0
        pipeline = max((command.pipeline for command in self.commands), default=-1) + 1
        position = 0
        words: list = []
        word: list = []
        in_word = False
        whole: list = []  # (word index, kind, text) for words that are exactly one substitution
        pieces: list = []  # substitutions inside the current word: (kind, text)
        literal = False  # whether the current word has any text beside its substitutions
        writes: list = []
        reads: list = []
        pending_redirect = None  # (kind, append) awaiting its target word
        heredocs: list = []  # (delimiter, strip tabs) awaiting their bodies
        bodies: list = []
        herestring = ""
        start = 0
        substitutions: list = []  # every (kind, text) in this command, whole word or not

        def end_word():
            nonlocal word, in_word, pieces, literal, pending_redirect, herestring

            if (not in_word):
                return

            value = "".join(word)

            if (pending_redirect is not None):
                (kind, append) = pending_redirect
                pending_redirect = None

                if (kind == ">"):
                    writes.append((value, append))
                elif (kind == "<"):
                    reads.append(value)
                elif (kind == "<<<"):
                    herestring = value
                elif (kind == "<<"):
                    heredocs.append((value, append))
            else:
                if (len(pieces) == 1 and not literal):
                    whole.append((len(words), pieces[0][0], pieces[0][1]))

                words.append(value)

            (word, in_word, pieces, literal) = ([], False, [], False)

        def end_command(piped: bool, stop: int):
            nonlocal words, writes, reads, whole, bodies, herestring, start, position, pipeline, substitutions

            end_word()

            if (words or writes or reads):
                command = Command(argv=words,
                                  raw=text[start:stop].strip(),
                                  pipeline=pipeline,
                                  position=position,
                                  piped_out=piped,
                                  writes=writes,
                                  reads=reads,
                                  heredoc="\n".join(bodies),
                                  herestring=herestring,
                                  via=self.via,
                                  parent=self.parent)
                self.commands.append(command)
                own = len(self.commands) - 1

                for (kind, inner) in substitutions:
                    children = self.nested(inner, kind, own)

                    for (word_index, whole_kind, whole_inner) in whole:
                        if (whole_kind == kind and whole_inner == inner):
                            command.whole_words.append((word_index, kind, children))

                if (len(self.commands) > MAX_COMMANDS):
                    raise _Unreadable("too many commands")

            (words, writes, reads, whole, bodies, herestring, substitutions) = ([], [], [], [], [], "", [])
            start = stop

            if (piped):
                position += 1
            else:
                position = 0
                pipeline += 1

        while index < length:
            character = text[index]

            if (character == "\\" and index + 1 < length):
                if (text[index + 1] == "\n"):
                    index += 2
                    continue

                if (self.windows and text[index + 1] not in "\"'"):
                    word.append("\\")
                    (in_word, literal) = (True, True)
                    index += 1
                    continue

                word.append(text[index + 1])
                (in_word, literal) = (True, True)
                index += 2
                continue

            if (character == "'"):
                close = text.find("'", index + 1)

                if (close == -1):
                    raise _Unreadable("quote")

                word.append(text[index + 1:close])
                (in_word, literal) = (True, literal or close > index + 1)
                index = close + 1
                continue

            if (character == '"'):
                index += 1
                in_word = True

                while index < length and text[index] != '"':
                    if (text[index] == "\\" and index + 1 < length):
                        # Inside double quotes a backslash escapes only these; before anything else it is itself.
                        if (text[index + 1] not in '$`"\\\n'):
                            word.append("\\")

                        if (text[index + 1] != "\n"):
                            word.append(text[index + 1])
                            literal = True

                        index += 2
                    elif (text.startswith("$(", index) and not text.startswith("$((", index)):
                        close = _matching(text, index + 2, "(", ")")
                        pieces.append(("$(", text[index + 2:close - 1]))
                        substitutions.append(pieces[-1])
                        word.append(text[index:close])
                        index = close
                    elif (text[index] == "`"):
                        close = text.find("`", index + 1)

                        if (close == -1):
                            raise _Unreadable("backtick")

                        pieces.append(("$(", text[index + 1:close]))
                        substitutions.append(pieces[-1])
                        word.append(text[index:close + 1])
                        index = close + 1
                    else:
                        word.append(text[index])
                        literal = True
                        index += 1

                if (index >= length):
                    raise _Unreadable("quote")

                index += 1
                continue

            if (text.startswith("$(", index) and not text.startswith("$((", index)):
                close = _matching(text, index + 2, "(", ")")
                pieces.append(("$(", text[index + 2:close - 1]))
                substitutions.append(pieces[-1])
                word.append(text[index:close])
                in_word = True
                index = close
                continue

            if (character == "`"):
                close = text.find("`", index + 1)

                if (close == -1):
                    raise _Unreadable("backtick")

                pieces.append(("$(", text[index + 1:close]))
                substitutions.append(pieces[-1])
                word.append(text[index:close + 1])
                in_word = True
                index = close + 1
                continue

            if (character in "<>" and text.startswith("(", index + 1) and not in_word):
                close = _matching(text, index + 2, "(", ")")
                kind = "<(" if character == "<" else ">("
                pieces.append((kind, text[index + 2:close - 1]))
                substitutions.append(pieces[-1])
                word.append(text[index:close])
                in_word = True
                index = close
                continue

            if (character in " \t"):
                end_word()
                index += 1
                continue

            if (character == "#" and not in_word):
                newline = text.find("\n", index)
                index = length if newline == -1 else newline
                continue

            if (character == "\n"):
                end_word()

                if (heredocs):
                    cursor = index + 1

                    for (delimiter, strip) in heredocs:
                        lines = []

                        while cursor <= length:
                            newline = text.find("\n", cursor)
                            line = text[cursor:] if newline == -1 else text[cursor:newline]
                            cursor = length + 1 if newline == -1 else newline + 1

                            if ((line.lstrip("\t") if strip else line) == delimiter):
                                break

                            lines.append(line)

                        bodies.append("\n".join(lines))

                    heredocs.clear()
                    end_command(False, min(cursor, length))
                    index = min(cursor, length)
                    start = index
                    continue

                end_command(False, index)
                index += 1
                start = index
                continue

            if (character == ";"):
                end_command(False, index)
                index += 2 if text.startswith(";;", index) else 1
                start = index
                continue

            if (character == "|"):
                if (text.startswith("||", index)):
                    end_command(False, index)
                    index += 2
                else:
                    end_command(True, index)
                    index += 2 if text.startswith("|&", index) else 1

                start = index
                continue

            if (character == "&"):
                if (text.startswith("&&", index)):
                    end_command(False, index)
                    index += 2
                    start = index
                    continue

                if (text.startswith("&>", index)):
                    end_word()
                    append = text.startswith("&>>", index)
                    pending_redirect = (">", append)
                    index += 3 if append else 2
                    continue

                end_command(False, index)
                index += 1
                start = index
                continue

            if (character in "<>"):
                # A descriptor number glued to the operator belongs to it, not to the word before.
                if (in_word and "".join(word).isdigit() and not pieces):
                    (word, in_word, literal) = ([], False, False)

                end_word()

                if (character == ">"):
                    if (text.startswith(">&", index)):
                        # `>&2`, `2>&1`: a descriptor, not a file; `>& file` is a file.
                        cursor = index + 2
                        target = ""

                        while cursor < length and (text[cursor].isdigit() or text[cursor] == "-"):
                            target += text[cursor]
                            cursor += 1

                        if (target):
                            index = cursor
                        else:
                            pending_redirect = (">", False)
                            index += 2

                        continue

                    append = text.startswith(">>", index)
                    pending_redirect = (">", append)
                    index += 2 if append or text.startswith(">|", index) else 1
                    continue

                if (text.startswith("<<<", index)):
                    pending_redirect = ("<<<", False)
                    index += 3
                    continue

                if (text.startswith("<<", index)):
                    strip = text.startswith("<<-", index)
                    pending_redirect = ("<<", strip)
                    index += 3 if strip else 2
                    continue

                if (text.startswith("<>", index)):
                    # Open for reading and writing: how a shell binds a socket to a descriptor.
                    pending_redirect = (">", False)
                    reads.append("<>")
                    index += 2
                    continue

                if (text.startswith("<&", index)):
                    cursor = index + 2

                    while cursor < length and (text[cursor].isdigit() or text[cursor] == "-"):
                        cursor += 1

                    index = cursor
                    continue

                pending_redirect = ("<", False)
                index += 1
                continue

            if (character in "(){}" and not in_word):
                # Grouping and subshells: the commands inside run, so read on through them.
                if (character in "({"):
                    start = index + 1
                else:
                    end_command(False, index)
                    start = index + 1

                index += 1
                continue

            word.append(character)
            (in_word, literal) = (True, True)
            index += 1

        end_command(False, length)


_SEPARATORS = re.compile(r"\|\||&&|[;|\n]")


def _fallback_commands(text: str, via: str = "", parent: int = -1) -> list:
    """
    What can be said of a line this reader cannot follow, an unbalanced quote being the usual reason: its pieces
    between separators, each as plain words, each marked uncertain. Quotes are not honoured, so a quoted separator
    splits a command; a caller that needs certainty checks `uncertain`.
    """
    found = []

    for (number, piece) in enumerate(_SEPARATORS.split(text[:MAX_PARSE])[:200]):
        words = piece.split()[:400]

        if (words):
            command = Command(argv=words,
                              raw=piece.strip()[:4000],
                              pipeline=number,
                              via=via,
                              parent=parent,
                              uncertain=True)
            _resolve(command)
            found.append(command)

    return found


def _resolve(command: Command) -> None:
    """Set aside assignments and wrappers to find the program that runs, and the code it is handed."""
    words = list(command.argv)

    while words and "=" in words[0] and words[0].split("=", 1)[0].replace("_", "a").isalnum() \
            and not words[0][0].isdigit():
        (name, value) = words.pop(0).split("=", 1)
        command.assignments[name] = value

    # `do curl ...`, `then rm ...`, `! grep ...`: the command is what follows the reserved word.
    while len(words) > 1 and words[0] in RESERVED_PREFIXES:
        words.pop(0)

    for _ in range(6):
        if (not words):
            break

        name = program_name(words[0])

        if (name not in WRAPPERS):
            break

        takes_value = WRAPPERS[name]
        words.pop(0)

        while words and (words[0].startswith("-") or (name == "env" and "=" in words[0]) or
                         (name == "timeout" and words[0][:1].isdigit())):
            option = words.pop(0)

            if (name == "timeout" and option[:1].isdigit()):
                break

            if (option in takes_value and words):
                words.pop(0)

            if (option == "--"):
                break

    if (not words):
        if (command.argv and not command.assignments):
            command.program = program_name(command.argv[0])

        return

    command.program = "" if words[0][:2] in ("$(", "<(", ">(") or words[0][:1] == "`" else program_name(words[0])
    command.args = words[1:]

    # `python3.12` is `python3`.
    for base in ("python3", "python2", "python"):
        if (command.program.startswith(base) and command.program[len(base):].replace(".", "").isdigit()):
            command.program = base
            break


def _inline(reader: _Reader, commands: list) -> None:  # pylint: disable=too-many-branches
    """Read as commands the text each command would run: `-c` code, `eval`, `ssh`, and stdin fed to a shell."""
    written: dict = {}
    index = 0

    while index < len(commands) and len(commands) < MAX_COMMANDS:
        command = commands[index]
        index += 1

        if (not command.program and command.argv):
            _resolve(command)

        program = command.program
        flags = INLINE_CODE_FLAGS.get(program)

        if (program in SHELLS):
            codes = [
                command.args[i + 1] for (i, word) in enumerate(command.args[:-1])
                if word.startswith("-") and not word.startswith("--") and word.endswith("c")
            ]

            for code in codes[:1]:
                position = command.argv.index(code) if code in command.argv else -1
                substituted = [(kind, children) for (word_index, kind, children) in command.whole_words
                               if word_index == position]

                # `sh -c "$(curl ...)"`: the code is one substitution, so what it prints is what runs.
                for (kind, children) in substituted:
                    for child in children:
                        commands[child].via = "-c" + kind

                if (not substituted):
                    reader.nested(code, "-c", index - 1)

            if (not codes):
                script = next((word for word in command.operands if word not in ("-", "-s")), "")

                if (script in written):
                    reader.nested(written[script], "script", index - 1)
                elif (not script):
                    if (command.heredoc or command.herestring):
                        reader.nested(command.heredoc or command.herestring, "heredoc", index - 1)
                    elif (command.position > 0):
                        producer = next((c for c in reversed(commands[:index - 1])
                                         if c.pipeline == command.pipeline and c.position == command.position -
                                         1 and c.via == command.via),
                                        None)

                        if (producer is not None and producer.program in ("echo", "printf") and producer.args):
                            reader.nested(" ".join(producer.operands), "pipe", index - 1)
                        elif (producer is not None and producer.program == "cat" and producer.heredoc):
                            reader.nested(producer.heredoc, "pipe", index - 1)
        elif (program == "cmd" and command.args[:1] and command.args[0].lower() in ("/c", "/k", "/r")):
            reader.nested(" ".join(command.args[1:]), "-c", index - 1)
        elif (program == "eval"):
            for (_, kind, grandchildren) in command.whole_words:
                for child in grandchildren:
                    commands[child].via = "eval" + kind

            if (not command.whole_words):
                reader.nested(" ".join(command.args), "eval", index - 1)
        elif (program in ("source", ".")):
            script = next(iter(command.operands), "")

            if (script in written):
                reader.nested(written[script], "script", index - 1)
        elif (program == "ssh"):
            operands = _ssh_operands(command)

            if (len(operands) > 1):
                reader.nested(" ".join(operands[1:]), "ssh", index - 1)
        elif (flags is not None):
            for (position, word) in enumerate(command.args[:-1]):
                if (word in flags or word.lower() in flags):
                    command.code = command.args[position + 1]
                    break
            else:
                if ((command.heredoc or command.herestring) and not command.operands):
                    command.code = command.heredoc or command.herestring

        # A file this line writes may be run later in the same line: remember what went into it.
        content = command.heredoc or (" ".join(command.operands) if program in ("echo", "printf") else "")

        if (content and program in ("cat", "tee", "echo", "printf")):
            for (target, _) in command.writes:
                written[target] = content

            if (program == "tee"):
                for target in command.operands:
                    written[target] = content

        # ... and running it is running what was written.
        invoked = command.argv[len(command.argv) - len(command.args) - 1] if program and command.argv else ""

        if (program not in SHELLS and invoked):
            for path in (invoked, invoked.removeprefix("./")):
                if (path in written):
                    reader.nested(written[path], "script", index - 1)
                    break


SSH_VALUE_OPTIONS = frozenset([
    "-i",
    "-p",
    "-l",
    "-o",
    "-F",
    "-J",
    "-L",
    "-R",
    "-D",
    "-b",
    "-c",
    "-m",
    "-S",
    "-W",
    "-w",
    "-E",
    "-e",
    "-B",
    "-I",
    "-Q",
    "-P"
])


def _ssh_operands(command: Command) -> list:
    """`[destination, remote command words...]` of an `ssh` command, skipping options and their values."""
    found = []
    skip = False

    for word in command.args:
        if (skip):
            skip = False
        elif (found):
            found.append(word)
        elif (word in SSH_VALUE_OPTIONS):
            skip = True
        elif (not word.startswith("-")):
            found.append(word)

    return found


@functools.lru_cache(maxsize=4096)
def parse(text: str) -> tuple:
    """
    The commands a line runs, in order, nested code included.

    Returns
    -------
    tuple of Command
        Never raises. A line this reader cannot follow comes back as one `uncertain` command.
    """
    if (not isinstance(text, str) or not text.strip()):
        return ()

    cut = len(text) > MAX_PARSE
    text = text[:MAX_PARSE]
    commands: list = []
    reader = _Reader(commands, 0, "", -1, WINDOWS_LINE.search(text) is not None)

    try:
        reader.read(text)

        for command in commands:
            _resolve(command)

        _inline(reader, commands)

        for command in commands:
            if (not command.program and command.argv):
                _resolve(command)
    except (_Unreadable, RecursionError, IndexError, ValueError):
        commands = _fallback_commands(text)

    if (cut):
        for command in commands:
            command.uncertain = True

    return tuple(commands)


def pipeline_of(commands: tuple, command: Command) -> list:
    """The commands of one pipeline, in order, at the same nesting as `command`."""
    return [
        other for other in commands
        if other.pipeline == command.pipeline and other.via == command.via and other.parent == command.parent
    ]
