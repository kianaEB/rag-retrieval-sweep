---
inclusion: always
---

# Shell conventions

Every command written for me to run — task done-checks, verification
steps, setup instructions, anything in a `tasks.md` — must run in **Git
Bash on Windows** (MINGW64). That is the shell I use.

Use POSIX shell or Python one-liners:

- `grep` / `grep -c`, not `Select-String`
- `ls`, `find`, not `Get-ChildItem`
- `diff`, `cmp`, not `Compare-Object`
- `cat`, `head`, `sed -n`, not `Get-Content`
- `$?`, not `$LASTEXITCODE`
- `mktemp -d`, `rm -rf`, not `New-TemporaryFile` / `Remove-Item`

PowerShell cmdlets and PowerShell variables do not exist in Git Bash, so
a done-check written in PowerShell cannot be run and cannot verify
anything. When a check is easier to express in Python than in shell,
write it as `python -c "..."` or a short heredoc script — those run
identically in either shell and are preferred for anything involving
CSV, JSON or float comparison.

This rule exists because PowerShell commands appeared in two
consecutive task lists and had to be rewritten both times.
