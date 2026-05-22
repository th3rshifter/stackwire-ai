# Git

## Core model
Git stores snapshots as commits. Branches are movable references to commits, not separate copies of the repository. `HEAD` points to the current branch or directly to a commit in detached HEAD state.

## Everyday workflow
Use `git status` to see working tree and index state, `git diff` for unstaged changes, `git diff --cached` for staged changes, and `git log --oneline --graph --decorate` to inspect history.

## Troubleshooting
When history looks wrong, first identify where you are:

```bash
git status
git branch --show-current
git log --oneline --decorate -5
git reflog -5
```

`git reflog` is useful for recovering recent branch positions after reset, rebase or checkout mistakes.
