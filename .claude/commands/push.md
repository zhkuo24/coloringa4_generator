---
allowed-tools: Bash(git status:*), Bash(git log:*), Bash(git branch:*), Bash(git push:*), Bash(git remote:*)
description: Push commits to remote repository safely
---

# Git Push Command

## Current State

- **Branch:** !`git branch --show-current`
- **Remote:** !`git remote -v | head -2`
- **Status:** !`git status --short`
- **Unpushed commits:** !`git log @{u}..HEAD --oneline 2>/dev/null || echo "No upstream or new branch"`

## Task

Push commits to remote repository safely.

### Step 1: Pre-push Checks
1. Run `git status` to ensure working directory is clean
2. Check if there are commits to push
3. Verify current branch name

### Step 2: Push Strategy

**For existing branches with upstream:**
```bash
git push
```

**For new branches (no upstream):**
```bash
git push -u origin <branch-name>
```

**If behind remote (need to pull first):**
```bash
git pull --rebase origin <branch-name>
git push
```

### Step 3: Verify
Confirm push was successful by checking output.

## Safety Rules
- NEVER use `--force` or `-f` unless user explicitly requests
- NEVER force push to `main` or `master` branch
- If push is rejected, inform user and suggest `git pull --rebase`
- Check for unpushed commits before pushing

## User Request
$ARGUMENTS
