---
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git log:*), Bash(git add:*), Bash(git commit:*)
description: Create a well-formatted git commit following best practices
---

# Git Commit Command

## Current State

- **Branch:** !`git branch --show-current`
- **Status:** !`git status --short`
- **Recent commits (for style reference):** !`git log --oneline -5`

## Task

Create a git commit following these best practices:

### Step 1: Analyze Changes
1. Run `git status` to see all untracked and modified files
2. Run `git diff` to see unstaged changes
3. Run `git diff --staged` to see staged changes

### Step 2: Stage Files
- Only stage files relevant to a single logical change
- Do NOT stage files containing secrets (.env, credentials, etc.)
- Use `git add <file>` for specific files, or `git add .` for all

### Step 3: Write Commit Message
Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Build, config, dependencies

**Rules:**
- Subject: imperative mood, lowercase, no period, max 50 chars
- Body: explain "why" not "what", wrap at 72 chars
- Reference issues if applicable: `Fixes #123`

### Step 4: Create Commit
Use HEREDOC format:
```bash
git commit -m "$(cat <<'EOF'
<type>(<scope>): <subject>

<body>
EOF
)"
```

### Step 5: Verify
Run `git status` after commit to confirm success.

## Safety Rules
- NEVER use `--force` or `--no-verify`
- NEVER commit secrets or credentials
- NEVER amend pushed commits without explicit user request
- Do NOT push unless user explicitly asks

## User Request
$ARGUMENTS
