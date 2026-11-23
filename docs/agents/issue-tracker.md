# Issue tracker

MewHelp tracks tickets in [GitHub Issues](https://github.com/EricKingWhy/intelligence-help/issues). For CLI commands, use `gh` in the checkout or pass `-R EricKingWhy/intelligence-help` explicitly.

## Operations

| Intent | Command |
| --- | --- |
| Fetch a ticket and its discussion | `gh issue view <number> --comments` |
| List open tickets | `gh issue list --state open`, with `--label` when filtering by triage state |
| Publish a ticket | `gh issue create --title "<title>" --body-file <file>` |
| Add a comment | `gh issue comment <number> --body-file <file>` |
| Update triage state | `gh issue edit <number> --add-label "<label>" --remove-label "<old-label>"` |
| Close as completed | `gh issue close <number> --reason completed` |
| Review implementation for a ticket | `gh pr view <number> --comments` and `gh pr diff <number>` |

Issue and PR numbers share a sequence. If a reference does not resolve as an issue, check the corresponding PR.

**PRs as a request surface: no.**

## Linked work

- For a work map, use a parent issue and GitHub sub-issues. Link an existing child with `gh issue edit <parent> --add-sub-issue <child>`. If sub-issues are unavailable, add the child to the parent's task list and put `Part of #<parent>` in the child.
- Record blockers using GitHub issue dependencies. To add one with the CLI, get the blocker's database ID with `gh api repos/EricKingWhy/intelligence-help/issues/<blocker> --jq .id`, then pass it as `-F issue_id=<database-id>` to `gh api --method POST repos/EricKingWhy/intelligence-help/issues/<child>/dependencies/blocked_by`. If dependencies are unavailable, use `Blocked by: #<blocker>` in the child.
- When using Wayfinder, label the parent `wayfinder:map` and children with one of `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`, or `wayfinder:task`. Work from the first open, unassigned child with all blockers closed. Record the result on the child and link it from the parent.
