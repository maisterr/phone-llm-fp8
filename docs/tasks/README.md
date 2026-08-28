# Tasks — work board

Every unit of work in flight, one file each. A unit is usually a **plan**
(staged, architectural — the default size here) but a small standalone task is
fine too when it doesn't earn a plan. Size lives in the content, not the folder.

Its **folder is its status** — move the file, not a field.

```
backlog/   queued, not fully planned
next/      ready for implementation
done/      shipped, kept as record
```

## Workflow

1. New work → copy `_template.md` into `backlog/` as `0-slug.md`. Backlog stays
   **unnumbered** (`0-` prefix) — order isn't committed yet. A plan-sized item
   carries its full staging in that file; a small task stays short.
2. Ready it → `git mv` `backlog/ → next/`, rename `0-slug` → next free `NNN-slug`.
   Numbers run one sequence across `next/` + `done/`, zero-padded to 3 digits.
3. Shipped it → `git mv` `next/ → done/`, keep the number.
