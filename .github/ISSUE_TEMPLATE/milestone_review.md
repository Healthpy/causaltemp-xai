---
name: Milestone Review
about: Gate review before a milestone is declared complete and its branch merges to main
title: "[MILESTONE] M? Review — "
labels: milestone-review
assignees: ""
---

## Milestone
<!-- e.g., M1 — Metric Integrity -->

## Review Date

## Reviewers
- PI: 
- Methodology & Coding: 
- Results & Writing (adversarial challenge): 

---

## DoD Checklist (from ROADMAP.md)

Copy the milestone's Definition of Done here and check each item:
- [ ] 
- [ ] 

---

## Standing Rules Gate (CLAUDE.md)

- [ ] **R1** — pytest -q passes: 0 failures, 0 errors (attach CI run link)
- [ ] **R2** — All experiment scripts run on committed code (git log attached)
- [ ] **R3** — No proxy is exported under its original method's name
- [ ] **R5** — All docs referenced in the milestone's commits match the current code
- [ ] **R6** — Every new metric has a committed adversarial test (fail on violation, pass on oracle)
- [ ] **R7** — All result files in this milestone contain `git_commit` and `seed`
- [ ] **R8** — CI is green on the feature branch (attach CI run link)
- [ ] **R9** — No new unwired public exports
- [ ] **R10** — Pipeline is reproducible from clean checkout (smoke test run attached)

---

## Results & Writing Adversarial Challenge

For each metric introduced or changed in this milestone, Results & Writing must record:

| Metric | Constructed violation CF | Does metric flag it? | Oracle CF | Does metric pass it? |
|--------|--------------------------|----------------------|-----------|----------------------|
| | | [ ] Yes / [ ] No | | [ ] Yes / [ ] No |

If any row shows "No" in the "flags violation" column, the milestone is BLOCKED.

---

## Open Items at Review Time

<!-- List anything that was not completed in this milestone and where it goes next -->

| Item | Disposition |
|------|-------------|
| | Deferred to M? |

---

## Decision

- [ ] **APPROVED** — all DoD items checked; CI green; adversarial challenge passed
- [ ] **BLOCKED** — open items listed above must be resolved before merge
- [ ] **PARTIAL MERGE** — subset approved (document exactly what is excluded)

**PI sign-off:**  
**Date:**
