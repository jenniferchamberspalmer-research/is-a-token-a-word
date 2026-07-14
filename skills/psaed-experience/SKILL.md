---
name: psaed-experience
description: Co-design and build a guided, PSAED-compliant interactive learning experience (a single-file HTML page) through a structured collaboration between an instructional designer and a subject-matter expert. Use when a team wants to turn SME expertise into a branching, guided web experience with formative assessment built in — a course intro, a concept walkthrough, an onboarding journey, or a "meet the expert" page.
---

# PSAED Experience Builder

A skill for **two humans and Claude working together**: an **instructional designer (ID)** who owns *what is learnable*, and a **subject-matter expert (SME)** who owns *what is true*. The output is a single self-contained HTML page — no build step, no server, no API calls — that walks a learner through the SME's material as a guided, branching experience with visible formative assessment.

**Working reference:** this skill was extracted from a live demonstration page (`hello/index.html` in this repository). Open it before building anything; every pattern below exists there in working form, and its "Show the pedagogy" toggle explains each design decision in place.

## The quality bar: PSAED

Every learning object shipped with this skill must pass five tests:

| Test | The object must… | In the template, this is… |
|---|---|---|
| **P — Personal** | meet learners where they are; let them choose their entry point | the "doors" — learner-chosen questions, one decision per screen |
| **S — Social** | be dialogic — an exchange, not a broadcast | the opening register choice + the scripted two-avatar scene |
| **A — Active** | make the learner do something and see feedback | check-in beats, think-pair-share, clickable reveals |
| **E — Experiential** | put the learner inside the real thing, not a summary of it | scenes, live artifacts, era/persona conversations |
| **D — Developmentally appropriate** | sequence to the learner's edge; assess formatively and visibly | concrete-before-abstract ordering + the always-visible rubric strip |

If a station fails a test, fix the station — don't relax the test.

## Roles and posture

- **The SME owns truth.** Nothing ships that the SME hasn't verified. The ID never overrides SME judgment on content accuracy.
- **The ID owns learnability.** Chunking, sequencing, interaction design, and assessment are the ID's call. The SME never ships a wall of text because "learners need all of it."
- **Service leadership, both directions.** The ID listens first, then translates; the SME trusts the translation once they've verified the truth survived it. The negotiation between the two ownerships is where quality lives.
- **Claude is the third chair**: interviewer, drafter, and consistency-checker — never the arbiter of truth (SME's job) or of design (ID's job).

## The workflow

Run the five sessions in `references/workshop-guide.md`. In brief:

1. **Pre-scope** — decide whether this experience should exist at all. Kill freely.
2. **Scope** — fix the learner, the outcome, and the *evidence of learning* before any content. Write the definition of done.
3. **Story harvest** — Claude interviews the SME; the ID listens for doors, scenes, and checkpoints. This session produces the raw material for everything.
4. **Build backward** — assessments first, then experiences, then exposition. Fill `references/template.html` section by section (it is annotated with `«REPLACE»` markers and duplication instructions).
5. **Review & ship** — run the PSAED gate and the calibration pass (below), then publish (any static host: GitHub Pages, an LMS file upload, a shared drive).

## Claude's operating instructions

When this skill is invoked:

1. Ask who is present (ID, SME, or both) and which session (1–5) they're in. If they're new, start at pre-scope — do not skip to building.
2. Read `references/design-spec.md` before proposing any page structure, and `references/workshop-guide.md` before facilitating any session.
3. In story harvest, interview the SME using the question scripts in the workshop guide. Capture answers verbatim first; translate second. Show the SME every translation for accuracy sign-off.
4. When building, copy `references/template.html` to a new file and replace content only — do not restructure the engine unless the ID asks. Every `«REPLACE»` marker must be resolved or deleted before ship.
5. Keep the **design-notes layer** honest: every station's `dnote()` must state the real pedagogical reason the station is built the way it is. If there is no reason, the station needs redesign, not a better excuse.
6. Enforce the **calibration pass**: before ship, walk the SME through every factual claim and tag it *established / demonstrated-here / reported-by-others / hypothesis*. Anything the SME won't tag gets cut or hedged in the text itself.
7. Enforce the **disclosure norm**: the page must say what it is (static, scripted, no data collected) in its own footer. Never fake a live system.
8. Verify before declaring done: open the built page in a browser (or ask the team to), click every door, answer every checkpoint, and confirm the rubric strip grows and no console errors appear.

## Constraints (non-negotiable)

- Single self-contained HTML file. No CDN dependencies except fonts (which must degrade gracefully). No API calls. No analytics that leave the page.
- No proprietary or licensed content without written clearance — describe, don't reproduce.
- The formative rubric measures *exposure and engagement*, never judgment of the learner. Its copy must say so.
- Accessibility floor: real buttons for interactions, aria-labels on decorative SVG, readable contrast, no information carried by color alone, no horizontal scroll on mobile.

## Files

- `references/workshop-guide.md` — the five-session ID+SME collaboration protocol, with Claude's interview scripts.
- `references/design-spec.md` — the anatomy of the pattern: every component, its pedagogical rationale, and when to cut it.
- `references/template.html` — the working engine with placeholder content. Open it in a browser as-is; it runs.

## Installing as a Claude skill

This folder already follows the Agent Skills convention. To make it available in a project, copy it to `.claude/skills/psaed-experience/` in that repository (or `~/.claude/skills/` for personal use), then invoke it by asking Claude to build a PSAED experience — or type `/psaed-experience`. It also works as a plain document: hand `SKILL.md` and the workshop guide to any ID/SME pair and the process runs on paper.
