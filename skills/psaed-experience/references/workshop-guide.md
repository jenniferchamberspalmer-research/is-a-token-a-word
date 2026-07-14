# The Five Sessions — ID + SME Collaboration Protocol

Each session is 45–60 minutes, ID and SME both present, Claude in the third chair.
Every session ends with a written artifact; no artifact, no next session.
The sessions map onto the production pipeline: pre-scope → scope → build → go-to-market → improve.

---

## Session 1 — Pre-scope: should this exist?

**Goal:** an honest go/kill decision.
**Artifact:** a one-paragraph charter, or a documented kill.

Claude asks the room:

1. Who, specifically, is the learner? (A name-able person, not a demographic. "Second-semester nursing students who failed the dosage exam once" beats "students.")
2. What will they be able to *do* after this that they can't do now?
3. What already exists that covers this? Why is it insufficient?
4. What's the cost of building this — and the cost of *not* building it?
5. Kill test: if the answer to 2 is "they'll be aware of…", kill it or re-scope it. Awareness is not an outcome.

The ID writes the charter. The SME signs it. Claude keeps it in view for every later session.

---

## Session 2 — Scope: fix the evidence of learning

**Goal:** a measurable definition of done *before format is discussed*.
**Artifact:** the scope sheet.

Claude asks:

1. **The learner sentence.** "After this experience, ⟨learner⟩ will be able to ⟨observable behavior⟩ in ⟨context⟩." No verbs like *understand* or *appreciate* — if you can't watch it happen, it's not evidence.
2. **The rubric dimensions.** Pick 3–5 dimensions of understanding this experience should grow (these become the visible strip at the bottom of the page). Give each a short name and a one-line hint. Example from the reference page: *values, technical depth, leadership, complexity*.
3. **The registers.** The page opens by asking the learner how much time/depth they want. Define the three registers (e.g., *quick / guided / deep*) and what route each one gets.
4. **The failure condition.** What result, observed after launch, would mean this experience should be redesigned or retired?

The SME's role in this session: veto any outcome that isn't true to the discipline. The ID's role: veto any outcome that isn't observable.

---

## Session 3 — Story harvest: Claude interviews the SME

**Goal:** the raw material — doors, scenes, checkpoints, evidence.
**Artifact:** the harvest doc (verbatim quotes + the ID's structural tags).

This is the longest session. The ID mostly listens. Claude interviews the SME:

**Finding the doors** (the learner-chosen entry questions — aim for exactly 3):

1. "What are the three questions a curious learner would actually ask you about this, in their own words?"
2. "Which question do learners *think* they want answered first, and which one do they actually need first?"
3. "What question are you never asked but wish you were?"

**Finding the scenes** (the sub-30-second opening dialogues, one per register):

4. "Explain the heart of this topic to me in under ten sentences of back-and-forth conversation — I'll play the skeptic." (Claude plays the skeptic; the transcript, tightened, becomes a scene script.)
5. "Now the same conversation, but I'm in a hurry." (→ quick register)
6. "Now the same conversation, but I want the full story over dinner." (→ deep register)

**Finding the concrete anchor** (the P in PSAED):

7. "What is the single most concrete, specific, physical example in this whole topic — one object, one moment, one word?" (The reference page opens its personal door with one word: *y'all*. Every door needs an anchor this specific.)

**Finding the checkpoints** (the A and D):

8. "If a learner got the wrong idea from this door, what would the wrong idea be?" (Wrong ideas become checkpoint options — remember: no wrong *answers*, only answers that grow different rubric dimensions.)
9. "What connects door 1's story to door 2's story? A learner who sees that connection has learned something real — that's a checkpoint question."

**Finding the caveats** (the calibration pass starts here):

10. "What does this field overclaim? What do *you* refuse to claim? Where are the edges of your own certainty?" (This material goes in the page verbatim-ish. Stated limits are the most credibility-building content an expert can ship.)

The ID tags the transcript: `DOOR`, `SCENE`, `ANCHOR`, `CHECKPOINT`, `CAVEAT`, `EVIDENCE` (links/artifacts to embed), `CUT`.

---

## Session 4 — Build backward

**Goal:** a filled template.
**Artifact:** the draft page, running in a browser.

Order of construction — assessments first, exposition last:

1. **Rubric dimensions** (from session 2) → the `DIMS` config.
2. **Check-in questions** (from harvest tags) → one per door, three options each, every option mapped to a dimension with genuinely affirming feedback.
3. **Doors and routes** → which stations each register visits, and the guide's one-recommendation-at-a-time sequence.
4. **Scenes** → tighten harvest transcripts to ≤8 lines, ≤30 seconds. Read them aloud once; cut anything that doesn't survive being spoken.
5. **Stations** → one idea per screen, one interaction per station, concrete anchor first, guide line on top (`guideBar`), design note on the bottom (`dnote`).
6. **Closing** → the synthesis station: what the doors add up to, and the learner's rubric handed back to them as the receipt.

Division of labor: Claude drafts into the template; the SME red-pens truth; the ID red-pens learnability. Nobody edits the other's red ink.

---

## Session 5 — Review, gate, ship, loop

**Goal:** shipped page + scheduled review.
**Artifact:** the live URL and the improvement log.

1. **PSAED gate.** Walk every station against the five tests (checklist in `design-spec.md`). A station may fail forward only with a dated note in the improvement log.
2. **Calibration pass.** SME tags every factual claim: *established / demonstrated-here / reported-by-others / hypothesis*. Untaggable claims get cut or explicitly hedged in the page text.
3. **Disclosure check.** Footer says what the page is (static, scripted, nothing collected). The scene's register labels don't promise live systems that aren't there.
4. **Browser verification.** Every door clicked, every checkpoint answered, rubric grows, mobile has no horizontal scroll, console has no errors.
5. **Ship** to any static host.
6. **Schedule the loop.** Put a review date on the calendar now. At review: did the failure condition from session 2 trigger? Redesign, retire, or template what worked.

---

## Anti-patterns (the ID enforces these)

- **The lecture in a trench coat** — stations that are just paragraphs with a Next button. One interaction per station, minimum.
- **The quiz that judges** — checkpoints with wrong answers. Every option must teach; the rubric measures exposure, not the learner.
- **The fourth-wall spree** — self-referential "notice what this page is doing" moments. You get one, where the framework *is* the content.
- **The menu** — more than one decision per screen, or a hub the learner keeps falling back to. The guide always knows the next step.
- **The overclaim** — content the SME wouldn't defend to a peer. If the SME hesitates, the calibration pass already failed.
