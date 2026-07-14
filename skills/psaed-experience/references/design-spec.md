# Design Spec — Anatomy of a PSAED Guided Experience

Every component below exists, working, in the reference page (`hello/index.html` at the repository root of this skill's home repo) and in skeletal form in `template.html`. For each: what it is, why it works, and when to cut it.

---

## 1. The register choice (front door) — *Social*

**What:** The page opens with one question — "How much time do you have?" — and three cards (quick / guided / deep). No content is visible before this choice.

**Why:** Learning starts with a relationship, so the page starts by letting the learner set the relational register. It is also a commitment device (people finish what they chose) and it parameterizes the whole visit: route, pacing, and when the payoff station arrives.

**Cut when:** the experience is under ~5 minutes total. A single register needs no choice.

## 2. The scene (the "commercial") — *Social + Experiential*

**What:** A sub-30-second auto-playing dialogue between two illustrated avatars, with a scene-setting animation, timed speech bubbles, a progress bar, skip and replay. One script per register — same facts, three bandwidths.

**Why:** The learner *watches a relationship* instead of reading a claim. Dialogue survives skimming; monologue doesn't. The three scripts also demonstrate audience translation — the core skill of writing one truth for multiple audiences.

**Rules:** ≤8 lines. Read it aloud; cut what dies when spoken. Show a "read the unabridged version" accordion after. Disclose that it's scripted.

**Cut when:** you cannot get a script the SME will vouch for. A bad scene is worse than none.

## 3. Doors as learner questions — *Personal*

**What:** Exactly three entry paths, each phrased as a question in the learner's own words ("Who were you before Claude?" — not "Background"). The learner picks which question gets asked next.

**Why:** Choosing a question is an act of curiosity; choosing a menu item is navigation. Three is enough for real agency without decision fatigue.

**Rules:** Door titles come from the SME harvest (what learners actually ask), not from the content outline.

## 4. The guide — *the connective tissue*

**What:** A persistent persona (in the reference page: the interviewer from the scene) who (a) opens every station with one italic line of context, and (b) after every completed beat recommends exactly **one** next step, with a quiet "let me choose instead" escape.

**Why:** Guidance is what turns a website into a curriculum. One-recommendation-at-a-time removes decision cost everywhere agency isn't the point, so the learner's agency is spent where it matters (the doors, the checkpoints).

**Rules:** The guide has a voice and a stake ("this is the part that made me sit up"), never a narrator's neutrality. The guide's route respects the register chosen at the front door.

## 5. Stations — *one idea, one interaction*

**What:** A screen holds one idea, one interaction (a reveal choice, a tab set, a clickable diagram), a guide line on top, and a design note at the bottom.

**Why:** Chunking is not shortening; it's giving each idea a completable shape. The interaction forces a micro-decision, which is what encoding needs.

**Rules:** Concrete anchor before abstraction, always (the reference page teaches a 25-institution portfolio through one welding booth). Sequencing decides memory: last position in a path goes to the most durable takeaway, not the most impressive one.

## 6. Check-in beats — *Active + Developmental*

**What:** After each door, a dedicated screen: "What have you learned so far?" — one question, three options, **no wrong answers**. Each option grows a different rubric dimension and returns genuinely affirming, information-adding feedback. Then the guide's single next-step invitation appears.

**Why:** Retrieval at peak freshness (the testing effect), kept at 20 seconds and zero stakes so it teaches instead of judging. Placing it *between* doors — not buried inside them — makes the rhythm legible: content, synthesis, invitation.

## 7. The visible rubric strip — *Developmental*

**What:** A slim always-on strip pinned to the bottom of every screen ("What have you learned so far?") with one emoji meter per dimension (⚪ → 🌱 → 🌿 → 🌳), growing visibly the moment the learner does anything. Expands into a full panel with the formative-data log.

**Why:** Assessment as something the learner *watches happen*, never something that happens to them. The strip is also the experience's own instrumentation — the same data the design team reviews at the improvement loop.

**Rules:** The copy must say it measures exposure and engagement, not judgment. Every point-award is logged in learner-readable language.

## 8. Think-pair-share — *Active*

**What:** The learner picks an idea worth interrogating, picks a simulated partner (each partner is a genuinely different critical lens — skeptic, decision-maker, home-audience), reads the exchange, and sees it land in their formative data.

**Why:** Articulation under mild social pressure forces reorganization. Choosing a partner = choosing which stress-test the idea gets, which is itself a metacognitive act.

**Cut when:** the content has no live controversy to interrogate. Don't fake friction.

## 9. Persona / era conversations — *Experiential*

**What:** The learner "talks to" a version of the expert (or a practitioner) situated inside a specific past project — with that moment's knowledge and none of what came later. Canned questions, written answers, temporal boundary respected.

**Why:** Reading about finished work is the least experiential genre there is. Entering the work while it was still uncertain is the honest way to show how thinking develops.

**Rules:** Disclose that answers are reconstructed. Breaking the era boundary is allowed once, labeled, for a payoff.

## 10. The pedagogy toggle — *the medium is the message*

**What:** A top-bar toggle ("Show the pedagogy") that reveals a design note on every station: which PSAED test it serves and why it's built this way.

**Why:** For faculty audiences this layer *is* professional development — the page teaches its own design. It also keeps the builders honest: a station you can't annotate is a station you can't justify.

**Rules:** Notes state real reasons, not marketing. Exactly one station may quote this layer in its main text (the self-referential reveal); more than one spends the learner's attention on the frame.

## 11. The closing synthesis

**What:** The final station integrates the doors into one argument, hands the learner their rubric as the receipt ("look at the strip — that's what you built"), and links every artifact referenced.

**Why:** Currere's synthetical moment: regressive (where we started), progressive (where this goes), analytical (what it means), integrated. A learner who walked multiple doors *experiences* the synthesis rather than being told it.

---

## Engineering constraints

- **One file.** Self-contained HTML/CSS/JS. Fonts may load from a CDN but must degrade gracefully. No frameworks, no build step.
- **Static and honest.** No API calls; scripted scenes disclosed as scripted; footer states nothing is collected or sent.
- **State:** `localStorage` only, one namespaced key, versioned (bump the key when structure changes).
- **Engine shape:** a `STATIONS` map of render functions, a `go(id)` state machine, `award(dim, pts, why)` as the single scoring entry point, and a `nextStep()` function that encodes the guide's routing per register. Content lives in config objects (`DIMS`, `COMM`, `TABS`, checkpoint options) — builders touch config, not engine.
- **Accessibility floor:** real `<button>`s, `aria-label` on meaningful SVG, no color-only information, no horizontal scroll at 390px, animations under `@keyframes` short enough not to trap attention.
- **Verification before ship:** every door clicked, every checkpoint answered, rubric grows, console clean, mobile clean. (The reference build was verified with a headless-browser script; manual clicking is acceptable.)
