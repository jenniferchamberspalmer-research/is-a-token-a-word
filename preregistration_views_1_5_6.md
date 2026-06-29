# Pre-registration: Cross-linguistic sign identity across network depth (Views 1, 5, 6)

Status: design fixed before data collection. Any change after the first logged run is recorded in the Deviations section with a date, not edited silently.

Model: Gemma 2 2B with Gemma Scope SAEs. Activations bare-word, BOS excluded, pooling = last, matching View 1 word-level framing. Layers 0 through 25.

## Purpose

Test, across a small set of items presented as cases rather than as a general law, whether a word's cross-linguistic translation equivalents share representational structure that a matched control word's equivalents do not, and how that structure changes with network depth. The unit of analysis is the word. The finding is scope-limited to the items run.

## Items (n = 6)

Two are already run and replicated; four are proposed for adjudication.

| # | Item | Class | Translation cognate density | Status |
|---|------|-------|------------------------------|--------|
| 1 | water | mass / elemental | mixed (agua, eau, Wasser, acqua, água) | run, replicated |
| 2 | brother | kinship / relational | mixed (hermano, frère, Bruder, fratello, irmão) | run, replicated under chair |
| 3 | dog | concrete animate | low, non-cognate (perro, chien, Hund, cane, cão) | proposed |
| 4 | bread | artifact / food | mixed (pan, pain, Brot, pane, pão) | proposed; ties to View 2/3 dissociation work |
| 5 | stone | natural kind / mass | mixed (piedra, pierre, Stein, pietra, pedra) | proposed; Stratum A item |
| 6 | fear | abstract | low to mixed (miedo, peur, Angst, paura, medo) | proposed; tests whether method extends past concrete nouns |

Rationale for the set: range across semantic class (mass, kinship, animate, artifact, natural kind, abstract) without claiming the set is exhaustive. dog is included as a deliberate form-confound control item, because its five translations share almost no surface form, so any neighborhood sharing it shows cannot be reduced to orthographic similarity. fear is the one stretch item; if it fails it is reported as a failure, not dropped.

Cognate-density column is a stated interpretive caution, not an exclusion. For high-cognate items, early-layer clustering could be orthographic rather than semantic, and that reading is reported alongside the result.

## Controls

Three control schemes are run, each applied identically to all six items. Every item is therefore tested against three independent controls across all layers. The robustness evidence is agreement across the three: a separation that holds against all three controls is a property of the item, not of any one reference word.

| Run | Control | Translations | Class | Cognate density | Status |
|-----|---------|--------------|-------|-----------------|--------|
| A | chair | silla, chaise, Stuhl, sedia, cadeira | household artifact | low | validated on water and brother |
| B | window | ventana, fenêtre, Fenster, finestra, janela | household artifact | medium (fenêtre, Fenster, finestra share fenestra) | unvalidated |
| C | bird | pájaro, oiseau, Vogel, uccello, pássaro | animate natural kind | very low, non-cognate | unvalidated |

Rationale. Run A is the validated low-cognate artifact baseline. Run B holds semantic class fixed but raises cognate density, testing whether the effect survives a control whose own translations share more surface form. Run C changes semantic class entirely while keeping cognate density low, so the three controls bracket the items rather than clustering. table is excluded from all three because its cognate-inflated translations raise the control band and mask the item signal, the confound that suppressed the brother result before chair replaced table.

Stated caveat on Run C. bird is unvalidated, and animate natural kinds may themselves carry cross-linguistic neighborhood structure, so bird's control band may come in higher than chair's. That makes bird a conservative control. If bird behaves like an item rather than a baseline, that is reported as a finding, not concealed.

## Views and metrics logged

View 1, embedding neighbors. Top k = 50, static embedding, cosine. Two passes: unfiltered multilingual and English-only filtered (validated filter, precision/recall/specificity 1.0). Per neighbor: rank, cosine, language, code. Reported per item: count of cross-linguistic translation equivalents in top k and their ranks, ranks of English paradigmatic co-hyponyms, and presence or absence of synonyms and collocates in the filtered set.

View 5, single-word pairwise trajectory. Per layer 0 to 25: cosine and rank of each translation equivalent to the source word. Reported per item: the layer range over which translations are nearest neighbors, and the layer at which rank collapses.

View 6, neighborhood dynamics. Top N = 100 primary, N = 50 and N = 200 as mandatory robustness. Per layer 0 to 25, for source translations, within-language synonyms, and control equivalents: Jaccard, reciprocal-rank-weighted overlap, mean rank of shared neighbors, count of shared neighbors. Headline per item: per-layer translation-mean Jaccard against control-mean Jaccard, with band separation reported as translation-minimum versus control-maximum.

## Interpretive frame, fixed before any output is read

What each view measures and what counts as evidence. View 5 is a pairwise-identity measure: per layer, whether each cross-linguistic equivalent is among the source word's nearest neighbors and at what rank. It takes no control. Its role is to locate the depth at which cross-linguistic equivalence is geometrically legible. Evidence consistent with the negative-differential reading is early-layer pairwise identity followed by collapse. The result counting against it is translations remaining nearest neighbors across all layers with no collapse, which is the persistent computed equivalence a shared-representation account predicts. View 6 is the neighborhood-dynamics measure and the only view carrying a control. Its role is concept-specificity: whether the source word's translations share neighborhoods more than a control word's translations do. Evidence for the effect is separation above control across layers 0 to 9 against all three controls; no separation, or separation against only one or two, is reported as control-dependent.

View 5 curve reading, decided in advance. The layer curve is read as legibility-by-depth, the depth at which cross-linguistic equivalence is geometrically present, and nothing more. It is not narrated as a developmental trajectory of meaning forming, and not read as convergence toward a shared semantic object. A rising or consolidating segment is not described as meaning consolidating, because that is the semantic-hub or developmental-gradient reframe that remains unadjudicated from a prior session and is not adopted here. Positive distributional proximity of translations is held separate from negative differential value; the two are not collapsed. The Wu et al. semantic-hub work is referenced as contrast, not alignment. The reframe is held formally open, not resolved by the shape of a curve.

Item-set status. This is a cross-class case set, not a stratum. It ranges deliberately across semantic class, mass (water, stone, bread), kinship (brother), animate (dog), abstract (fear), to test whether cross-linguistic differential identity is class-general. It is presented as cases, selected for range, not as a systematic or exhaustive stratification.

Relation to the dissociation spine. The salt and bread rank reversal lives in the View 2/3 behavioral and feature data, a different view battery, and stands in its own record. The present View 1/5/6 study measures neither behavior nor internal features and therefore does not carry, weaken, or relocate the dissociation. Bread appearing in both is a connective thread, not a transfer of the dissociation onto bread alone. The dissociation remains anchored where it was generated, in the salt and bread View 2/3 data, with salt retained in that record.

Role of the controls. chair, window, and bird function as both foil and baseline, but only at the distributional level. As foil they are the direct comparison in the View 6 contrast; as baseline they establish what an ordinary noun's cross-linguistic neighborhood-sharing looks like. They are not a baseline for differential structure in the Saussurean sense. The control contrast is positive-distributional evidence about neighborhood overlap and does not by itself demonstrate negative-differential value, which remains a theoretical inference held at a different level.

Stated limitation. The three controls are concrete count nouns, so for brother and fear the control is class-mismatched, and separation there could partly reflect class difference rather than cross-linguistic identity. Those two items are read with that caution, and a class-matched control is named as the cleaner follow-up.

## Analysis plan, fixed in advance

Primary pre-registered test: early-layer concept-specificity. For each item, mean Jaccard across the five translations must exceed mean Jaccard across the five control equivalents across layers 0 to 9, and must do so against all three controls. This is the robust regime and the load-bearing claim, because a large effect there surviving three independent controls cannot be a single control's idiosyncrasy.

Cross-control agreement is the headline robustness measure. For each item and layer band, report the separation against chair, window, and bird side by side. An item is reported as passing only where it separates from all three. Where the three disagree, the result is reported as control-dependent.

Secondary, exploratory: mid-network neighborhood persistence, layers 10 to 13. Reported as suggestive, not confirmed, given small effect sizes. The three-control design matters most here, since this is the band where a single control could manufacture an effect.

Reporting standard: effect sizes and direction, not significance stars. Per-item results presented as cases. No aggregation across items into a single scalar, and no claim that the pattern generalizes to words not run.

## Falsification conditions, stated up front

- An item fails early-layer concept-specificity if its translation-mean Jaccard does not exceed control-mean across layers 0 to 9 against any one of the three controls. A failing item is reported as failing.
- If the three controls give materially different separation for an item, that item's effect is reported as control-dependent rather than as a property of the item.
- The View 5 pairwise-collapse pattern is absent for an item if translations remain nearest neighbors through mid-network rather than collapsing. Absence is reported.
- If bird's own translation Jaccard tracks the items rather than sitting below them, bird is reported as failing to function as a baseline, and the cross-linguistic identity is provisionally read as broader than the item set.

## Run order and compute

View 6 sweeps every layer for every translation and control equivalent, so three controls across six items is the bulk of the cost. To avoid burning compute on a misconfigured pipeline, run water against all three controls first as the validated anchor and confirm the outputs before proceeding. Then brother against all three. Then the four new items. Within each item, View 1 and View 5 are cheap relative to View 6.

Desktop browser for cold-start runs. Mobile tab-switching drops the websocket and kills queued tasks.

## Deviations log

(empty at pre-registration; date-stamped entries added here if the design changes after the first logged run)
