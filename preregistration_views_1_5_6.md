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
