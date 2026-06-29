# Table 3 — View 1 summary (top-20 input-embedding neighbors)

Token counts in the top-20 input-embedding neighbors of each item, from `data/processed/view1/view1_<item>.json`. Each top-20 entry classified by hand into: self / case variant / inflection of the same lemma / cross-linguistic translation / within-language co-hyponym or synonym / emoji / other. Wording in the three narrative columns reviewed by ChatGPT before commit.

| Item | self | case | infl | trans | cohyp/syn | emoji | other | Initial neighborhood pattern | Strength | Major constraint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| water | 1 | 2 | 1 | 16 | 0 | 0 | 0 | translation-dominated | 16/20 cross-linguistic equivalents; 0 co-hyponyms | Strong — cleanest cross-linguistic signal in the set |
| brother | 1 | 2 | 3 | 8 | 6 | 0 | 0 | mixed translation + within-language kinship | 8/20 translations; 6/20 kinship co-hyponyms (sister, sibling, cousin) | Within-language kinship co-hyponyms compete for neighbor slots |
| stone | 1 | 2 | 4 | 11 | 1 | 0 | 1 | translation-dominated with one co-hyponym | 11/20 translations; 1 co-hyponym (rock) | Negligible co-hyponym competition; 1 fragment token ("ston") |
| fear | 1 | 2 | 4 | 9 | 4 | 0 | 0 | mixed translation + within-language emotion synonyms | 9/20 translations; 4/20 synonyms (afraid, scared, fearful, worry) | Strong synonym competition characteristic of abstract emotions |
| dog | 1 | 2 | 3 | 9 | 5 | 0 | 0 | mixed translation + within-language co-hyponyms | 9/20 translations; 5/20 co-hyponyms (canine, puppy, pet, doggy, Canine) | Co-hyponym layer (canine/puppy/pet) intervenes between source and translations |
| bread | 1 | 2 | 1 | 8 | 6 | 2 | 0 | mixed translation + co-hyponyms + emoji intrusion | 8/20 translations; 6/20 co-hyponyms (loaf, breadcrumbs, bakery, sourdough); 2 emoji (🍞, 🥖) | Co-hyponym and emoji intrusion; only 8 cross-linguistic neighbors in top 20 |
