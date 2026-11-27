# Review sheets, train_v7

All sheets read by me (Claude); sam only needs to spot-check. Label decisions with reasons: `labels/iso_decisions.txt`
(100 dead-time detections), `../train_v6/labels/d/own_20260529_0007.txt` row 93 (one fresh-set hit added in v7).

Shipped this round: the alternation side-flip rule and the dead-ball filter (events only). Shuttle tracks and player boxes
are unchanged, so every v5/v6 contact sheet is unchanged; the new sheets show exactly the hits that changed.

| folder | sheets | what |
|---|---|---|
| `diff/` | 7 | every hit in the labelled windows of the 17 own videos that v7 added or removed vs v6 (27 rows), 4 frames -0.3/-0.1/+0.1/+0.3 s, track yellow; title = added/removed : label |
| `fresh_diff/` | 3 | same for the 3 held-out fresh videos (13 rows) |
| `iso/` | 25 | 100 isolated pipeline hits outside the labelled windows, labelled by me as dead-time training examples (59 not a hit / 32 hit / 9 unsure) |
| `fresh_269.jpg` | 1 | the one fresh-set detection that had no label row (0007 269.20): a real near overhead, added |

## Counts per sheet

### diff/ (17 own videos, labelled windows)

| sheet | rows | true hits added | not-a-hit removed | true hits removed | false hits added | unsure / ignored | unlabelled |
|---|---|---|---|---|---|---|---|
| r_00 | 4 | 0 | 2 | 0 | 0 | 2 | 0 |
| r_01 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| r_02 | 4 | 2 | 1 | 0 | 0 | 1 | 0 |
| r_03 | 4 | 0 | 0 | 0 | 0 | 3 | 1 |
| r_04 | 4 | 0 | 2 | 0 | 0 | 2 | 0 |
| r_05 | 4 | 1 | 2 | 0 | 0 | 0 | 1 |
| r_06 | 3 | 2 | 0 | 0 | 0 | 1 | 0 |

### fresh_diff/ (held-out videos)

| sheet | rows | true hits added | not-a-hit removed | true hits removed | false hits added | unsure / ignored | unlabelled |
|---|---|---|---|---|---|---|---|
| r_00 | 5 | 2 | 2 | 0 | 0 | 1 | 0 |
| r_01 | 5 | 1 | 1 | 2 | 0 | 1 | 0 |
| r_02 | 3 | 1 | 1 | 0 | 1 | 0 | 0 |

Notes: diff r_03 #13 is qa6 0615 707.21, a new detection with no label; to me it looks like a real near forehand, but qa6
stays unrelabelled so it counts as a false hit. diff r_05 #21 (0619 253.59, removed, unlabelled) is the v5 "next-court arc
between rallies" row r4#14: correctly gone. fresh_diff r_01 #6/#7 are true hits lost (0007 129.10 near overhead, 270.62 far
lift) against #3/#4/#5/#10 gained; #11 (0009 939.34) is a new false hit (caught shuttle).

## Remaining known issues

- Missed hits with no candidate at all (15 of 106) and candidates the scorer rates low (56): mostly fast net exchanges,
  partly hidden far players, white-wall 0245 contacts.
- The 8 between-rally next-court rows of v5 minus 0619 r4#14 (now removed by the dead-ball filter): 7 left.
- Doubles: 4 of 111 detected hits credited to the wrong player (2 have no track at all).
