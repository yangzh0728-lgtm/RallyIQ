# Review sheets, train_v6

Everything here was read by me (Claude) before it was used; sam only needs to spot-check. Every label decision has a
one-line reason: `labels/v6_decisions.txt` (train labels), `labels/d/*.txt` (fresh test videos),
`labels/shuttle_v6/hp_decisions.txt` (detector hard examples), `scripts/htruth.py` (doubles hitter truth).

## What changed since v5

No shipped change touches the shuttle tracks or the player boxes (the fine-tuned detector was tested and rejected, see
REPORT.md), so **the 68 v5 contact sheets are unchanged**: they are copied to `v5_tracks/` with their v5 counts below.
The one shipped change (doubles hitter from the partner's swing) changes which player a hit is credited to; the sheets draw
the labelled hitter, not the predicted one, so they do not change. New sheets this round:

| folder | sheets | what |
|---|---|---|
| `fresh/` | 52 | 3 never-labelled videos (own_20260529_0005 / 0007 / 0009), 3 fixed 40 s windows each; every pipeline hit (P) and rules candidate (R); 3 frames -0.2 / 0 / +0.2 s, last 0.25 s of track in yellow, cyan ring = track point now |
| `unlabelled_detections/` | 8 | hits that new tracks found inside already-labelled windows with no label (30 rows) |
| `hitter/` | 39 | doubles hits: `none_*` = labels without a hitter id (30), `chk_*` = label ids that disagreed with the pipeline (9); raw pose boxes numbered, green = tracked |
| `shuttle_labels/` | 10 | detector hard examples: `mos_*` 377 gap frames (100 px crops, red ring = interpolated estimate), `easy*` 2 x 48 random easy positives |
| `0245_white_wall/` | 4 | S4 / S15 / S19 of 0245 with every detector candidate, plus the zoomed racket-local search for S15 |

## Remaining known issues (unchanged from v5 unless said)

- 8 rows where the circle is on another court's shuttle **between rallies** (0617_02 r4#9, 0618_00 r4#0/#1, 0618_03 r4#13,
  0619_10 r4#14, 0619_14 r4b#34, 0616_01 S122 NEXT, 0616_03 S130 NEXT): a hit-anchored gate would remove them but also
  3.7-11% of our own hand-checked trajectory frames, so it was not shipped.
- 0245 S4 / S15 / S19: shuttle not visible at the contact (white shuttle, white wall, window light), not even zoomed.
- Fresh videos: 7 rows where the tracked shuttle is on the next court between rallies; 17 false hits, mostly between rallies
  (a player tapping, scooping or tossing the shuttle over while the other is not ready); 18 missed hits (fast net exchanges
  and partly hidden far player).
- Doubles hitter: 4 of 134 truth hitters have no track at the hit (2 hidden behind the partner, 1 detected but not
  tracked, 1 ambiguous). After the partner rule 2 of 103 detected hits still go to the wrong player.

## Fresh videos (`fresh/`), per sheet

Columns: rows; hits (my label); rows the pipeline counted as a hit; hits it missed; false hits (my label: not a hit);
unsure (ignored in scoring); rows where the tracked shuttle is the next court's.

| sheet | rows | hits | pipeline hits | missed hits | false hits | unsure | next-court shuttle tracked | note |
|---|---|---|---|---|---|---|---|---|
| own_20260529_0005_00 | 5 | 2 | 1 | 1 | 0 | 0 | 0 | #1 missed |
| own_20260529_0005_01 | 5 | 3 | 2 | 1 | 0 | 0 | 0 | #5 missed |
| own_20260529_0005_02 | 5 | 0 | 1 | 0 | 1 | 0 | 0 | #12 false hit: near player tosses the shuttle over to the far player (pass, |
| own_20260529_0005_03 | 5 | 1 | 0 | 1 | 0 | 0 | 0 | #19 missed |
| own_20260529_0005_04 | 5 | 2 | 2 | 0 | 0 | 1 | 0 |  |
| own_20260529_0005_05 | 5 | 1 | 1 | 0 | 0 | 0 | 0 |  |
| own_20260529_0005_06 | 5 | 1 | 1 | 0 | 0 | 0 | 0 |  |
| own_20260529_0005_07 | 5 | 2 | 2 | 0 | 0 | 0 | 0 |  |
| own_20260529_0005_08 | 5 | 1 | 1 | 0 | 0 | 0 | 0 |  |
| own_20260529_0005_09 | 5 | 2 | 2 | 0 | 0 | 0 | 0 |  |
| own_20260529_0005_10 | 5 | 3 | 2 | 1 | 0 | 0 | 0 | #52 missed |
| own_20260529_0005_11 | 5 | 1 | 2 | 0 | 1 | 0 | 0 | #56 false hit: near player lobs the shuttle across to the far player by han |
| own_20260529_0005_12 | 5 | 2 | 5 | 0 | 0 | 3 | 0 |  |
| own_20260529_0005_13 | 5 | 2 | 3 | 0 | 1 | 0 | 0 | #65 false hit: shuttle drops on the far side after #64, near player standin |
| own_20260529_0005_14 | 3 | 2 | 2 | 0 | 0 | 0 | 0 |  |
| own_20260529_0007_00 | 5 | 2 | 1 | 1 | 0 | 1 | 0 | #2 missed |
| own_20260529_0007_01 | 5 | 1 | 0 | 1 | 0 | 0 | 0 | #5 missed |
| own_20260529_0007_02 | 5 | 0 | 0 | 0 | 0 | 0 | 2 |  |
| own_20260529_0007_03 | 5 | 2 | 2 | 1 | 0 | 2 | 0 | #17 missed |
| own_20260529_0007_04 | 5 | 3 | 1 | 2 | 0 | 1 | 0 | #22 missed; #23 missed |
| own_20260529_0007_05 | 5 | 3 | 3 | 1 | 0 | 1 | 0 | #28 missed |
| own_20260529_0007_06 | 5 | 3 | 3 | 0 | 0 | 0 | 0 |  |
| own_20260529_0007_07 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260529_0007_08 | 5 | 4 | 3 | 1 | 0 | 0 | 0 | #43 missed |
| own_20260529_0007_09 | 5 | 2 | 3 | 0 | 1 | 0 | 0 | #45 false hit: shuttle dropping to the far player at the net; near player n |
| own_20260529_0007_10 | 5 | 0 | 1 | 0 | 1 | 0 | 0 | #52 false hit: near player throws the new shuttle by hand to the far player |
| own_20260529_0007_11 | 5 | 0 | 1 | 0 | 1 | 0 | 2 | #58 false hit: far player taps the shuttle up by himself; near player has h |
| own_20260529_0007_12 | 5 | 1 | 3 | 0 | 2 | 0 | 0 | #60 false hit: far player taps the shuttle over between rallies while the n; #61 false hit: it drops to the floor, near player scoops it (pipeline hit = |
| own_20260529_0007_13 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260529_0007_14 | 5 | 3 | 3 | 0 | 0 | 0 | 0 |  |
| own_20260529_0007_15 | 5 | 3 | 3 | 1 | 0 | 2 | 0 | #77 missed |
| own_20260529_0007_16 | 5 | 1 | 1 | 0 | 0 | 1 | 0 |  |
| own_20260529_0007_17 | 5 | 0 | 1 | 0 | 1 | 0 | 1 | #85 false hit: near player (walked to the net) scoops the shuttle up from t |
| own_20260529_0007_18 | 3 | 2 | 1 | 1 | 0 | 0 | 0 | #91 missed |
| own_20260529_0009_00 | 5 | 2 | 4 | 0 | 2 | 0 | 0 | #2 false hit: far player watches the shuttle land / catches it (rally over; #3 false hit: far player juggles, near player walking away |
| own_20260529_0009_01 | 5 | 3 | 1 | 2 | 0 | 0 | 0 | #6 missed; #7 missed |
| own_20260529_0009_02 | 5 | 0 | 1 | 0 | 1 | 0 | 0 | #11 false hit: near player throws the shuttle across to the far player by h |
| own_20260529_0009_03 | 5 | 1 | 3 | 0 | 1 | 1 | 0 | #15 false hit: juggle between rallies (far player standing behind the near  |
| own_20260529_0009_04 | 5 | 4 | 3 | 1 | 0 | 0 | 0 | #20 missed |
| own_20260529_0009_05 | 5 | 1 | 1 | 1 | 1 | 1 | 0 | #26 missed; #27 false hit: shuttle falls to the floor beside the near player (rally end |
| own_20260529_0009_06 | 5 | 2 | 3 | 0 | 1 | 0 | 0 | #31 false hit: near player (at the net, boxed far) scoops the shuttle up fr |
| own_20260529_0009_07 | 5 | 1 | 2 | 0 | 0 | 1 | 0 |  |
| own_20260529_0009_08 | 5 | 2 | 2 | 0 | 0 | 1 | 0 |  |
| own_20260529_0009_09 | 5 | 5 | 4 | 1 | 0 | 0 | 0 | #46 missed |
| own_20260529_0009_10 | 5 | 3 | 4 | 0 | 0 | 1 | 0 |  |
| own_20260529_0009_11 | 5 | 4 | 4 | 0 | 0 | 0 | 0 |  |
| own_20260529_0009_12 | 5 | 0 | 2 | 0 | 2 | 0 | 0 | #62 false hit: near player scoops the shuttle up from the floor (pipeline h; #63 false hit: shuttle passed across between rallies; far player is walking |
| own_20260529_0009_13 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260529_0009_14 | 5 | 4 | 4 | 0 | 0 | 0 | 0 |  |
| own_20260529_0009_15 | 5 | 0 | 0 | 0 | 0 | 0 | 2 |  |
| own_20260529_0009_16 | 5 | 0 | 2 | 0 | 0 | 2 | 0 |  |
| own_20260529_0009_17 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| **total** | **253** | **87** | **97** | **18** | **17** | **19** | **7** | |

## Unlabelled detections (`unlabelled_detections/`), per sheet

| sheet | rows | hit | not a hit | unsure | next-court shuttle |
|---|---|---|---|---|---|
| r_00 | 4 | 0 | 2 | 2 | 1 |
| r_01 | 4 | 4 | 0 | 0 | 0 |
| r_02 | 4 | 3 | 1 | 0 | 0 |
| r_03 | 4 | 1 | 1 | 2 | 1 |
| r_04 | 4 | 2 | 1 | 1 | 0 |
| r_05 | 4 | 1 | 3 | 0 | 0 |
| r_06 | 4 | 2 | 1 | 1 | 1 |
| r_07 | 2 | 1 | 1 | 0 | 0 |
## Hitter truth (`hitter/`)

30 `none_*` strips: 26 hitters picked and tracked by v5, 1 picked but detected-only (0241 286.77, untracked), 2 hidden
behind the partner with no detection at the hit (0241 262.86, 0622 308.44), 1 ambiguous (0310_01 305.64).
9 `chk_*` strips: 6 label ids were wrong after the v4 -> v5 id remap (0241 134.87 / 136.67 / 278.06 / 289.82,
0622 82.98 / 312.14), 2 ambiguous (0310_01 14.08, 0239 138.89), 1 confirmed (0310_01 474.79, still a pipeline error).

## Shuttle hard examples (`shuttle_labels/`)

| sheet | accepted | visible but off the estimate | not visible |
|---|---|---|---|
| mos_00 | 29 | 3 | 16 |
| mos_01 | 23 | 5 | 20 |
| mos_02 | 26 | 7 | 15 |
| mos_03 | 26 | 5 | 17 |
| mos_04 | 31 | 5 | 12 |
| mos_05 | 22 | 8 | 18 |
| mos_06 | 23 | 2 | 23 |
| mos_07 | 26 | 5 | 10 |
| **total** | **206** | **40** | **131** |
| easy_00 (48 random track points) | 43 on the shuttle | 5 off (logo, second shuttle; all > 5 px off a smooth path -> filter added) | |
| easy2_00 (48, after the filter) | 48 | 0 | |

## v5 sheets (`v5_tracks/`), counts carried over from train_v5 (tracks unchanged)

Columns as in train_v5: a = trail includes another court's shuttle, b = circle on another court's shuttle at AT HIT or NEXT,
c = bystander boxed, m = our shuttle missed at the hit. My reading of every sheet: `v5_tracks/check_v5.txt`.

| sheet | a before | b before | c before | a after | b after | c after | m after | note |
|---|---|---|---|---|---|---|---|---|
| own_20260302_07_00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | spectator in the foreground is not boxed (was not before either) |
| own_20260302_07_01 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | S67: was a next-court shuttle (b), now the circle sits on the smash high above the near player |
| own_20260310_01_00 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | far court tiny; 4 players boxed, the man on the far left (next court) is not |
| own_20260320_0239_00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | line judge by the net post no longer boxed; 4 players |
| own_20260320_0239_01 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | S58: AT HIT trail on the right is our shuttle after the lunge (by eye, unsure) |
| own_20260324_0241_00 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | near pair boxed inside the court; nobody by the bleachers |
| own_20260324_0241_01 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | sam's sheet: P3n no longer on the bystander by the bleachers (both NEXT panels): the 4 boxes are the 4 players |
| own_20260324_0241_02 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |  |
| own_20260324_0241_03 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |  |
| own_20260324_0245_00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260324_0245_01 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | S4: no shuttle at the hit (the old track mixed two shuttles there; the mixed one is now dropped) |
| own_20260324_0245_02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260324_0245_03 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | S15: near jump smash, no shuttle at the hit (white walls, shuttle against the window light); automatic count flagged S14/S15 as (a), by eye the trails are ours |
| own_20260324_0245_04 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | S19: no circle at the near player's contact |
| own_20260324_0245_05 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260324_0245_06 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | S24: near smash, the drawn trail is the far side's earlier flight (one gap), circle at the top on our shuttle |
| own_20260324_0245_07 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0616_00 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | S118 and S119 now on the real shuttle (sam's two marks) |
| own_20260906_0616_01 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | S122 NEXT +1.50: circle on a shuttle in the background left behind P2n |
| own_20260906_0616_02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0616_03 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | S130 NEXT +1.50: arc high upper left is the next court's (sam's mark); hit panel right |
| own_20260906_0616_04 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0616_05 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0617_00 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | r4#0-#3: trails now follow our shuttle (sam's 0617 rows); r4#2 label 48.41 is 0.3 s late (contact 48.11, retimed in amateur_v4) |
| own_20260906_0617_01 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0617_02 | 1 | 2 | 0 | 0 | 1 | 0 | 0 | r4#9 (not a hit, between rallies): circle and arc at the girl in white behind P1n, next court |
| own_20260906_0617_03 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0618_00 | 1 | 0 | 2 | 0 | 2 | 0 | 0 | r4#0 NEXT and r4#1 AT HIT: next court's shuttle between rallies (r4#1 is the not-a-hit row for exactly that); r4#0 relabelled a hit (lift, answers the serve at 8.6 s) |
| own_20260906_0618_01 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0618_02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0618_03 | 3 | 1 | 0 | 0 | 1 | 0 | 0 | r4#13 (not a hit, between rallies): circle on the next court's shuttle at the left, and again in NEXT |
| own_20260906_0619_00 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | r4#0 is a dead ball (relabelled not_a_hit); r4#1 not-a-hit row now shows no next-court shuttle |
| own_20260906_0619_01 | 0 | 2 | 2 | 0 | 0 | 0 | 0 |  |
| own_20260906_0619_02 | 0 | 2 | 4 | 0 | 0 | 0 | 0 | r4#10/#11 flagged by the automatic count but by eye the circle is on our shuttle by the racket |
| own_20260906_0619_03 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |  |
| own_20260906_0619_04 | 3 | 1 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0619_05 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0619_06 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0619_07 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0619_08 | 1 | 2 | 0 | 2 | 0 | 0 | 0 | r4b#19 NEXT and r4b#20 AT HIT (both not-a-hit rows, between rallies): the drawn arc is the next court's, high on the left |
| own_20260906_0619_09 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0619_10 | 1 | 3 | 0 | 0 | 1 | 0 | 0 | r4#14 (not a hit, standing between rallies): arc drawn across our air space, from the next court (passes between our two players, so the ownership test cannot tell) |
| own_20260906_0619_11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0619_12 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |  |
| own_20260906_0619_13 | 1 | 2 | 1 | 0 | 0 | 0 | 0 | r4b#28 arc is our shuttle lobbed back after the pickup (dead ball, correctly a non-hit) |
| own_20260906_0619_14 | 0 | 4 | 2 | 0 | 1 | 0 | 0 | r4b#34 (not a hit): arc across our court between rallies, labelled next court's; same case as 0619_10 r4#14 |
| own_20260906_0619_15 | 2 | 2 | 1 | 0 | 0 | 0 | 0 | r4b#38: the old next-court arc is gone; the row is our lift at 412.61 (relabelled in amateur_v4) |
| own_20260906_0621_00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | now tracked as singles (P1n/P2f); titles keep the old doubles ids (P3/P4) |
| own_20260906_0621_01 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0621_02 | 2 | 1 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0621_03 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0621_04 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | r4#13/#14: the old next-court shuttle is gone; this is our serve (147.68, label retimed); r4#13 row dropped in amateur_v4 |
| own_20260906_0621_05 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0621_06 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0621_07 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0621_08 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0621_09 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | r4c#5 (automatic count flagged it): by eye the circle is on our smash at the far racket |
| own_20260906_0621_10 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |  |
| own_20260906_0621_11 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |  |
| own_20260906_0622_00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | near partner on the left (standing, racket down) not boxed in S140-S142: an untracked player, not a bystander boxed |
| own_20260906_0622_01 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0622_02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0622_03 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0622_04 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | S157: the long orange trail runs on into the near player's lift (next hit capped at 2.5 s), all our shuttle |
| own_20260906_0622_05 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0622_06 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0622_07 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| own_20260906_0622_08 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | S173: the hitter is now boxed (P3f, the far partner was untracked before) |

Changed sheets: every sheet was regenerated; the ones whose before/after counts differ are those with a nonzero before column.
Remaining known issues (after): see rows with a/b/m > 0. The b rows left are all between rallies (nobody of ours hitting), and the
a rows are next-court arcs that pass between our two players, which the ownership test cannot tell from ours.
