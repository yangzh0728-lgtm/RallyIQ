# RallyIQ train-v9: own stroke labels, style uncertainty, net hits (2026-11-27)

Branch `claude/train-v9` = `claude/train-v8` + this work (not merged; nothing on main). Work folder `~/Desktop/RallyIQ/train_v9/`
(PLAN.md with every pre-registration, LOG.md, labels/, review/final/INDEX.md, scripts/, ckpt/). Fresh set test-only, qa6 untouched.

## Summary

| item | before | after | shipped? |
|---|---|---|---|
| Own stroke labels: sure rows (11 non-qa6 videos) | 146 | **197** (+51; 157 typed, 23 corrected, 111 left unknown) | labels (`amateur_v9`) |
| Stroke type, trusted LOVO, shipped-feature residual (189 trusted rows) | – | 77.2% (court_v2 alone 64.6%) | already in code |
| Residual v2 (+ time of flight, receiver depth / movement, early speed, wrist speed) | 77.2% | 74.1% | no |
| Shipped residual refit on the larger label set, fresh set (78 typed) | 55.6% | 52.4% | no |
| Style shares: 95% CI width over rallies (22 player-sessions, median) | not reported | 11–24 points | **yes**: the report now states it (540c5fa) |
| Net-zone admission rule for missed net hits | 0.8604 | 0.8604 (no hit changed) | no |

## N: own stroke data
I reviewed every labelled own hit whose stroke was not sure (268) on contact + next-hit sheets and typed 157 of them, marking
sure only where the flight and the receiver leave no doubt. 111 stay unknown: tangled or partial tracks, doubles net exchanges,
depth along the rear camera's line of sight. The brief's target of 400 trusted rows is not supported by what the footage shows,
and I did not guess. Two pre-registered model tests both lost: a residual with depth-robust features (74.1% vs 77.2%) and the
shipped residual refit on the larger set (fresh 52.4% vs 55.6%). The bundled residual stays; the new labels are kept for later.

## O: playing style
Rally-bootstrap 95% intervals of the stroke-mix shares are wide: smash 19, net 20, clear 16, drop 11, lift/block 24, attack 20,
defence 11 points (median widths; 29–121 shots per player-session). The near player in the S.YANG jersey across 0616 / 0619 /
0621: smash 18 / 17 / 21%, net 24 / 35 / 17%, attack 26 / 22 / 24%; every share's intervals overlap. One session rarely
separates two style prototypes. **Shipped:** the style report's share metrics carry their interval in the note
("95% 区间 14–39%（按回合重抽样）", seeded, 200 resamples). Proposal: pool a player's sessions for the verdict once the app
has a cross-video player identity.

## P: missed net-exchange hits
The pre-registered rule (lower admission and keep threshold for candidates at the net with a wrist at the shuttle) fired on few
candidates and changed no hit. The missed net hits mostly lack a close wrist-shuttle point at the contact or are scored below
0.05; they need a better view of the contact itself.

## Tests
448 passed, 12 skipped (full suite).
