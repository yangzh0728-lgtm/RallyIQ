# train_v11 review protocol (pre-labels checked row by row)

Goal: every shuttle contact by a racket during play in the whole video is labelled (hit / hitter / side / stroke);
everything else on the sheets is labelled not_a_hit. sam only spot-checks afterwards, so be careful and honest:
mark `unsure` instead of guessing.

## Sheets
`review/sheets/<KEY>/<KEY>_NNN.jpg`, 6 rows per sheet, rows numbered `#n` (global per video; sheet NNN holds rows
6*NNN .. 6*NNN+5). Row header: `#n time_s SRC hitter/side stroke`:
- SRC letters: P = production pipeline hit, L = scorer at a low threshold, R = rules candidate (no learned scorer).
  Rows with several letters were found by several sources. These are only *guesses* of the model.
- hitter = pipeline player id (P1..P4) of the box drawn on the frame (green box = near half of the court, i.e. the
  camera side of the net; orange = far half). `-` = no hitter assigned.
- stroke = production stroke classifier guess (only on P rows).
- Panels: -0.2 s, row time, +0.2 s, +0.4 s. Yellow dots = shuttle detections of the last 0.25 s, cyan ring = shuttle
  detection at that panel's time (detections can be wrong: a white shoe, a light, another court's shuttle).
- GAP rows (`#n GAP a-b: missed hit?`): 4 frames evenly inside a stretch of a rally where no candidate exists. Say whether a
  hit happens inside the gap (approximate time from the panel times), who hits and what stroke, or `none`.

## What counts
- hit: racket contacts the shuttle during a rally, including serves, net shots, mis-hits/frames that go over or not, and
  a rally's last shot. Contact time = the moment the shuttle leaves the racket (give the best estimate from the panels,
  e.g. "+0.1 s after the row time" -> row time + 0.1).
- not_a_hit: shuttle in flight with no contact, bounce on the floor, pickup, the shuttle passed/tossed/scooped between
  rallies (even with the racket, when nobody is playing: warm-up feeding, handing back), juggling while waiting, a shuttle
  on a neighbouring court, false detection, practice drills where one player hand-feeds (the hand throw is not a hit, the
  racket contact is). Rows within 0.25 s of a hit you already labelled on another row: `dup #m`.
- unsure_hit: cannot tell whether a contact happened (occluded, out of frame, too blurred).
- A hit needs the racket at (or swinging through) the shuttle and a change of the shuttle's direction or speed. A track
  that changes direction with no swing nearby is a false detection or a landing: not_a_hit.
- A hit you can see that has no row of its own (e.g. between two rows): add an extra line with the nearest row's number and
  label `missed_hit`, your best contact time estimate, side, hitter, stroke. These get a refine sheet later.
- Warm-up hitting before the match counts as hits if both players are hitting a continuous exchange over the net
  (it is play from the analysis point of view). Say "warm-up" in the reason.

## Fields per row (one line per row, every row of every sheet)
`n | label | t | side | hitter | stroke | sure | reason`
- label: hit / not_a_hit / unsure_hit / dup#m / gap_hit / gap_none / missed_hit
- t: contact time in seconds (for hits); for not_a_hit the row time.
- side: near / far (the half of the court the hitter stands in; near = camera side). `-` if not a hit.
- hitter: the pipeline id of the box around the hitter at contact (`P1`.. as drawn), `none` if the hitter has no box or
  the box drawn there is around someone else.
- stroke: short_serve, long_serve, clear, smash, drop, drive, net_shot, lift, push, block, or `unknown`.
  Definitions: serve = first hit of a rally (short_serve low and short over the net, long_serve high and deep);
  clear = overhead, high and deep to the back; smash = overhead, steep and fast down; drop = overhead or high contact,
  slow and falling short; drive = flat fast shot at about shoulder/net height; net_shot = from near the net, stays close
  over the net; lift = from low (often near the net / mid court), goes up and deep; push = soft flat push from
  the forecourt/mid court to mid court; block = soft return of a smash/drive, short.
- sure: `sure` when hit, side, hitter and stroke are all clear; otherwise `unsure:<which>` e.g. `unsure:stroke`,
  `unsure:hitter`, `unsure:hit`.
- reason: short, concrete visual evidence ("shuttle at the racket at 0, rises toward the far court at +0.2";
  "shuttle on the floor"; "same contact as #12").

Write the file `labels/raw/<KEY>.txt` with a header line `# n | label | t | side | hitter | stroke | sure | reason`.
