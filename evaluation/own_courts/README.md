# Hand-labelled courts of own footage

`image_from_court` homographies (court metres -> pixels of a 3840x2160 frame) for own
videos whose court the automatic calibration gets wrong. Use one as a video's calibration with

    python scripts/manual_court.py evaluation/own_courts/hs0320_0239.json --video-id <vid> --data ~/rallyiq-data/...

A label made on one file of a session fits another file of the same session only if the
camera did not move: draw it on a frame of that file before trusting it.
