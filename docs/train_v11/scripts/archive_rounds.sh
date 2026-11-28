#!/bin/bash
# Copy frozen rounds' data (everything except repo/) to the T7 archive; symlinks recorded in SYMLINKS.tsv, not copied.
A=/Volumes/T7/RallyIQ_training/archive
for n in 3 4 5 6 7 8 9 10; do
  S=$HOME/Desktop/RallyIQ/train_v$n; D=$A/train_v$n; mkdir -p "$D"
  ( cd "$S" && find . -path ./repo -prune -o -type l -print | while read -r l; do printf '%s\t%s\n' "$l" "$(readlink "$l")"; done ) > "$D/SYMLINKS.tsv"
  nice -n 15 rsync -rt --no-links --exclude /repo/ "$S/" "$D/" 2>>"$A/rsync_errors.log"
  echo "copied train_v$n rc=$?"
  chg=$(nice -n 15 rsync -rc --no-links --exclude /repo/ --dry-run --itemize-changes "$S/" "$D/" | grep -v '^\.d' | grep -c .)
  src=$(cd "$S" && find . -path ./repo -prune -o -type f -print | wc -l); dst=$(cd "$D" && find . -type f ! -name '._*' ! -name SYMLINKS.tsv | wc -l)
  echo "train_v$n bytes_src=$(du -sk --exclude=repo "$S" 2>/dev/null | cut -f1)k files_src=$src files_dst=$dst checksum_diffs=$chg symlinks=$(wc -l < "$D/SYMLINKS.tsv")"
done
echo ARCHIVE_DONE
