#!/usr/bin/env bash
# Check a local copy of the release against scripts/release-42ed227.tsv: every listed file present with the right size,
# and every LFS file (the weights) with the right sha256. Reads only, at low priority. Usage:
#   bash scripts/check_weights.sh MiniMax-H3 [parallel jobs, default 1]
# On a machine shared with other services start with 1 job and watch them before raising it.
set -euo pipefail
root="${1:?weights dir}"
jobs="${2:-1}"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/h3-check.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
here="$(cd "$(dirname "$0")" && pwd)"
list="$here/release-42ed227.tsv"
fail=0
while IFS=$'\t' read -r path size sha; do
  [[ -z "$path" || "$path" == \#* ]] && continue
  f="$root/$path"
  if [[ ! -f "$f" ]]; then echo "MISSING $path"; fail=1; continue; fi
  have=$(stat -f %z "$f" 2>/dev/null || stat -c %s "$f")
  if [[ "$have" != "$size" ]]; then echo "SIZE $path have $have want $size"; fail=1; fi
done < "$list"
# sha256 of the LFS files, in parallel
grep -v '^#' "$list" | awk -F'\t' '$3 != "" {print $3 "  " $1}' > "$tmp/expected.sha256"
(cd "$root" && awk '{print $2}' "$tmp/expected.sha256" | xargs -P "$jobs" -n 1 nice -n 15 shasum -a 256) \
  | sort -k2 > "$tmp/actual.sha256"
if diff <(sort -k2 "$tmp/expected.sha256") "$tmp/actual.sha256"; then
  echo "sha256: $(wc -l < "$tmp/actual.sha256" | tr -d ' ') weight files match"
else
  echo "sha256: MISMATCH (diff above)"; fail=1
fi
[[ $fail == 0 ]] && echo "OK: all listed files present, sizes and sha256 match" || { echo "FAILED"; exit 1; }
