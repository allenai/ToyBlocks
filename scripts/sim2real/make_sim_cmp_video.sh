#!/usr/bin/env bash
# Side-by-side video of the two sim rollouts compared in the sim2real replay
# reports: no-sysid, sysid, and the two overlaid.
#
# The pair is frame-exact (same episode, same joint targets, same fps and
# resolution), so the panels are stacked with no resampling and frame N is the
# same instant in all three. The shared counter across the bottom is there to
# make that verifiable on screen rather than taken on trust.
#
# The third panel averages the two halves, so anything the two sims agree on
# renders solid and anything they disagree on splits into two ghosts. Ghosting
# is the signal: its size is the sim-to-sim divergence at that frame.
#
# Usage: make_sim_cmp_video.sh LEFT.mp4 RIGHT.mp4 OUT.mp4 "caption"
set -euo pipefail

LEFT=$1; RIGHT=$2; OUT=$3; CAPTION=${4:-}
# drawtext parses ':' as an option separator and '\' as an escape, so an
# unescaped caption silently loses everything from its first colon onwards.
CAPTION=${CAPTION//\\/\\\\}
CAPTION=${CAPTION//:/\\:}
CAPTION=${CAPTION//\'/\\\'}
CAPTION=${CAPTION//%/\\%}

FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
PAD=52          # header strip above each panel
FOOT=34         # shared footer strip

for f in "$LEFT" "$RIGHT"; do
  [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

# Refuse to stack a mismatched pair: silently truncating to the shorter input
# would put the rollouts out of step and quietly invalidate the overlay.
nl=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$LEFT")
nr=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$RIGHT")
if [ "$nl" != "$nr" ]; then
  echo "frame count mismatch: left $nl vs right $nr -- not the same episode?" >&2
  exit 1
fi
echo "stacking $nl frames per panel"

label() {  # label <text> <bar colour>
  printf "pad=iw:ih+%d:0:%d:black,drawbox=x=0:y=0:w=iw:h=%d:color=%s:t=fill," \
         "$PAD" "$PAD" "$PAD" "$2"
  printf "drawtext=fontfile=%s:text='%s':fontcolor=white:fontsize=25:x=(w-tw)/2:y=14" \
         "$FONT" "$1"
}

ffmpeg -y -v warning -stats \
  -i "$LEFT" -i "$RIGHT" \
  -filter_complex "\
   [0:v]split=2[l0][lo]; [1:v]split=2[r0][ro]; \
   [lo][ro]blend=all_mode=average[ov]; \
   [l0]$(label 'SIM WITHOUT SYSID' '0x1f77b4')[l]; \
   [r0]$(label 'SIM WITH SYSID'    '0xff7f0e')[r]; \
   [ov]$(label 'OVERLAY 50/50 - ghosting = divergence' '0x555555')[o]; \
   [l][r][o]hstack=inputs=3[s]; \
   [s]pad=iw:ih+${FOOT}:0:0:black, \
      drawtext=fontfile=${FONT}:text='frame %{eif\\:n\\:d}  ·  t=%{pts\\:hms}  ·  ${CAPTION}':\
fontcolor=white:fontsize=20:x=(w-tw)/2:y=h-${FOOT}+7, \
      drawbox=x=iw/3-1:y=0:w=2:h=ih:color=white@0.85:t=fill, \
      drawbox=x=2*iw/3-1:y=0:w=2:h=ih:color=white@0.85:t=fill[v]" \
  -map "[v]" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -an "$OUT"

echo "wrote $OUT"
