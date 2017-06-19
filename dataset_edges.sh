#!/bin/bash

INFO="
Usage:
    dataset_edges.sh <image> <output-dir>
"

if [ ! -f "$1" -o "x$2" = x ]; then
    echo "$INFO" 1>&2
    exit 1
fi

IMGSIZE=800x800
GRID=100
PATCH=256

TMPDIR=$(mktemp -d) || exit 1
trap "rm -rf $TMPDIR" EXIT

convert "$1" \
    -resize "$IMGSIZE" \
    "$TMPDIR/orig.ppm" || exit 1

convert "$TMPDIR/orig.ppm" \
    -canny 0x1+5%+20% \
    -blur 1x1 -level '0,80%' \
    "$TMPDIR/edges.ppm" || exit 1

DIM=$(convert "$TMPDIR/edges.ppm" -format "%[w] %[h]" info:) || exit 1
read W H <<<"$DIM"

OUTNUM=0
for x in $(seq 0 $GRID $(($W - $PATCH))); do
    for y in $(seq 0 $GRID $(($H - $PATCH))); do
        MEAN=$(convert "$TMPDIR/edges.ppm" \
            -crop ${PATCH}x${PATCH}+$x+$y +repage \
            -format "%[fx:round(mean*1000)]" info:) || exit 1
        [ "$MEAN" -ge 40 ] || continue
        mkdir -p "$2" || exit 1
        OUTFILE="$2/$RANDOM$RANDOM.png"
        convert "$TMPDIR/edges.ppm" \
            -crop ${PATCH}x${PATCH}+$x+$y \
            "$TMPDIR/orig.ppm" \
            -crop ${PATCH}x${PATCH}+$x+$y \
            +append +repage "$OUTFILE" || exit 1
        let OUTNUM=OUTNUM+1
    done
done

echo "'$1' -> $OUTNUM patches"
