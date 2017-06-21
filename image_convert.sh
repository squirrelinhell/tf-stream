#!/bin/bash

INFO=$'
Usage:

\timage_convert.sh H,W [ ... ImageMagick options ... ]
\timage_convert.sh H,W,3 [ ... ImageMagick options ... ]
'

SIZE=
FMT=

case "$1" in
    *,*,*,*)
        echo "$INFO" 1>&2
        exit 1
    ;;
    *,*,3)
        SIZE="${1%,*}"
        SIZE="${SIZE##*,}x${SIZE%%,*}"
        FMT="RGB"
    ;;
    *,*,*)
        echo "$INFO" 1>&2
        exit 1
    ;;
    *,*)
        SIZE="${1##*,}x${1%%,*}"
        FMT="GRAY"
    ;;
    *)
        echo "$INFO" 1>&2
        exit 1
    ;;
esac

BYTES=$((${1//,/*})) || exit 1
shift

OUTPUT="${@: -1}"
case "x$OUTPUT" in
    x) OUTPUT="-depth 8 $FMT:-" ;;
    x-) OUTPUT='' ;;
    x-*) OUTPUT="-depth 8 $FMT:-" ;;
    *) OUTPUT='' ;;
esac

TMPDIR=$(mktemp -d) || exit 1
trap "rm -rf $TMPDIR" EXIT

while dd bs=1 count=$BYTES of="$TMPDIR/chunk" status=none ; do
    convert \
        -size "$SIZE" -depth 8 "$FMT:$TMPDIR/chunk" \
        "$@" \
        $OUTPUT || exit 1
done || exit 1

