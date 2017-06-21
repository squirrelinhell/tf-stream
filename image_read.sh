#!/bin/bash

INFO=$'
Usage:

\timage_read.sh <file or directory>
'

if [ ! -e "$1" ]; then
    echo "$INFO" 1>&2
    exit 1
fi

OUT_FMT=''
function read_fmt() {
    [ "x$OUT_FMT" = x ] || return 0
    local params=$(identify -format "%[fx:w] %[fx:h] %[channels]" "$1") \
        || exit 1
    IFS=' ' read -r w h c <<<"$params"
    case "$c" in
        gray)
            OUT_FMT="-resize ${w}x${h}! -depth 8 GRAY:-"
            echo "Image '$1' has shape ${h},${w}" 1>&2
        ;;
        *)
            OUT_FMT="-resize ${w}x${h}! -depth 8 RGB:-"
            echo "Image '$1' has shape ${h},${w},3" 1>&2
        ;;
    esac
}

if [ -d "$1" ]; then
    while IFS='' read -r f; do
        [ ! -d "$1/$f" ] || continue
        read_fmt "$1/$f"
        convert "$1/$f" $OUT_FMT || exit 1
    done <<<$(ls "$1") || exit 1
else
    read_fmt "$1"
    convert "$1" $OUT_FMT || exit 1
fi
