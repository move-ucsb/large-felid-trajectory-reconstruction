#!/bin/bash
set -euo pipefail
echo "whoami: $(whoami)"
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "---- df ----"
df -h .
df -h ~
echo "---- project size ----"
du -h --max-depth=1 . 2>/dev/null | sort -hr | head -30
echo "---- output size ----"
du -h --max-depth=2 outputs 2>/dev/null | sort -hr | head -50 || true
echo "---- largest files ----"
find . -type f -printf "%s %p\n" 2>/dev/null | sort -nr | head -30 | awk '{printf "%.2f GB  ", $1/1024/1024/1024; $1=""; print $0}'
