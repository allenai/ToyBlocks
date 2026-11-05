#!/usr/bin/env bash

set -e

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}"

SHOW_HELP=false

MODE=generate-task
USE_DEBUGGER=false
NO_UI=false

command -v fzf >/dev/null 2>&1 && HAS_FZF=true || HAS_FZF=false
command -v jq >/dev/null 2>&1 && HAS_JQ=true || HAS_JQ=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) SHOW_HELP=true; shift ;;
        --mode) MODE="$2"; shift 2 ;;
        --debug) USE_DEBUGGER=true; shift ;;
        --no-ui) NO_UI=true; shift ;;
        *) echo "Unkown option: $1. Run with --help for options"; exit 1 ;;
    esac
done
