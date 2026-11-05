# Shared port-wait helper for the eval pipeline.
#
# Sourced by scripts/eval/run_eval_pipeline.sh (the single
# pipeline) and scripts/eval/_common.sh (split-terminal workflow). Kept in one
# place so the policy-server / queue readiness check has a single definition.

# True when something accepts a TCP connection on $1 (localhost).
#
# Uses bash's /dev/tcp rather than `ss`, which is Linux-only and silently never
# matches on macOS -- the local eval path then always timed out. This also tests
# what callers actually care about (the socket accepts) instead of parsing a
# listening table.
port_is_listening() {
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3<&- 3>&-
}

# Wait until $port is listening or the tracked $pid dies. Exits the calling
# shell (1) on timeout or early server death, so it also fires the caller's
# `trap cleanup` teardown.
wait_policy_port() {
  local port="$1" pid="$2" label="$3" ready=0
  for _ in $(seq 1 180); do
    if port_is_listening "$port"; then
      ready=1
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "policy server (${label}) exited during startup" >&2
      wait "$pid" || true
      exit 1
    fi
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    echo "timeout waiting for policy server on port ${port} (${label})" >&2
    exit 1
  fi
}
