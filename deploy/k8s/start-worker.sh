#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "${EG_NODE_NAME:-unknown}" > /tmp/easygraph-node
ssh-keygen -q -t ed25519 -N '' -f /tmp/ssh_host_ed25519_key
exec /usr/sbin/sshd -De -f /home/mpiuser/.sshd_config
