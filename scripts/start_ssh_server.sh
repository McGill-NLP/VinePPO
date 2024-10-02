#!/bin/bash

SSHDIR=$HOME/ssh_files

mkdir -p $SSHDIR

chmod go-w $HOME
chmod 700 $HOME/.ssh
chmod 600 $HOME/.ssh/authorized_keys

ssh-keygen -q -N "" -t dsa -f $SSHDIR/ssh_host_dsa_key
ssh-keygen -q -N "" -t rsa -b 4096 -f $SSHDIR/ssh_host_rsa_key
ssh-keygen -q -N "" -t ecdsa -f $SSHDIR/ssh_host_ecdsa_key
ssh-keygen -q -N "" -t ed25519 -f $SSHDIR/ssh_host_ed25519_key

cat >$SSHDIR/sshd_config <<EOF
## Use a non-privileged port
Port 6322
## provide the new path containing these host keys
HostKey $SSHDIR/ssh_host_rsa_key
HostKey $SSHDIR/ssh_host_ecdsa_key
HostKey $SSHDIR/ssh_host_ed25519_key
## Enable DEBUG log. You can ignore this but this may help you debug any issue while enabling SSHD for the first time
LogLevel DEBUG3
ChallengeResponseAuthentication no
UsePAM no
X11Forwarding yes
PrintMotd no
## Provide a path to store PID file which is accessible by normal user for write purpose
PidFile $SSHDIR/sshd.pid
AcceptEnv LANG LC_*
Subsystem       sftp    $CONDA_PREFIX/libexec/sftp-server
AuthorizedKeysFile	.ssh/authorized_keys
EOF

cat >$HOME/.profile <<EOF
export LD_LIBRARY_PATH=/.singularity.d/libs
EOF

chmod 600 $SSHDIR/*
chmod 644 $SSHDIR/sshd_config
chown -R $USER $SSHDIR

while getopts ":b:" opt; do
  case $opt in
    :)
      $CONDA_PREFIX/bin/sshd -D -f $SSHDIR/sshd_config -E $SSHDIR/sshd.log
      exit 0
      ;;
  esac
done

$CONDA_PREFIX/bin/sshd -f $SSHDIR/sshd_config -E $SSHDIR/sshd.log