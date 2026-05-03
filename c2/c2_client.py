"""
Per-host watcher (telemetry agent).

Runs as a long-lived systemd unit on every scenario host. Tails local
sensor sources and forwards alerts to the controller-side C2 server
via the PushAlert RPC. Sensor sources:

  - /var/log/auth.log     -> ssh_login alerts (sshd session opened)
  - inotify on /usr/bin   -> suid_modify alerts when a watched suid
                              wrapper is touched
  - inotify on cron path  -> cron_modify alerts when /var/backup.sh is
                              modified (world-writable cron-as-root vuln)
  - inotify on cowrie tty -> trap_fired alerts when an attacker enters
                              the Cowrie SSH honeypot
  - inotify on data path  -> trap_fired alerts when a flagged data
                              asset is read or modified

The agent also serves as the host's liveness signal: a heartbeat
every poll interval keeps the host_up flag set on the controller.
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
import time
from typing import Optional

try:
    import grpc
    from . import c2_pb2 as pb
    from . import c2_pb2_grpc as pbg
except ImportError as e:
    raise ImportError(
        "grpc and the C2 protobuf bindings are required. "
        "Run c2/generate_bindings.sh after `pip install grpcio-tools`."
    ) from e

try:
    import inotify.adapters
except ImportError:
    inotify = None  # type: ignore


_LOG = logging.getLogger("c2.client")


# ---------------------------------------------------------------------------
# Watch sources
# ---------------------------------------------------------------------------


WATCH_PATHS = {
    "suid_modify":  ["/usr/bin/script1"],
    "cron_modify":  ["/var/backup.sh"],
    "trap_fired":   [
        "/home/user/cowrie/var/lib/cowrie/tty/",
        "/root/important_data.txt",
    ],
}

AUTH_LOG = "/var/log/auth.log"


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------


class C2Client:
    def __init__(self, controller_endpoint: str, host_id: str, heartbeat_s: float = 5.0):
        self.controller_endpoint = controller_endpoint
        self.host_id = host_id
        self.heartbeat_s = heartbeat_s

        self.channel = grpc.insecure_channel(controller_endpoint)
        self.stub = pbg.C2Stub(self.channel)
        self._running = True

    def stop(self) -> None:
        self._running = False

    # ---- alert push ----
    def _push(self, kind: str, detail: str) -> None:
        try:
            req = pb.PushAlertReq(
                host_id=self.host_id, kind=kind, detail=detail, ts=time.time(),
            )
            self.stub.PushAlert(req, timeout=2.0)
        except Exception as e:
            _LOG.debug("push_alert failed: %s", e)

    # ---- watchers ----
    def _watch_inotify(self, kind: str, paths: list) -> None:
        if inotify is None:
            _LOG.warning("inotify not available; %s watcher disabled", kind)
            return
        ino = inotify.adapters.Inotify()
        for path in paths:
            try:
                ino.add_watch(path)
            except Exception as e:
                _LOG.debug("add_watch %s failed: %s", path, e)
        for event in ino.event_gen(yield_nones=False):
            if not self._running:
                break
            (_header, type_names, watch_path, filename) = event
            if any(t in ("IN_MODIFY", "IN_CREATE", "IN_OPEN", "IN_CLOSE_WRITE") for t in type_names):
                self._push(kind, f"{watch_path}/{filename}")

    def _watch_authlog(self) -> None:
        if not os.path.exists(AUTH_LOG):
            _LOG.warning("%s not present; ssh_login watcher disabled", AUTH_LOG)
            return
        with open(AUTH_LOG, "r") as f:
            f.seek(0, os.SEEK_END)
            while self._running:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                if "sshd" in line and "session opened" in line:
                    self._push("ssh_login", line.strip())
                elif "Failed password" in line:
                    self._push("ssh_login_failed", line.strip())

    def _heartbeat(self) -> None:
        while self._running:
            try:
                self.stub.Hello(pb.HelloReq(nonce=int(time.time()) & 0xFFFFFF), timeout=2.0)
            except Exception:
                pass
            time.sleep(self.heartbeat_s)

    def run(self) -> None:
        threads = []
        for kind, paths in WATCH_PATHS.items():
            t = threading.Thread(target=self._watch_inotify, args=(kind, paths), daemon=True)
            t.start()
            threads.append(t)
        threads.append(threading.Thread(target=self._watch_authlog, daemon=True))
        threads[-1].start()
        threads.append(threading.Thread(target=self._heartbeat, daemon=True))
        threads[-1].start()
        _LOG.info("c2_client started: host=%s endpoint=%s", self.host_id, self.controller_endpoint)
        try:
            while self._running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.stop()
        _LOG.info("c2_client exiting")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", required=True, help="controller host (resolves :50051)")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--host-id", default=socket.gethostname())
    parser.add_argument("--heartbeat-s", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    endpoint = f"{args.controller}:{args.port}"
    client = C2Client(endpoint, args.host_id, args.heartbeat_s)
    client.run()


if __name__ == "__main__":
    main()
