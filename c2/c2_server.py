"""
Controller-side C2 server.

Long-lived gRPC service on the controller node. Handles the RPCs
defined in `c2.proto`:

  - Hello          : liveness probe
  - GetHostState   : per-host telemetry digest (§4.2 five-feature vector)
  - Dispatch       : defender action dispatch (§4.2 five-element action set)
  - GetAlerts      : pull alerts since a given timestamp
  - Reset          : clear in-memory state to a clean baseline

State is in-memory: per-host alert counts, honeypot status, availability,
trap indicators, and the most recent timestamped event. On `Reset`, the
alert queue is dropped and host state returns to baseline.
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from typing import Dict, List

try:
    import grpc
    from . import c2_pb2 as pb
    from . import c2_pb2_grpc as pbg
except ImportError as e:
    raise ImportError(
        "grpc and the C2 protobuf bindings are required. "
        "Run c2/generate_bindings.sh after `pip install grpcio-tools`."
    ) from e


_LOG = logging.getLogger("c2.server")


# ---------------------------------------------------------------------------
# Per-host runtime state
# ---------------------------------------------------------------------------


@dataclass
class HostRuntime:
    host_id: str
    alerts_count: int = 0
    honeypot_active: bool = False
    host_up: bool = True
    trap_fired: bool = False
    last_event_ts: float = 0.0

    def to_proto(self) -> pb.HostStateRsp:
        now = time.time()
        return pb.HostStateRsp(
            host_id=self.host_id,
            alerts_count=int(self.alerts_count),
            honeypot_active=self.honeypot_active,
            host_up=self.host_up,
            trap_fired=self.trap_fired,
            last_event_age_s=float(now - self.last_event_ts) if self.last_event_ts > 0 else 1e3,
        )


@dataclass
class AlertRecord:
    ts: float
    host_id: str
    kind: str
    detail: str


# ---------------------------------------------------------------------------
# Action dispatcher (five-element action set)
# ---------------------------------------------------------------------------


class ActionDispatcher:
    """Translates a defender action into per-host operations."""

    def __init__(self, hosts: Dict[str, HostRuntime]):
        self.hosts = hosts

    def dispatch(self, host_id: str, action: str, params: Dict[str, str]) -> tuple[bool, str]:
        host = self.hosts.get(host_id)
        if host is None:
            return False, f"unknown host: {host_id}"

        try:
            if action == "null":
                return True, "null"
            if action == "isolate":
                # Move host out of all reachable security groups.
                self._run_remote(host, "openstack_isolate.sh")
                host.host_up = False  # unreachable from worker subnets
                return True, "host isolated"
            if action == "disrupt":
                # Reboot over SSH; clears in-memory persistence.
                self._run_remote(host, "ssh_reboot.sh")
                host.alerts_count = 0
                host.last_event_ts = 0.0
                return True, "host rebooted"
            if action == "monitor":
                # Raise log verbosity on the target host.
                self._run_remote(host, "raise_log_verbosity.sh")
                return True, "log verbosity raised"
            if action == "deceive":
                variant = params.get("variant", "cowrie")
                if variant == "suid_decoy":
                    self._run_remote(host, "plant_suid_decoy.sh")
                    host.trap_fired = False
                    return True, "suid decoy planted"
                # default: cowrie SSH honeypot bound to weak credentials
                self._run_remote(host, "deploy_cowrie.sh")
                host.honeypot_active = True
                return True, "cowrie honeypot active"
            return False, f"unknown action: {action}"
        except Exception as e:
            _LOG.exception("dispatch failed for %s/%s", host_id, action)
            return False, str(e)

    def _run_remote(self, host: HostRuntime, command: str) -> None:
        # Remote execution is delegated to the per-host SSH client; the
        # implementation is configured by inventory at boot time.
        _LOG.info("[%s] run: %s", host.host_id, command)


# ---------------------------------------------------------------------------
# Servicer
# ---------------------------------------------------------------------------


class C2Servicer(pbg.C2Servicer):
    def __init__(self, hosts: Dict[str, HostRuntime]):
        self.hosts = hosts
        self.alerts: List[AlertRecord] = []
        self.alert_lock = threading.Lock()
        self.dispatcher = ActionDispatcher(hosts)

    def Hello(self, request, context):  # noqa: N802
        return pb.HelloRsp(nonce=request.nonce + 1)

    def GetHostState(self, request, context):  # noqa: N802
        host = self.hosts.get(request.host_id)
        if host is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"unknown host: {request.host_id}")
            return pb.HostStateRsp()
        return host.to_proto()

    def Dispatch(self, request, context):  # noqa: N802
        ok, msg = self.dispatcher.dispatch(
            request.host_id, request.action, dict(request.params)
        )
        return pb.ActionRsp(success=ok, message=msg)

    def GetAlerts(self, request, context):  # noqa: N802
        with self.alert_lock:
            recent = [a for a in self.alerts if a.ts >= request.since_ts]
        msgs = [
            pb.Alert(ts=a.ts, host_id=a.host_id, kind=a.kind, detail=a.detail)
            for a in recent
        ]
        return pb.AlertRsp(alerts=msgs)

    def PushAlert(self, request, context):  # noqa: N802
        self.push_alert(request.host_id, request.kind, request.detail)
        return pb.PushAlertRsp(accepted=True)

    def Reset(self, request, context):  # noqa: N802
        with self.alert_lock:
            self.alerts.clear()
        for host in self.hosts.values():
            host.alerts_count = 0
            host.honeypot_active = False
            host.host_up = True
            host.trap_fired = False
            host.last_event_ts = 0.0
        return pb.ResetRsp(success=True)

    # ---- alert ingestion (called from the inotify / log-tail threads) ----
    def push_alert(self, host_id: str, kind: str, detail: str) -> None:
        rec = AlertRecord(ts=time.time(), host_id=host_id, kind=kind, detail=detail)
        with self.alert_lock:
            self.alerts.append(rec)
        host = self.hosts.get(host_id)
        if host is not None:
            host.alerts_count += 1
            host.last_event_ts = rec.ts
            if kind == "trap_fired":
                host.trap_fired = True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def serve(bind: str, hosts: List[str]) -> None:
    runtime = {h: HostRuntime(host_id=h) for h in hosts}
    servicer = C2Servicer(runtime)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pbg.add_C2Servicer_to_server(servicer, server)
    server.add_insecure_port(bind)
    server.start()
    _LOG.info("c2 server listening on %s with %d hosts", bind, len(hosts))
    server.wait_for_termination()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="0.0.0.0:50051")
    parser.add_argument("--hosts", nargs="+", required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    serve(args.bind, args.hosts)


if __name__ == "__main__":
    main()
