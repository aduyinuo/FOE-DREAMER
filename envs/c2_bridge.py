"""
gRPC bridge between the FOE-Dreamer training loop and the C2 server.

The defender's five-element action set and per-host observation vector
defined in §4.2 of the manuscript are applied here. Each host
contributes five features per polling step; the defender chooses one
of five actions (with an optional variant for Deceive).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import grpc
    from c2 import c2_pb2 as pb
    from c2 import c2_pb2_grpc as pbg
except ImportError:
    grpc = None
    pb = None
    pbg = None


_LOG = logging.getLogger(__name__)


# Five-element defender action vocabulary (§4.2).
DEFENDER_ACTIONS = (
    "isolate",
    "disrupt",
    "monitor",
    "deceive",
    "null",
)
ACTION_NAME_TO_ID = {a: i for i, a in enumerate(DEFENDER_ACTIONS)}


# Five-feature per-host observation schema (§4.2).
@dataclass
class HostState:
    host_id: str
    alerts_count: int
    honeypot_active: bool
    host_up: bool
    trap_fired: bool
    last_event_age_s: float


def host_state_to_vec(h: HostState) -> np.ndarray:
    return np.asarray(
        [
            float(h.alerts_count),
            float(h.honeypot_active),
            float(h.host_up),
            float(h.trap_fired),
            float(h.last_event_age_s),
        ],
        dtype=np.float32,
    )


HOST_FEATURE_DIM = 5


class C2Bridge:
    """gRPC client bound to the controller's C2 service."""

    def __init__(
        self,
        host: str,
        port: int,
        host_inventory: List[str],
        action_timeout_s: float = 5.0,
        connect_retries: int = 6,
        connect_backoff_s: float = 1.5,
    ):
        if grpc is None or pb is None or pbg is None:
            raise ImportError("grpc and the C2 protobuf bindings are required")
        self._pb, self._pbg = pb, pbg
        self.host_inventory = list(host_inventory)
        self.host_index = {h: i for i, h in enumerate(self.host_inventory)}
        self.action_timeout_s = action_timeout_s

        endpoint = f"{host}:{port}"
        last_err: Optional[Exception] = None
        for attempt in range(connect_retries):
            try:
                self.channel = grpc.insecure_channel(endpoint)
                grpc.channel_ready_future(self.channel).result(timeout=connect_backoff_s)
                break
            except Exception as e:
                last_err = e
                _LOG.warning("c2 connect attempt %d failed: %s", attempt + 1, e)
                time.sleep(connect_backoff_s * (1 + attempt))
        else:
            raise ConnectionError(f"could not connect to C2 at {endpoint}: {last_err}")

        self.stub = pbg.C2Stub(self.channel)
        self.ping()

    def close(self) -> None:
        try:
            self.channel.close()
        except Exception:
            pass

    def ping(self) -> bool:
        try:
            req = self._pb.HelloReq(nonce=int(time.time()) & 0xFFFFFF)
            resp = self.stub.Hello(req, timeout=self.action_timeout_s)
            return resp.nonce == req.nonce + 1
        except Exception as e:
            _LOG.debug("c2 ping failed: %s", e)
            return False

    def get_host_state(self, host_id: str) -> HostState:
        req = self._pb.HostStateReq(host_id=host_id)
        resp = self.stub.GetHostState(req, timeout=self.action_timeout_s)
        return HostState(
            host_id=resp.host_id,
            alerts_count=resp.alerts_count,
            honeypot_active=resp.honeypot_active,
            host_up=resp.host_up,
            trap_fired=resp.trap_fired,
            last_event_age_s=resp.last_event_age_s,
        )

    def dispatch_action(
        self, host_id: str, action: str, params: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, str]:
        if action not in ACTION_NAME_TO_ID:
            raise ValueError(f"unknown defender action: {action}")
        req = self._pb.ActionReq(host_id=host_id, action=action, params=(params or {}))
        resp = self.stub.Dispatch(req, timeout=self.action_timeout_s)
        return resp.success, resp.message

    def get_alerts_since(self, ts: float) -> List[dict]:
        req = self._pb.AlertReq(since_ts=ts)
        resp = self.stub.GetAlerts(req, timeout=self.action_timeout_s)
        return [
            {"ts": a.ts, "host": a.host_id, "kind": a.kind, "detail": a.detail}
            for a in resp.alerts
        ]

    def collect_observation(self) -> np.ndarray:
        vecs = [host_state_to_vec(self.get_host_state(h)) for h in self.host_inventory]
        return np.concatenate(vecs, axis=0)

    def apply_action_index(
        self, host_idx: int, action_idx: int, variant: Optional[str] = None
    ) -> Tuple[bool, str]:
        host = self.host_inventory[host_idx]
        action = DEFENDER_ACTIONS[action_idx]
        params = {"variant": variant} if variant else None
        return self.dispatch_action(host, action, params)

    def reset(self) -> None:
        req = self._pb.ResetReq()
        self.stub.Reset(req, timeout=self.action_timeout_s)
