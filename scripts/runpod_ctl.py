#!/usr/bin/env python3
"""Tiny RunPod control helper: create / status / wait / terminate.

Usage:
    python scripts/runpod_ctl.py create   [--name yi-phonikud]
    python scripts/runpod_ctl.py wait     <pod_id>
    python scripts/runpod_ctl.py status   [<pod_id>]
    python scripts/runpod_ctl.py balance
    python scripts/runpod_ctl.py terminate <pod_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "data" / "scratch" / "runpod_pod.json"

for _line in (REPO / ".env").read_text().splitlines():
    if "=" in _line and not _line.strip().startswith("#"):
        _k, _, _v = _line.strip().partition("=")
        os.environ.setdefault(_k, _v.strip().strip("\"'"))

KEY = os.environ["RUNPOD_API_KEY"]
H = {"Authorization": f"Bearer {KEY}"}
REST = "https://rest.runpod.io/v1"
GQL = f"https://api.runpod.io/graphql?api_key={KEY}"

IMAGE = "runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204"
# Preference order: the user asked for a 3090; fall back to the closest 24GB cards.
GPUS = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
]


def gql(query: str):
    return requests.post(GQL, json={"query": query}, timeout=60).json()


def balance():
    return gql("{myself{clientBalance currentSpendPerHr pods{id desiredStatus costPerHr}}}")


def create(name: str, clouds=("COMMUNITY", "SECURE")):
    pubkey = Path.home().joinpath(".ssh/id_ed25519.pub").read_text().strip()
    last = None
    for cloud in clouds:
        for gpu in GPUS:
            body = {
                "name": name,
                "imageName": IMAGE,
                "gpuTypeIds": [gpu],
                "gpuCount": 1,
                "cloudType": cloud,
                "containerDiskInGb": 60,
                "volumeInGb": 0,
                "ports": ["22/tcp"],
                "env": {"PUBLIC_KEY": pubkey},
                "supportPublicIp": True,
            }
            r = requests.post(f"{REST}/pods", headers=H, json=body, timeout=120)
            if r.status_code in (200, 201):
                pod = r.json()
                pod["_requested_gpu"] = gpu
                pod["_cloud"] = cloud
                STATE.write_text(json.dumps(pod, indent=2))
                print(f"created {pod['id']}  {gpu}  {cloud}  ${pod.get('costPerHr')}")
                return pod
            last = f"{cloud} {gpu}: {r.status_code} {r.text[:200]}"
            print(f"  no luck -> {last}")
    sys.exit(f"could not create any pod. last error: {last}")


def get(pod_id: str):
    return requests.get(f"{REST}/pods/{pod_id}", headers=H, timeout=60).json()


def wait(pod_id: str, timeout_s: int = 900):
    """Pods report RUNNING ~2.5 min before SSH is reachable. Poll for publicIp AND
    a port mapping for 22 -- desiredStatus is not trustworthy."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        p = get(pod_id)
        ip = p.get("publicIp")
        pm = p.get("portMappings") or {}
        if ip and pm.get("22"):
            info = {"id": pod_id, "ip": ip, "port": pm["22"], "costPerHr": p.get("costPerHr"),
                    "gpu": p.get("machine", {}).get("gpuTypeId") or p.get("gpuTypeId"),
                    "elapsed_s": round(time.time() - t0)}
            prev = json.loads(STATE.read_text()) if STATE.exists() else {}
            STATE.write_text(json.dumps(prev | info, indent=2))
            print(json.dumps(info))
            return info
        print(f"  [{round(time.time()-t0)}s] status={p.get('desiredStatus')} ip={ip} ports={pm}")
        time.sleep(15)
    sys.exit("timed out waiting for ssh")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["create", "wait", "status", "terminate", "balance"])
    ap.add_argument("pod_id", nargs="?")
    ap.add_argument("--name", default="yi-phonikud")
    ap.add_argument("--cloud", default=None, choices=["COMMUNITY", "SECURE"])
    a = ap.parse_args()

    pid = a.pod_id
    if pid is None and STATE.exists():
        pid = json.loads(STATE.read_text()).get("id")

    if a.cmd == "create":
        pod = create(a.name, (a.cloud,) if a.cloud else ("COMMUNITY", "SECURE"))
        pid = pod["id"]
    elif a.cmd == "wait":
        wait(pid)
    elif a.cmd == "status":
        print(json.dumps(get(pid), indent=2)[:3000])
    elif a.cmd == "balance":
        print(json.dumps(balance(), indent=2))
    elif a.cmd == "terminate":
        r = requests.delete(f"{REST}/pods/{pid}", headers=H, timeout=60)
        print(f"delete {pid} -> {r.status_code}")
        print(json.dumps(balance(), indent=2))


if __name__ == "__main__":
    main()
