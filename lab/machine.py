import json
import os
import platform
import re
import subprocess
from pathlib import Path

CONFIG = Path(os.environ.get("LAB_CONFIG", Path.home() / ".config" / "lab" / "machine.json"))


def _sh(*cmd, env=None):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                              env={**os.environ, **(env or {})}).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _gpu():
    try:
        data = json.loads(_sh("system_profiler", "SPDisplaysDataType", "-json"))
        gpu = data["SPDisplaysDataType"][0]
        return int(gpu.get("sppci_cores", 0)) or None, gpu.get("spdisplays_mtlgpufamilysupport")
    except (ValueError, KeyError, IndexError):
        return None, None


def probe():
    chip = _sh("sysctl", "-n", "machdep.cpu.brand_string")
    cores, metal = _gpu()
    xcode = _sh("xcodebuild", "-version").splitlines()
    metal_cc = _sh("xcrun", "metal", "--version", env={"TOOLCHAINS": "Metal"}).splitlines()
    return {
        "chip": chip,
        "gpu_cores": cores,
        "metal_family": metal,
        "memory_gb": round(int(_sh("sysctl", "-n", "hw.memsize") or 0) / 2**30),
        "macos": platform.mac_ver()[0],
        "macos_build": _sh("sw_vers", "-buildVersion"),
        "xcode": " ".join(x.split()[-1] for x in xcode[:2]) if xcode else None,
        "metal_compiler": metal_cc[0] if metal_cc else None,
    }


def default_id(info):
    slug = re.sub(r"[^a-z0-9]+", "", (info["chip"] or "mac").lower().replace("apple", ""))
    return f"{platform.node().split('.')[0].lower()}-{slug}"


def load(new_id=None):
    saved = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    info = probe()
    info["id"] = new_id or saved.get("id") or default_id(info)
    if new_id or not CONFIG.exists():
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps({"id": info["id"]}, indent=2) + "\n")
    return info
