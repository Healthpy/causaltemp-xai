#!/usr/bin/env python3
import argparse
import json
import platform
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    assert platform.machine() == "aarch64"
    assert torch.cuda.is_available()

    values = torch.arange(4096, device="cuda", dtype=torch.float32)
    result = (values * values).sum()
    torch.cuda.synchronize()

    report = {
        "status": "PASS",
        "architecture": platform.machine(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "result": result.item(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
