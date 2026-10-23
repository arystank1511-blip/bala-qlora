"""Save runtime details and the exact inference/training code with the results."""

import hashlib
import importlib.metadata
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch

    results = ROOT / "results"
    source = results / "source"
    files = [ROOT / "requirements.txt", ROOT / "requirements-dev.txt"]
    for folder in ("bala", "scripts"):
        files.extend(sorted((ROOT / folder).glob("*.py")))
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoints = []
    for path in (ROOT / "outputs/checkpoints").glob("checkpoint-*/trainer_state.json"):
        state = json.loads(path.read_text())
        checkpoints.append({key: state.get(key) for key in
                            ("global_step", "epoch", "best_metric", "best_model_checkpoint")})
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets", "sacrebleu")},
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "source_sha256": hashes,
        "checkpoints": sorted(checkpoints, key=lambda x: x["global_step"]),
    }
    (results / "runtime.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
