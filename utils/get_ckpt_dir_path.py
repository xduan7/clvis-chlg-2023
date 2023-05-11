import hashlib
import json
import os

import nni


def get_ckpt_dir_path(args) -> str:
    if nni.get_experiment_id() == "STANDALONE":
        _sorted_args = dict(sorted(vars(args).items()))
        # Device is irrelevant for checkpointing
        _sorted_args.pop("device", None)
        _sorted_args_str = json.dumps(_sorted_args)
        _sha256_hash = hashlib.sha256(
            _sorted_args_str.encode("utf-8")
        ).hexdigest()
        _ckpt_dir_path = f"./checkpoints/{_sha256_hash[:8]}"
    else:
        _ckpt_dir_path = (
            f"./checkpoints/{nni.get_experiment_id()}/{nni.get_trial_id()}"
        )

    os.makedirs(_ckpt_dir_path, exist_ok=True)
    return _ckpt_dir_path
