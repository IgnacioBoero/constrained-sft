"""
Offline augmentation for the SAFETY experiment dataset.

Creates N augmented copies of ONLY the safe rows (safety_label == True) by applying
synonym replacement (nlpaug WordNet SynonymAug) to selected prompt fields.

Output: JSONL files you can load with `datasets.load_dataset("json", data_files=...)`.

Example:
  python -m src.scripts.augment_safety_dataset \
    --source-dataset ihounie/skillmix-safe-1k \
    --out-dir ./src/datasets/safety/skillmix-safe-1k-aug3 \
    --num-aug 3 \
    --seed 42
    --no-augment-validation

Dependencies:
  pip install -e '.[augmentation]'
  python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from datasets import load_dataset


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int,)):
        return bool(int(x))
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y"}
    return bool(x)


def _maybe_augment_text(aug, text: str, seed: int, prob: float, min_words: int) -> str:
    if not text or not isinstance(text, str):
        return text
    if len(text.split()) < min_words:
        return text

    import random
    import numpy as np

    r = random.Random(seed)
    if r.random() >= prob:
        return text

    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed & 0xFFFFFFFF)

    try:
        out = aug.augment(text)
    except LookupError as exc:
        raise RuntimeError(
            "Required NLTK data not found for synonym augmentation. Run:\n"
            "  python -c \"import nltk; "
            "nltk.download('wordnet'); nltk.download('omw-1.4'); "
            "nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')\""
        ) from exc
    except Exception:
        return text

    if isinstance(out, list):
        return out[0] if out else text
    if isinstance(out, str):
        return out
    return text


def augment_split(
    rows: Iterable[Dict],
    *,
    num_aug: int,
    seed: int,
    prob: float,
    fields: Sequence[str],
    min_words: int,
    max_aug_words: int,
    max_rows: int | None,
) -> List[Dict]:
    try:
        import nlpaug.augmenter.word as naw
    except Exception as exc:
        raise RuntimeError(
            "Offline augmentation requires optional deps. Install with "
            "`pip install -e '.[augmentation]'` (needs `nlpaug` + `nltk`)."
        ) from exc

    aug = naw.SynonymAug(aug_src="wordnet", aug_p=0.2, aug_max=max_aug_words)

    out: List[Dict] = []
    for i, ex in enumerate(rows):
        if max_rows is not None and i >= max_rows:
            break

        out.append(dict(ex))  # keep original row

        is_safe = _to_bool(ex.get("safety_label", False))
        if not is_safe:
            continue

        # Create N augmented variants of the safe example.
        for k in range(num_aug):
            ex2 = dict(ex)
            for f in fields:
                if f not in ex2:
                    continue
                val = ex2.get(f)
                if val is None:
                    continue
                if not isinstance(val, str):
                    val = str(val)
                mixed_seed = (seed * 1_000_003 + i * 10_007 + k * 1_001 + (hash(f) & 0xFFFF)) & 0xFFFFFFFF
                ex2[f] = _maybe_augment_text(aug, val, int(mixed_seed), prob, min_words)
            ex2["augmented"] = True
            ex2["aug_id"] = k
            out.append(ex2)
    return out


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dataset", type=str, default="ihounie/skillmix-safe-1k")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--num-aug", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prob", type=float, default=1.0)
    p.add_argument("--fields", type=str, default="instruction,input")
    p.add_argument("--min-words", type=int, default=3)
    p.add_argument(
        "--max-aug-words",
        type=int,
        default=8,
        help="Maximum number of words to augment per field (passed to nlpaug as aug_max).",
    )
    p.add_argument("--max-rows", type=int, default=None, help="Optional cap for debugging.")
    p.add_argument(
        "--no-augment-validation",
        action="store_true",
        help="If set, do not create augmented copies for the validation split (only keep originals).",
    )
    p.add_argument(
        "--download-nltk",
        action="store_true",
        help="Attempt to download required NLTK data packages before augmenting.",
    )
    args = p.parse_args()

    if args.num_aug < 0:
        raise ValueError("--num-aug must be >= 0")
    if not (0.0 <= args.prob <= 1.0):
        raise ValueError("--prob must be in [0, 1]")
    if args.max_aug_words < 1:
        raise ValueError("--max-aug-words must be >= 1")

    fields = [x.strip() for x in args.fields.split(",") if x.strip()]

    if args.download_nltk:
        import nltk

        for pkg in [
            "wordnet",
            "omw-1.4",
            "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng",
        ]:
            nltk.download(pkg)

    ds = load_dataset(args.source_dataset)
    if "train" not in ds or "validation" not in ds:
        raise RuntimeError(
            f"Expected dataset to have train/validation splits, got: {list(ds.keys())}"
        )

    out_dir = Path(args.out_dir)
    train_rows = augment_split(
        ds["train"],
        num_aug=args.num_aug,
        seed=args.seed,
        prob=args.prob,
        fields=fields,
        min_words=args.min_words,
        max_aug_words=args.max_aug_words,
        max_rows=args.max_rows,
    )
    val_rows = augment_split(
        ds["validation"],
        num_aug=0 if args.no_augment_validation else args.num_aug,
        seed=args.seed + 1,  # different but deterministic
        prob=args.prob,
        fields=fields,
        min_words=args.min_words,
        max_aug_words=args.max_aug_words,
        max_rows=args.max_rows,
    )

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "validation.jsonl", val_rows)

    print(f"Wrote {len(train_rows)} train rows to {out_dir/'train.jsonl'}")
    print(f"Wrote {len(val_rows)} validation rows to {out_dir/'validation.jsonl'}")


if __name__ == "__main__":
    # Avoid tokenizers parallelism warnings in some environments.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

