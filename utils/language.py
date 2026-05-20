from __future__ import annotations

import hashlib
from typing import Sequence

import torch


def language_ids(languages: Sequence[str], vocab_size: int, device: torch.device | None = None) -> torch.Tensor:
    ids = []
    for language in languages:
        text = str(language).strip().lower()
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        ids.append(int.from_bytes(digest, byteorder="little") % int(vocab_size))
    return torch.tensor(ids, dtype=torch.long, device=device)
