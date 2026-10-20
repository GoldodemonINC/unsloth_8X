# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Post-merge tokenizer validation -- the quiet failure that ships broken models.

A merge that blends weights perfectly can still produce a model that loads, runs,
and is subtly wrong, because the *tokenizer* decides what the weights ever see:

* a different ``bos_token_id`` prepends a start token the model never trained on,
* a different ``eos_token_id`` means generation never stops,
* a different ``pad_token_id`` fills batches with a real token,
* a different chat template silently changes every prompt the model is given.

None of those raise anywhere. The model just answers slightly worse, which is
indistinguishable from "the merge didn't help".

So every merge runs this check **before the first shard is written**, and a
conflict refuses the run: :func:`check_before_merge` raises
:class:`TokenizerConflictError` and :func:`merge_checkpoints` never creates the
output folder.

Only stdlib is needed: the tokens are read straight out of the checkpoint's JSON
files (``special_tokens_map.json``, ``tokenizer_config.json``, ``added_tokens.json``,
``config.json``), so the gate works on a CPU-only host with no transformers
installed. Transformers is used only, when present, to name the tokenizer class.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# The roles whose disagreement is fatal. ``unk`` is included because a different
# unknown-token id turns unrepresented text into a different embedding.
CRITICAL_ROLES = ("bos_token", "eos_token", "pad_token")
ALL_ROLES = ("bos_token", "eos_token", "pad_token", "unk_token")

_ROLE_ALIASES = {
    "bos_token": ("bos_token", "bos_token_id", "bos"),
    "eos_token": ("eos_token", "eos_token_id", "eos"),
    "pad_token": ("pad_token", "pad_token_id", "pad"),
    "unk_token": ("unk_token", "unk_token_id", "unk"),
}

_TOKEN_FILES = (
    "special_tokens_map.json",
    "tokenizer_config.json",
    "added_tokens.json",
    "config.json",
)


class TokenizerConflictError(ValueError):
    """Two checkpoints disagree about a special token the merge cannot reconcile."""

    def __init__(self, message: str, *, report: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.report = dict(report or {})


@dataclass(frozen=True)
class SpecialToken:
    """One special token, by string and id, as a checkpoint declares it."""

    role: str
    token: str | None = None
    token_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "token": self.token, "token_id": self.token_id}

    @property
    def declared(self) -> bool:
        return self.token is not None or self.token_id is not None


@dataclass(frozen=True)
class TokenizerFingerprint:
    """What a tokenizer *is*, as far as a merge cares."""

    source: str
    tokenizer_class: str | None
    vocab_size: int | None
    special: dict[str, SpecialToken] = field(default_factory=dict)
    added_token_count: int | None = None
    chat_template_sha256: str | None = None
    has_chat_template: bool = False
    files: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["special"] = {role: token.to_dict() for role, token in self.special.items()}
        return payload


@dataclass(frozen=True)
class TokenizerCheck:
    """The result of comparing fingerprints: ``ok`` | ``warning`` | ``conflict``."""

    status: str
    conflicts: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    fingerprints: tuple[TokenizerFingerprint, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status != "conflict"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ok": self.ok,
            "conflicts": list(self.conflicts),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "tokenizers": [item.to_dict() for item in self.fingerprints],
        }

    def summary(self) -> str:
        """One sentence per finding, for the job record and the UI card."""
        if self.status == "ok":
            return "Tokenizer check passed: special tokens, vocabulary and chat template agree."
        parts = list(self.conflicts) + list(self.warnings)
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Reading a checkpoint's tokenizer from disk
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _checkpoint_root(source: str | Path) -> Path:
    path = Path(source).expanduser()
    if path.is_file():
        return path.parent
    # A HF cache folder holds one snapshot per revision; take the first with config.
    if (path / "snapshots").is_dir():
        for child in sorted((path / "snapshots").iterdir()):
            if child.is_dir() and (child / "config.json").is_file():
                return child
    return path


def _first_int(payload: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
    return None


def _coerce_token(value: Any) -> tuple[str | None, int | None]:
    """``special_tokens_map`` holds strings; ``tokenizer_config`` may hold objects."""
    if value is None:
        return None, None
    if isinstance(value, str):
        return value, None
    if isinstance(value, dict):
        content = value.get("content")
        return (str(content) if content is not None else None), _first_int(value, "id")
    if isinstance(value, int) and not isinstance(value, bool):
        return None, int(value)
    return str(value), None


def _role_from_config(config: Mapping[str, Any], role: str) -> tuple[str | None, int | None]:
    token: str | None = None
    token_id: int | None = None
    for key in _ROLE_ALIASES[role]:
        if key not in config:
            continue
        if key.endswith("_id"):
            if token_id is None:
                token_id = _first_int(config, key)
            continue
        found_token, found_id = _coerce_token(config.get(key))
        token = token if token is not None else found_token
        token_id = token_id if token_id is not None else found_id
    return token, token_id


def fingerprint_tokenizer(source: str | Path) -> TokenizerFingerprint:
    """Read one checkpoint's tokenizer identity, tolerating missing files."""
    root = _checkpoint_root(source)
    files = tuple(name for name in _TOKEN_FILES if (root / name).is_file())

    # special_tokens_map.json is authoritative for strings; tokenizer_config.json
    # for ids; config.json also carries bos/eos ids for the model itself.
    special_map = _read_json(root / "special_tokens_map.json") or {}
    tokenizer_config = _read_json(root / "tokenizer_config.json") or {}
    model_config = _read_json(root / "config.json") or {}

    # ``added_tokens.json`` maps a token string to its vocabulary id. It is the
    # authority when a config names a token but does not give its id, and it is
    # what proves two differently-labelled tokens are the same token.
    added = _read_json(root / "added_tokens.json")
    added_map: dict[str, int] = (
        {
            str(key): int(value)
            for key, value in added.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        if isinstance(added, dict)
        else {}
    )
    added_count = len(added) if isinstance(added, dict) else None

    special: dict[str, SpecialToken] = {}
    for role in ALL_ROLES:
        token, token_id = _role_from_config(special_map, role)
        config_token, config_id = _role_from_config(tokenizer_config, role)
        token = token if token is not None else config_token
        token_id = token_id if token_id is not None else config_id
        if token_id is None and token is not None:
            token_id = added_map.get(token)
        if role in ("bos_token", "eos_token") and token_id is None:
            token_id = _first_int(model_config, role if role.endswith("_id") else f"{role}_id")
        special[role] = SpecialToken(role = role, token = token, token_id = token_id)

    template = tokenizer_config.get("chat_template")
    if template is None:
        # Newer checkpoints keep the template in its own file.
        if (root / "chat_template.jinja").is_file():
            template = (root / "chat_template.jinja").read_text(
                encoding = "utf-8", errors = "replace"
            )
    if isinstance(template, list):
        template = json.dumps(template, sort_keys = True)
    digest = (
        hashlib.sha256(str(template).encode("utf-8")).hexdigest() if template else None
    )

    tokenizer_class = tokenizer_config.get("tokenizer_class")
    vocab_size = (
        _first_int(tokenizer_config, "vocab_size")
        or _first_int(model_config, "vocab_size", "text_config.vocab_size")
    )

    return TokenizerFingerprint(
        source = str(root),
        tokenizer_class = str(tokenizer_class) if tokenizer_class else None,
        vocab_size = vocab_size,
        special = special,
        added_token_count = added_count,
        chat_template_sha256 = digest,
        has_chat_template = bool(template),
        files = files,
    )


# ---------------------------------------------------------------------------
# Comparing them
# ---------------------------------------------------------------------------


def _describe(token: SpecialToken) -> str:
    if token.token is not None and token.token_id is not None:
        return f"{token.token!r} (id {token.token_id})"
    if token.token_id is not None:
        return f"id {token.token_id}"
    if token.token is not None:
        return repr(token.token)
    return "not set"


def _same_role(left: SpecialToken, right: SpecialToken) -> tuple[list[str], list[str]]:
    """Compare one role, returning ``(problems, notes)``.

    The **id** is what changes behaviour: it is the vocabulary index the model
    actually emits. Two parents that declare the same id under different labels
    (``<|endoftext|>`` vs ``<|end_of_text|>``, both 151643) tokenize and generate
    identically, so that is a note, not a conflict. Only ids that differ -- or
    unlabelled tokens with no id to resolve -- are a real disagreement.
    """
    if left.token_id is not None and right.token_id is not None:
        if left.token_id != right.token_id:
            return [f"id {left.token_id} vs {right.token_id}"], []
        if left.token is not None and right.token is not None and left.token != right.token:
            return [], [
                f"both use id {left.token_id} with different labels "
                f"({left.token!r} vs {right.token!r}); behaviour is identical"
            ]
        return [], []
    problems: list[str] = []
    if left.token is not None and right.token is not None and left.token != right.token:
        problems.append(
            f"token {left.token!r} vs {right.token!r} (no vocabulary id recorded for "
            "at least one side, so they cannot be proven equal)"
        )
    return problems, []


def compare_tokenizers(
    sources: Sequence[str | Path], *, label: str = "merge"
) -> TokenizerCheck:
    """Compare every parent's tokenizer; conflicts are fatal, the rest are warnings."""
    if len(sources) < 2:
        raise ValueError("A tokenizer comparison needs at least two checkpoints.")
    fingerprints = [fingerprint_tokenizer(source) for source in sources]
    base = fingerprints[0]

    conflicts: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    for other in fingerprints[1:]:
        name = Path(other.source).name or other.source
        for role in ALL_ROLES:
            left, right = base.special[role], other.special[role]
            if not left.declared and not right.declared:
                continue
            if left.declared and right.declared:
                problems, agreements = _same_role(left, right)
                notes.extend(f"{role}: {note}" for note in agreements)
                if problems:
                    text = (
                        f"{role} disagrees between the parents: {_describe(left)} vs "
                        f"{_describe(right)} ({' and '.join(problems)})."
                    )
                    if role == "eos_token":
                        text += (
                            " Generation would stop at a different token for each parent's "
                            "text, which is the silent failure this check exists to catch."
                        )
                    elif role == "pad_token":
                        text += " Batches would be padded with a different real token."
                    if role in CRITICAL_ROLES:
                        conflicts.append(text)
                    else:
                        warnings.append(text)
                continue
            declared, missing = (left, right) if left.declared else (right, left)
            message = (
                f"{role} is set on one parent ({_describe(declared)}) and unset on {name}. "
                "The merged tokenizer will inherit the set value."
            )
            if role in CRITICAL_ROLES:
                warnings.append(message)
            else:
                notes.append(message)

        if (
            base.vocab_size
            and other.vocab_size
            and base.vocab_size != other.vocab_size
        ):
            warnings.append(
                f"Vocabulary sizes differ ({base.vocab_size} vs {other.vocab_size}); "
                "rows without a counterpart are cropped or zero-padded during the merge."
            )
        if base.tokenizer_class and other.tokenizer_class and base.tokenizer_class != other.tokenizer_class:
            warnings.append(
                f"Tokenizer classes differ ({base.tokenizer_class} vs {other.tokenizer_class})."
            )
        if base.has_chat_template != other.has_chat_template:
            warnings.append(
                f"Only one parent ships a chat template ({name}). Prompts will be formatted "
                "differently from what one of the parents was trained on."
            )
        elif (
            base.has_chat_template
            and other.has_chat_template
            and base.chat_template_sha256 != other.chat_template_sha256
        ):
            warnings.append(
                "Chat templates differ between the parents; the merged tokenizer keeps "
                "one of them, so the other parent was trained on different formatting."
            )

    if conflicts:
        status = "conflict"
    elif warnings:
        status = "warning"
    else:
        status = "ok"
    notes.append(
        f"Checked {label} parents: " + ", ".join(Path(f.source).name for f in fingerprints) + "."
    )
    return TokenizerCheck(
        status = status,
        conflicts = tuple(conflicts),
        warnings = tuple(warnings),
        notes = tuple(notes),
        fingerprints = tuple(fingerprints),
    )


def check_before_merge(
    sources: Sequence[str | Path], *, allow_conflict: bool = False
) -> TokenizerCheck:
    """The mandatory gate: raise on a conflict unless explicitly overridden."""
    report = compare_tokenizers(sources, label = "merge")
    if report.status == "conflict" and not allow_conflict:
        raise TokenizerConflictError(
            "Merge refused: the parents' tokenizers disagree on a special token, so the "
            "output would load and run but silently mis-handle prompts. "
            + " ".join(report.conflicts)
            + " Fix the parents (align bos/eos/pad) or re-run with "
            "allow_tokenizer_conflict=true to build it anyway; the override is recorded "
            "in the run manifest.",
            report = report.to_dict(),
        )
    return report


def verify_output(output_dir: str | Path, expected: TokenizerFingerprint) -> dict[str, Any]:
    """Confirm the merged folder carries the tokenizer the report promised."""
    actual = fingerprint_tokenizer(output_dir)
    problems: list[str] = []
    if not actual.files:
        problems.append(
            "No tokenizer files were copied into the merged folder; the result cannot "
            "tokenize text and every downstream consumer will fall back to the base model's."
        )
    elif expected.vocab_size and actual.vocab_size and expected.vocab_size != actual.vocab_size:
        problems.append(
            f"The merged tokenizer reports vocab_size {actual.vocab_size} but the parent "
            f"declared {expected.vocab_size}."
        )
    return {
        "status": "broken" if problems else "ok",
        "problems": problems,
        "tokenizer": actual.to_dict(),
    }


def validate_sources(sources: Iterable[str | Path]) -> dict[str, Any]:
    """Public entry point behind ``POST /merge/validate``."""
    return compare_tokenizers(list(sources)).to_dict()
