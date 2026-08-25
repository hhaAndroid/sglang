#!/usr/bin/env python3
"""Generate deterministic text workloads for agentic prefix-cache benchmarks.

The output contains:
  - one global prefix shared by every session;
  - one unique prefix per session;
  - one observation per turn, shared as a sequence by all sessions.

The benchmark runner should append the model's real output to each session's
history before sending the next observation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an engine-neutral agentic prefix-cache workload."
    )
    parser.add_argument("--tokenizer", required=True, help="Model/tokenizer path")
    parser.add_argument("--corpus", required=True, help="UTF-8 source text")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--num-sessions", type=int, default=75)
    parser.add_argument("--num-turns", type=int, default=30)
    parser.add_argument("--global-prefix-tokens", type=int, default=20_000)
    parser.add_argument("--session-prefix-tokens", type=int, default=10_000)
    parser.add_argument("--observation-tokens", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "num_sessions",
        "num_turns",
        "global_prefix_tokens",
        "session_prefix_tokens",
        "observation_tokens",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def make_piece(
    tokenizer: Any,
    token_ids: list[int],
    start: int,
    target_tokens: int,
    label: str,
) -> tuple[dict[str, Any], int]:
    end = start + target_tokens
    if end > len(token_ids):
        raise ValueError(
            f"Corpus is too short while creating {label}: "
            f"need token offset {end}, corpus has {len(token_ids)} tokens"
        )

    text = tokenizer.decode(
        token_ids[start:end],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    actual_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return (
        {
            "text": text,
            "target_tokens": target_tokens,
            "actual_tokens": actual_tokens,
        },
        end,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    corpus_text = Path(args.corpus).read_text(encoding="utf-8")
    corpus_token_ids = tokenizer.encode(corpus_text, add_special_tokens=False)

    required_tokens = (
        args.global_prefix_tokens
        + args.num_sessions * args.session_prefix_tokens
        + args.num_turns * args.observation_tokens
    )
    if len(corpus_token_ids) < required_tokens:
        raise ValueError(
            "Corpus is too short: "
            f"need at least {required_tokens} tokens, got {len(corpus_token_ids)}. "
            "Use a larger corpus or reduce workload sizes."
        )

    rng = random.Random(args.seed)
    max_start = len(corpus_token_ids) - required_tokens
    cursor = rng.randint(0, max_start) if max_start > 0 else 0

    global_prefix, cursor = make_piece(
        tokenizer,
        corpus_token_ids,
        cursor,
        args.global_prefix_tokens,
        "global_prefix",
    )

    sessions = []
    for session_id in range(args.num_sessions):
        session_prefix, cursor = make_piece(
            tokenizer,
            corpus_token_ids,
            cursor,
            args.session_prefix_tokens,
            f"session_{session_id}_prefix",
        )
        sessions.append(
            {
                "session_id": session_id,
                "session_prefix": session_prefix,
            }
        )

    observations = []
    for turn_id in range(args.num_turns):
        observation, cursor = make_piece(
            tokenizer,
            corpus_token_ids,
            cursor,
            args.observation_tokens,
            f"observation_{turn_id}",
        )
        observations.append(
            {
                "turn_id": turn_id,
                **observation,
            }
        )

    output = {
        "schema_version": 1,
        "metadata": {
            "tokenizer": args.tokenizer,
            "seed": args.seed,
            "num_sessions": args.num_sessions,
            "num_turns": args.num_turns,
            "required_corpus_tokens": required_tokens,
            "corpus_tokens": len(corpus_token_ids),
            "observation_scope": "shared_sequence",
        },
        "global_prefix": global_prefix,
        "sessions": sessions,
        "observations": observations,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote workload: {output_path}")
    print(f"Sessions: {args.num_sessions}")
    print(f"Turns per session: {args.num_turns}")
    print(f"Required corpus tokens: {required_tokens}")
    print(f"Available corpus tokens: {len(corpus_token_ids)}")


if __name__ == "__main__":
    main()
