"""The four-word verification phrase (SPEC §8).

The phrase is the one thing in a message a scammer cannot fake, so it has three jobs: be
the same for the same veteran-day every time (rehearsal and stage must agree), be different
for different veteran-days (or it proves nothing), and be short enough to read down a phone.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

from leeward.outreach import verify

ROOT = Path(__file__).resolve().parents[1]
DAY = date(2026, 7, 16)


def test_word_list_is_512_distinct_common_words() -> None:
    words = verify.WORDS
    assert len(words) == 512, "512 words = 9 bits a word; the bit-mask in phrase() relies on it"
    assert len(set(words)) == 512, "duplicate words shrink the phrase space"
    for w in words:
        assert w.isascii() and w.isalpha() and w == w.lower(), f"{w!r}: lowercase letters only"
        assert 3 <= len(w) <= 8, f"{w!r}: keep words short enough to read down a phone"


def test_phrase_is_four_distinct_words_from_the_list() -> None:
    words = verify.phrase("SYN-000000", DAY).split(" ")
    assert len(words) == 4
    assert len(set(words)) == 4, "a repeated word makes the phrase easier to guess and to garble"
    assert set(words) <= set(verify.WORDS)


def test_same_veteran_day_always_gives_the_same_phrase() -> None:
    assert verify.phrase("SYN-000000", DAY) == verify.phrase("SYN-000000", DAY)


def test_day_may_be_a_date_a_datetime_or_an_iso_string() -> None:
    want = verify.phrase("SYN-000000", DAY)
    assert verify.phrase("SYN-000000", datetime(2026, 7, 16, 23, 59)) == want
    assert verify.phrase("SYN-000000", "2026-07-16") == want
    assert verify.phrase("SYN-000000", "2026-07-16T08:30:00") == want


def test_phrase_pins_the_current_word_list_and_seed() -> None:
    """A regression pin, not a correctness proof: it fails if anyone reorders WORDS, changes
    SEED or changes the derivation, any of which silently changes every phrase already sent."""
    assert verify.phrase("SYN-000000", DAY) == "cotton peacock comet raven"
    assert verify.phrase("SYN-000001", DAY) == "helmet desk lizard canyon"


def test_different_veterans_and_days_give_different_phrases() -> None:
    seen = {verify.phrase(f"SYN-{i:06d}", date(2026, 7, 1 + d % 28))
            for i in range(500) for d in range(4)}
    assert len(seen) == 2000, "collisions across 2,000 veteran-days; the phrase is not verifying"


def test_every_word_is_reachable() -> None:
    """Catches a bit-mask bug that quietly uses only part of the list."""
    used: set[str] = set()
    for i in range(3000):
        used.update(verify.phrase(f"SYN-{i:06d}", DAY).split(" "))
    assert used == set(verify.WORDS)


def test_seed_changes_the_phrase() -> None:
    assert verify.phrase("SYN-000000", DAY, seed=1) != verify.phrase("SYN-000000", DAY, seed=2)


def test_phrase_does_not_depend_on_pythons_per_process_hash_salt() -> None:
    """`hash()` is salted per process, so a phrase built on it would differ between the
    rehearsal and the stage run. hashlib is not. Prove it across real processes."""
    code = ("from datetime import date; from leeward.outreach import verify; "
            "print(verify.phrase('SYN-000000', date(2026, 7, 16)))")
    got = set()
    for salt in ("1", "2", "random"):
        env = dict(os.environ, PYTHONHASHSEED=salt)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=env, cwd=ROOT, check=True)
        got.add(out.stdout.strip())
    assert got == {verify.phrase("SYN-000000", DAY)}


def test_bad_input_fails_loudly() -> None:
    with pytest.raises(ValueError):
        verify.phrase("", DAY)
    with pytest.raises(ValueError):
        verify.phrase("SYN-000000", "next thursday")
