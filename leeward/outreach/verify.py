"""Four-word verification phrases (SPEC §8).

A scammer can copy every word the VA sends except these four: they are derived for *this*
veteran on *this* day, so a real VA caller can say them and an impostor cannot. A veteran
whose caller cannot say them hangs up.

Deterministic on purpose. The same veteran-day always yields the same phrase, so every
message and call to that veteran on that day agree, and rehearsal matches the stage. It is
built on SHA-256 rather than `hash()` (salted per process) or a stateful RNG, so it also
matches across machines and Python versions.

`SEED` is a demo value committed to the repo, so anyone who reads this file can compute a
phrase. That is fine for synthetic veterans and not for real ones: a deployment would swap
the plain hash for an HMAC under a secret key the VA holds, and nothing else here would change.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime

#: Demo seed (the hackathon's first day). Change it and every phrase changes.
SEED = 20260919

PHRASE_LENGTH = 4

#: 512 = 2**9, so nine bits of a digest pick a word with no modulo bias. Short, common,
#: lowercase, ASCII words, chosen to be read down a phone: no homophone pairs (sea/see), and
#: nothing that sounds like a symptom, a diagnosis, a drug or money, because these words
#: arrive inside a health message. **The order is part of the contract**: reordering or
#: editing this list changes every phrase already sent, and `tests/test_verify.py` pins it.
WORDS: tuple[str, ...] = tuple("""
    otter heron finch robin badger beaver bison camel crane
    donkey eagle falcon ferret gecko goose horse koala lemur
    llama lynx magpie moose newt panda parrot pigeon pony
    rabbit salmon seal sparrow swan tiger trout turtle walrus
    weasel wren zebra mule goat sheep beetle lark quail
    raven stork yak dove fox frog lamb duck crab
    clam whale perch gull kite acorn amber anchor aspen
    autumn bamboo basin birch blossom boulder breeze brook canyon
    cedar clover cliff cloud coral creek crest delta desert
    dune fern field forest glacier glade grove harbor hazel
    hill island jungle lagoon lake leaf ledge lily maple
    marsh meadow moss mountain oak oasis orchard pebble pine
    pond prairie reef ridge river sage sand shore spruce
    spring stream summit timber trail valley willow basket bell
    blanket bottle bowl bridge brush bucket button candle carpet
    chair clock coat compass copper cup curtain desk dish
    door drum feather fence flag frame garden gate glove
    hammer hat jar kettle key ladder lamp lantern lock
    mirror mug notebook paper pencil pillow pitcher plate pocket
    quilt radio ribbon rope rug scarf shelf spoon stool
    table teapot ticket towel trumpet umbrella vase wagon wallet
    window apple apricot bagel banana barley basil biscuit butter
    cabbage carrot celery cherry chestnut cinnamon cocoa corn cracker
    cucumber ginger grape honey lemon lentil lime mango melon
    mint muffin mustard noodle nutmeg oatmeal olive onion orange
    papaya peach peanut pepper pickle plum potato pumpkin radish
    raisin rice salad soup spinach squash tomato vanilla walnut
    waffle yogurt blue bronze brown crimson gold gray green
    indigo ivory lilac maroon pink purple scarlet silver tan
    teal violet white yellow circle square triangle bright calm
    clever gentle happy honest kind lively lucky merry noble
    polite proud quiet ready silent smooth soft steady sunny
    swift tender wise witty anthem arrow balloon banner beacon
    bicycle cabin canoe castle chapter circus cottage crayon crown
    diary dragon engine fable feast festival gadget galaxy gem
    harp helmet jewel journal kayak library marble medal melody
    museum orbit palace parade piano picnic pilot planet puzzle
    rocket saddle sailor shuttle signal skate sketch story studio
    sunrise tent theater tower train tunnel violin voyage wheel
    whistle bolt brick cable chain chalk cotton cork denim
    glass iron leather linen metal paint plank rubber satin
    silk stone tile velvet wax wool yarn thread string
    banjo cello chess choir chorus cymbal domino fiddle flute
    guitar harmony opera soccer tennis tempo tune bingo hockey
    barn bakery cafe campus corner court factory farm fountain
    garage market mill office plaza porch school shop stadium
    station street terrace village yard moon star comet rainbow
    sunset dawn daylight twilight arch bench backpack barrel boat
    bookcase carriage cart cheese coach cushion diamond doll envelope
    fabric globe handle harvest jacket kitchen lawn letter lumber
    magnet napkin oven package parcel pasture postcard pottery puppet
    raft sandal scooter shovel slate sponge statue sweater teacup
    tulip cactus cypress daisy ivy jasmine juniper lavender lotus
    magnolia orchid petal reed rose rosemary vine wheat alpaca
    antelope bobcat buffalo canary chipmunk cricket dolphin flamingo gazelle
    giraffe hamster hawk hippo kitten lizard octopus panther peacock
    penguin pelican puppy rooster swallow toucan tortoise turkey
""".split())

assert len(WORDS) == 512 and len(set(WORDS)) == 512, "WORDS must be 512 distinct words"


def _iso(day: date | str) -> str:
    if isinstance(day, datetime):
        return day.date().isoformat()
    if isinstance(day, date):
        return day.isoformat()
    try:
        return datetime.fromisoformat(str(day)).date().isoformat()
    except ValueError as exc:
        raise ValueError(f"day must be a date or an ISO date string, got {day!r}") from exc


def phrase(veteran_id: str, day: date | str, *, seed: int = SEED) -> str:
    """The four-word phrase for one veteran on one day, e.g. "maple lantern river copper".

    The words are distinct, so a garbled or repeated word is never mistaken for a match.
    """
    veteran_id = str(veteran_id).strip()
    if not veteran_id:
        raise ValueError("veteran_id is empty; a phrase for nobody verifies nothing")
    stamp = _iso(day)

    words: list[str] = []
    counter = 0
    while len(words) < PHRASE_LENGTH:
        digest = hashlib.sha256(f"{seed}|{veteran_id}|{stamp}|{counter}".encode()).digest()
        counter += 1
        for i in range(0, len(digest) - 1, 2):
            word = WORDS[int.from_bytes(digest[i:i + 2], "big") & 0x1FF]
            if word not in words:
                words.append(word)
                if len(words) == PHRASE_LENGTH:
                    break
    return " ".join(words)
