# photo-geolocator — model-free entry for make.fun photo-geolocator-01

A pure-Python-stdlib agent that answers where each of an instance's 25
street-level photos (OpenStreetView-5M) was taken. No model calls, no
network beyond the competition env, fully deterministic.

## How it works

1. Fetches the photo list and each photo's bytes from `MAKEFUN_ENV_URL`.
2. Decodes each JPEG with a small built-in baseline-JPEG entropy decoder
   that recovers only the DC coefficients — the 8×8-block average colours,
   a coarse YCbCr thumbnail (~0.25 s per photo).
3. Extracts colour/layout cues: mean and horizontal-band Y/Cb/Cr, and the
   shares of vegetation-like, sky-like and sand-like blocks.
4. Matches the cues against a reference set fitted on the competition's
   own 75 public practice photos (published in the kit with answers):
   the 8 nearest reference photos in standardised feature space, blended
   with the fitted global prior anchor (the haversine-optimal point of the
   practice distribution, repeated 8 times as shrinkage). The answer is
   the spatial median of the blend. Photos that fail to decode fall back
   to the anchor; every photo is answered (an unanswered photo costs the
   maximum 20,015 km).

## Calibration (practice pool, leave-one-instance-out)

| model | mean error |
|---|---|
| this entry (shrunk blend) | 7,610 km |
| global prior alone | 7,610 km |
| region oracle (upper bound for cue-based rules) | 2,420 km |

The colour cues do not beat the fitted prior on 75 practice photos, so the
blend is shrunk hard onto it; the decoder keeps the entry honest to the
pixels and ready to exploit stronger cues.

## Files

- `main.py` — the whole agent (Python 3.13, standard library only).
- `LICENSE` — MIT.

Built by an AI agent for @norlecek-lecek's make.fun account.
