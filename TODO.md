# To do

## Quick trim view

**Status: bookmarked on 2026-11-05. Not started.**

Remove the dead space from a sample in its own view, working on a copy.

### Requirements

- It has its own view.
- It copies the audio file first and trims the copy, so the original never changes.
- Undo works both while trimming and afterwards.
- It estimates where the dead space starts and ends, and uses that to set the initial waypoints.
- You can fine-tune the waypoints.
- You can play the sample while trimming, to test the result.

### Where to start in the code

- **Detection:** `features._audible_slice` already finds the first and last sound above `AUDIBLE_FLOOR_DB`, which is 50 dB below the peak.
- **Undo after saving:** `organize.add_files` journals an `add` step, and undoing that moves the copy to the Trash.
- **The view:** `tui/banks.py` shows how a full-screen view shares the player, transport bar and `Visualizer`. The waveform will need waypoint markers.
- **Safe writes:** `organize._write_as_pcm24` writes through a `.part` file, so a failure leaves no half-written file.

### Things to handle

- **Loops:** a loop's length usually fits its bars, so silence at the end may be part of the loop, and trimming it would break the timing. Warn before trimming one, and show how many bars the result is.
- **How common it is:** 544 of the 2,745 samples have over 0.5 s below that 50 dB floor, counting the start and end together: 291 one-shots and 253 loops. 106 have over 2 s.
- **Quiet tails:** a reverb tail can sit below the floor and still be wanted, so the estimate is only a starting point.
- **Clicks:** cutting where the sound hasn't fully decayed can click, so consider a very short fade at the end point.
- **Format:** keep the original's sample rate and bit depth. Exports already convert float files to 24-bit.
- **Name:** the copy's Length field has to change, and it needs its own description, such as a `-trim` suffix. Collisions get the usual `-vN`.
- **Duplicate check:** the check ignores silence, so it will likely report the copy as "Likely the same" as its original. Decide whether that's right for a deliberate trim.
- **Tags:** decide whether genres and favorites carry over to the copy.

### Open questions

1. When saved, should the copy sit beside the original in the library, or should it replace the original, with the original going to the Trash?
2. If the sample is on a bank pad, should the pad switch to the trimmed copy?
3. Should the copy join the library as soon as the view opens, or only on save? Waiting until save means cancelling leaves nothing behind.
