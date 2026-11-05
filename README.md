# crate

Browse, audition, filter and organize a sample library from the terminal.

- **See every sample** in a fast table, with BPM, key, length and description, plus a tree of the library's folders.
- **Filter as you type**, with a small query language: `kick dark bpm:120-130 key:~Am genre:house -clap`.
- **Hear it instantly**, with seamless loops and sample-accurate sync between sound and picture.
- **Watch it** in four views: waveform, live spectrum, oscilloscope and spectrogram.
- **See what it sounds like** on two maps: how it compares to others of its kind, and where it sits overall.
- **Find a sound by describing it**: place a dot on a blank plane and hear the closest matches.
- **Tag it** with as many genres as you like, then filter and sort by them.
- **Add new samples** from files, folders or zips, and find out right away if you already have them.
- **Organize safely**: edit metadata and the file renames and refiles itself. Every change can be undone, even after a restart.

It works on any folder of samples that follows the [naming convention](docs/NAMING.md).

## Install

Needs [uv](https://docs.astral.sh/uv/). From this folder:

```sh
uv tool install .      # puts `crate` on your PATH
crate
```

To run it without installing, use `uv run crate`. To update an installed copy after changing the code, use
`uv tool install --reinstall .`, since a plain install can reuse an older cached build.

The library is `~/Music/sorted` by default. To use another one, pass `--library PATH` or set `$CRATE_LIBRARY`.

To try it without a library of your own, `examples/library` holds one snare for each corner of the snare map. The
Finder needs at least 5 samples in a category, so it has nothing to search here.

```sh
uv run crate --library examples/library
```

| Corner | Sound | Sample |
|---|---|---|
| Woody | Dark and tight | `organic-trap-1-wet-dark-tight` |
| Snappy | Bright and tight | `tight-22` |
| Fat | Dark and long | `dark-noisy-11` |
| Splashy | Bright and long | `dub` |

## The TUI

Press `?` inside the app for the full list of keys. The ones you'll use most:

| Key | Action | Key | Action |
|---|---|---|---|
| `/` | filter | `v` | next view |
| `Space` | play / pause | `s` | stop |
| `Enter` | play from the start | `a` | Play on select on or off |
| `t` | open the Finder | `l` | loop on or off |
| `1`-`8` | sort by column | `b` / `h` | match tempo / key |
| `m` / `M` | mark / mark all shown | `f` | favorite |
| `#` | genres | `e` | edit metadata |
| `x` | export to a folder | `d` | move to Trash |
| `u` | undo | `ctrl+g` | stop, even while typing |
| `A` | add samples | `D` | check for duplicates |
| `F` | auto-file what's under Needs fixing | | |
| `B` | open the bank builder | `p` | put on the next empty pad |

Actions apply to the marked samples if there are any, otherwise to the selected one. The bar above the
visualizer has clickable Play, Stop, Play on select, Loop and volume controls.

### Play on select

With **Play on select** on (`a`, or click it), choosing a sample plays it, so you can arrow through a list and
compare by ear. With it off, nothing plays until you press `Space` or `Enter`, and moving to another sample stops
whatever was playing. A filter or sort moving the cursor never counts as choosing. It starts off, and remembers your setting.

### Visualizers

| View | Shows | Stopped, it shows |
|---|---|---|
| Waveform | The whole sample, with the RMS core brighter than the peaks. Marks bars when the BPM is known. | The same, with the playhead parked |
| Spectrum | Live frequency bars, with peak hold | The sample's average spectrum |
| Scope | The live wave, triggered so steady tones hold still | The loudest moment |
| Spectrogram | Frequency over time, as a heat map | The same, with the playhead parked |

Click the waveform or spectrogram to jump to that point. Use a truecolor terminal for the best
spectrogram: iTerm2, kitty, WezTerm and Ghostty are all good. Terminal.app works, with fewer colors.

### Sound maps

`v` also cycles through two maps, each showing the samples that match your filter.

**Character** compares a sample to others in its own category, by rank. Up is Dark to Bright on every map. Across
depends on the kind of sound:

| Kind of sound | Across | Measured by |
|---|---|---|
| Hits and bass one-shots | Tight to Long | how fast it decays |
| Rhythmic loops | Sparse to Busy | hits per second |
| Pads and atmospheres | Static to Moving | how far its center of sound travels |

So a dot at the top means "the brightest of your kicks", not bright in absolute terms. The corners of the most common
categories have words, such as Thuddy, Clicky, Boomy and Ringing for kicks. To change them or add more, edit
`CORNER_WORDS` in `planes.py`.

**Footprint** uses one fixed scale for everything, so you can compare a kick to a hat:

- **Across:** audible length, from hits through notes to phrases and loops.
- **Up:** Low (center of energy below 250 Hz), Mid, or High (above 4 kHz). The selected sample's vertical bar shows how far
  its energy reaches.

Click a dot to select that sample. Double-click to play it.

### The Finder

Press `t`. Pick a category, then place a dot on the blank plane with the mouse, or with the arrow keys and `Enter`.
You get the closest 3, 5 or 10 sounds (press `3`, `5` or `0`), each labeled **Close**, **Fair** or **Far**, so you can tell when your
library simply doesn't have that sound. Arrow through them to compare.

- **Any category** searches the whole library on the Footprint plane, for requests like "low and short".
- **The filter box** narrows the search, as in a dark, long kick with `key:F`. Genres work here too.
- **Near-duplicates** don't use up your result slots. The status line says how many were hidden.
- Categories with fewer than 5 samples aren't listed, since two toms don't make a useful map. They can still turn up under Any category.
- `m` marks a result, and `L` shows it in the library. Marks carry over, so you can tag or export them afterwards.

### Genres

Press `#` to tag the selected or marked samples. A sample can have any number of genres. Tick from the list or type a new one. Then:

- Filter with `genre:house`, `genre:house,techno` (either), `genre:house genre:techno` (both), or `genre:na` (untagged).
- Browse them under **Genres** in the tree. They combine with folders.
- Sort by them with `7`.

Tags are stored in `.crate/genres.json`, not in filenames, so tagging never renames a file. They follow files through
renames, and `u` undoes them.

To get started quickly, `crate genres --suggest` shows the genres implied by each sample's pack and source folder, and
`--apply` tags them all as one undoable step.

### Adding samples

`crate add ~/Downloads/new-pack.zip`, or press `A` in the app. It takes files, folders and zips, and your originals stay
where they are. Each incoming file is checked against the library and the rest of the upload, then:

- **Already there:** left out, and the report names the sample it matches.
- **New:** named by the convention and filed in the right folder. The filename, the file's folder within the pack, and its
  sound all count.
- **Can't tell what it is:** copied to `Incoming/`, and listed under **Needs fixing**. Pressing `e` starts with whatever could be worked out.

Then check the results under **✚ Last upload** in the tree (the filter `is:new`), and press `e` to correct anything. One `u` reverses the
whole upload.

`--dry-run` shows where each file would go without copying anything, which is worth doing with an unfamiliar pack.
`--no-file` puts everything in `Incoming/`. `--include-likely` also adds files that only probably match, and
`--allow-duplicates` adds everything.

Files dragged straight into the library folder wait under **Needs fixing**. `F` or `crate file` files them.

How each value gets decided:

| Value | Comes from |
|---|---|
| Category, loop or one-shot | Words in the filename, then pack folders such as `One_Shot/Drums`, then how the sound behaves |
| BPM | The filename. If it has none, loops whose length fits a whole number of beats get an estimate, which may read half or double time |
| Key | The filename only. The exception is bass one-shots, where the note is read from the audio if it's clear. Otherwise `NA` |
| Description | The filename's meaningful words, plus how it compares with its category |

Every value's source is recorded in `_index.csv`. A filing rule is only as good as a pack's names, so expect more to need you with
packs whose names say little.

### Duplicates

Two levels of match:

- **Same audio:** the decoded sound is identical, even with different metadata or format (WAV and FLAC, or 24-bit and 16-bit).
- **Likely the same:** the waveforms match closely, which catches copies at another volume or sample rate, or as an MP3.
  Files that measure alike are compared directly, and must reach 0.97 similarity.

The check only runs when files are added or when you ask for it, never just because the app started:

- **Adding files:** `crate add` and `A` check everything they bring in. If you drag files into the library folder instead, the
  app checks just those files the next time it finds them, on launch or on a rescan (`R`), and warns you if one matches.
- **On request:** press `D`, or use the commands below. The results stay up to date as you rename or remove files, and last until you quit.

Anything it finds is listed under **⧉ Duplicates** in the tree. A filter for `is:dupe` or `is:copy` tells you if no check has run yet.

```sh
crate dupes                      # the whole library
crate dupes ~/Downloads/pack     # a folder or zip, against itself and the library
crate dupes FOLDER --no-library  # a folder against itself only
crate dupes --exact -f json      # only identical audio, as JSON
```

`crate dupes` exits with status 1 if it finds any, and it never touches your samples. To clear them out, filter with `is:copy`, which
leaves out the suggested keeper in each set (favorites and tagged samples are preferred), then `M` and `d`. The Trash and `u` protect you if you change your mind.

## Filter language

The search box and `crate ls` share one syntax. All terms must match; put `-` in front of any term to exclude it.

| Term | Meaning |
|---|---|
| `kick dark` | words in the category, key, description or original filename |
| `main:drums` `sub:kick` | category, by prefix (`main:drums,percs` for either) |
| `sub:=Hat` | exactly that sub, not `Hat-Open` |
| `type:loop` `type:shot` | loops or one-shots |
| `bpm:120` `bpm:120-130` `bpm:>140` `bpm:<90` | tempo |
| `bpm:~120` | tempo-compatible: within 3%, counting half and double time |
| `bpm:na` `key:na` | no value |
| `key:Am` `key:F#min` `key:Bbmaj` `key:8A` | key, in any common spelling or Camelot |
| `key:F` | any F: `F`, `Fmin` or `Fmaj` |
| `key:~Am` | harmonically compatible keys |
| `genre:house` `genre:house,techno` `genre:na` | genre tags; `dnb`, `lofi` and similar work too |
| `range:low` `range:mid` `range:high` | where the sound's energy sits |
| `len:<2` `len:1-4` | file length in seconds |
| `audible:<1` | seconds of actual sound |
| `desc:vinyl` | description contains |
| `in:Drums/Loops/Kick` | a folder and everything under it |
| `is:fav` `is:marked` `is:invalid` | favorites, marked, or names that break the convention |
| `is:new` | from the most recent upload |
| `is:dupe` `is:copy` | every copy of a duplicate, or all but the one to keep |

`key:Fmin` also matches a bare `F`, because a bare note means nobody stated the mode.

## Command line

```sh
crate                                # open the TUI
crate tui bpm:~174 key:~Am           # open it with a filter applied

crate ls kick dark -clap             # list matches
crate ls type:loop --sort=-bpm -n 20
crate ls key:Am --format json        # or: table, paths, csv
crate ls in:Bass/One-Shots/808 --format paths | head

crate stats                          # totals, BPM histogram, common keys
crate show sub:reese                 # details and a waveform for the first match

crate lint                           # check names and folders; exits 1 on problems
crate lint --deep                    # also compare the length in each name to the audio
crate lint --fix                     # move misfiled files and correct lengths (undoable)
crate lint "some name.wav"           # explain what's wrong with a filename

crate retag FILE --key Am --bpm 128  # renames and refiles
crate retag --where "sub:wub key:na" --sub Growl --dry-run
crate tag FILE --add "House, Techno"  # add or remove genres
crate tag --where "sub:reese" --add "Drum & Bass" --remove Dubstep
crate genres                         # genres in use
crate genres --suggest               # genres implied by pack names; add --apply to tag them
crate add ~/Downloads/new-pack.zip   # add samples and file them, leaving out any you already have
crate add pack.zip --dry-run         # see where each would go first
crate file                           # file samples dragged into the library folder
crate dupes                          # find duplicates in the library, or pass a folder
crate analyze                        # measure everything now, rather than in the background
crate export is:fav --to ~/Desktop/session
crate bank                           # list your banks
crate bank new "Dusty Kit"
crate bank put "Dusty Kit" 1 sub:kick dark    # a file path, or the first match for a filter
crate bank show "Dusty Kit"
crate bank export "Dusty Kit" --to /Volumes/CARD/IMPORT
crate undo
```

`--read-only` refuses any change to the library, and `--no-audio` runs without sound, which is useful over SSH.

## Building banks for an SP-404MK2

A bank is 16 samples laid out like the sampler's pads, with pad 1 at the top left. You can keep as many as you like.

**Filling it.** Press `p` on any sample, or on marked samples, to put it on the next empty pad of the bank you last used.
It also works on a Finder result. To choose for one particular pad, press `B` to open the bank builder, move to the
pad, and press `p` to pick from the library or `t` to use the Finder. Then `p` puts your choice there and brings you back.

**Inside the bank builder:**

| Key | Does | Key | Does |
|---|---|---|---|
| arrows | move between pads | `Enter` / `Space` | play from the start / play or pause |
| `p` / `t` | pick from the library / with the Finder | `Backspace` | clear the pad, keeping the file |
| `g` | pick up a pad; the arrows carry it, `g` puts it down | `shift`+arrows | carry it directly, in terminals that send them |
| `k` | kit mode | `x` | export, numbered by pad |
| `[` `]` | previous / next bank | `n` | new bank |
| `r` | rename the bank | `D` | delete the bank |
| `u` | undo | `Esc` | back |

Play on select and `s` work as they do everywhere else. **Kit mode** plays pads from the keyboard, whatever Play on
select is set to, using keys in the same shape as the pads: `1234`, `qwer`, `asdf`, `zxcv`. `ctrl+g` stops, and `Esc` leaves.
It plays one sound at a time, so a new pad cuts off the last one, and the SP-404MK2 itself will layer them.

**Warnings on a pad.** The heading shows the tempo most loops can play at, counting half and double time, and the key
most pitched samples mix with. A pad that doesn't fit says so, with `≠ 120 bpm` or `≠ Amin`. Drums don't count towards the
key. A pad also warns `under 100 ms`, which Roland says may not import, and `file missing` if its file has gone.

**Exporting.** `x` or `crate bank export` makes a folder named after the bank inside the folder you choose, normally your
card's `IMPORT` folder. Each file starts with its pad number, which is the first thing the sampler's screen shows:

```text
IMPORT/dusty-kit/01__Drums__Kick__OneShot__0.31s__subby-30.wav
IMPORT/dusty-kit/05__Drums__Snare__OneShot__0.47s__3.wav
```

Then on the SP-404MK2, hold `SHIFT` and press pad 13, and choose `IMPORT`, `IMPORT from SD-CARD`, `SAMPLE`. Press a pad,
turn `VALUE` to the file with that pad's number, and press `VALUE`.

Nothing on the card is overwritten or deleted. Exporting again adds any newly filled pads to the same folder. If you've
rearranged the bank, the new layout goes in `dusty-kit-v2` instead, so no pad number ends up with two files.

Banks are saved in `<library>/.crate/banks.json`. Pads follow a file when it's renamed. If a file goes to the Trash, its pad
waits, and undoing the trash brings it back.

## Exporting to an SP-404MK2

`x` or `crate export` copies samples into one folder, ready for the sampler's `IMPORT` folder:

- Names lose their `NA` fields, so `Drums__Kick__OneShot__NA__NA__0.31s__subby-30.wav` becomes
  `Drums__Kick__OneShot__0.31s__subby-30.wav`.
- 32-bit float WAVs become 24-bit, because the SP-404MK2 rejects them as UNSUPPORTED FILE. A file peaking past full
  scale is turned down to fit rather than clipped. 16 and 24-bit files are copied exactly as they are.
- Your library files aren't changed.

## Safety

- Nothing is ever overwritten. A name that would collide gets a `-v2` suffix, and exports skip files that already exist.
- "Delete" moves files to the Trash, and `u` brings them back.
- Every rename, genre change, bank change and trash goes in a journal, so undo works across restarts, as many steps back as you like.
- A batch that fails partway through is rolled back.
- Adding samples only ever copies. Undoing an upload moves the copies to the Trash.
- `_index.csv` stays in sync with renames, and anything you change is marked `manual` in it.

`crate` keeps its favorites, genres, banks, measurements, preferences and undo history in `<library>/.crate/`. The first launch
measures every sample in the background, which took about 17 seconds for 2,745 samples. After that, only new or changed files are measured.

## Development

```sh
uv sync
uv run pytest
```

The tests never touch your library or your Trash, and no sound plays. The one exception is
`tests/test_real_library.py`, which reads `~/Music/sorted` without changing it, and is skipped if that folder doesn't exist.

| Module | Responsibility |
|---|---|
| `naming.py` | The convention: parse, format, validate, keys, Camelot |
| `library.py` | Scanning, sorting, the index, favorites, banks and preferences |
| `query.py` | The filter language |
| `organize.py` | Everything that changes files, with the undo journal |
| `audio.py` | Decoding, peaks, spectrograms, the clip cache |
| `features.py` | Measuring each sample, and caching the results |
| `planes.py` | The two sound maps, and the search behind the Finder |
| `filing.py` | Working out where a new sample belongs |
| `genres.py` | Genre names, aliases and suggestions |
| `duplicates.py` | Finding samples you already have, in folders, zips and the library |
| `banks.py` | Whether a bank's samples belong together, and describing its export |
| `player.py` | Playback, on a real or silent output stream |
| `viz.py` | Visualizer math: pure functions, no UI |
| `tui/` | The Textual app, table, visualizer, maps, Finder, bank builder and dialogs |

## License

The code is under the [MIT License](LICENSE). The four samples in `examples/library` aren't covered by it: they come from
third-party sample packs.
