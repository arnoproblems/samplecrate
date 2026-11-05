# Sample file naming convention

Every sample in the library carries its metadata in its filename, so the files stay
self-describing without a database, and any tool can read them. `crate` treats the filename
as the source of truth.

```
Synths__Pad__Loop__080bpm__Dmin__24.00s__haze-lowest.wav
```

| Field | Value in this example |
|---|---|
| Main category | `Synths` |
| Sub category | `Pad` |
| Type | `Loop` |
| Tempo | `080bpm` |
| Key | `Dmin` |
| Length | `24.00s` |
| Description | `haze-lowest` |
| Extension | `wav` |

## Fields

Fields are separated by a double underscore (`__`) and always appear in this order. A field that
doesn't apply or isn't known holds `NA`, so every name has the same seven fields.

| # | Field | Format | Notes |
|---|---|---|---|
| 1 | Main | Letters only: `Drums`, `FX` | Top-level folder. See [Categories](#categories). |
| 2 | Sub | Letters and digits, with single hyphens between words: `Hat-Open`, `808` | Third-level folder. |
| 3 | Type | `Loop` or `OneShot` | See [Loop or one-shot](#loop-or-one-shot). |
| 4 | BPM | Three digits plus `bpm` (`080bpm`, `174bpm`), or `NA` | Zero-padded so names sort numerically. `000bpm` isn't valid. |
| 5 | Key | Note plus `maj` or `min` (`F#min`), a bare note (`F#`), or `NA` | See [Keys](#keys). |
| 6 | Length | Seconds with exactly two decimals, plus `s`: `3.87s`, `214.66s` | The real length of the audio, including any silence. |
| 7 | Description | Lowercase letters and digits, with single hyphens between words | Free text. Must not be empty. |

The extension is lowercase and one of `wav`, `aif`, `aiff`, `flac`, `ogg` or `mp3`.

Since `__` only ever appears between fields, you can always split a name with a plain
`split("__")`. No field can contain it.

### Keys

- **Sharps only.** There's one spelling per pitch: `A#min`, never `Bbmin`.
- `maj` or `min` when the mode is known: `Cmaj`, `F#min`.
- **A bare note** (`F`, `C#`) means the root is known but the mode isn't. This is normal for
  one-shots such as 808s, where a pack labels the note. A bare note isn't the same as major.
- `NA` when the key is unknown or doesn't apply, as with most drums.

When you enter a key in `crate`, it accepts any common spelling (`Am`, `A minor`, `Bbmaj`, Camelot
`8A`) and writes the canonical form to the filename.

### Collisions

Two different samples can't share a name. If they would, the later one gets `-v2`, `-v3` and so on
at the end of its description:

```
Drums__Hat__OneShot__NA__NA__0.48s__hh.wav
Drums__Hat__OneShot__NA__NA__0.48s__hh-v2.wav
```

## Folder layout

The folder is derived from the name, so a file's location never disagrees with its metadata:

```
<Main>/<Loops or One-Shots>/<Sub>/<filename>
```

`Loop` files live under `Loops`, and `OneShot` files under `One-Shots`:

```
Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__F__0.57s__punchy.wav
Synths/Loops/Pad/Synths__Pad__Loop__080bpm__Dmin__24.00s__haze-lowest.wav
```

## Categories

`Main` and `Sub` accept any value that fits the format. New ones appear in `crate` automatically.
These are the ones in use, and how they're applied:

| Main | Sub categories | What goes here |
|---|---|---|
| `Drums` | `Kick` `Snare` `Clap` `Snap` `Rim` `Tom` `Hat` `Hat-Open` `Hat-Closed` `Cymbal` `Crash` `Ride` `Top` | Individual drum sounds, one-shot or looped. |
| `Drums` | `Full` `Top` `Stripped` `Break` `Fill` | Multi-element drum loops. `Top` means no kick. |
| `Percs` | `Perc` `Bongo` `Conga` `Shaker` `Tambourine` `Cowbell` `Tabla` `Log-Drum` `Chimes` | Hand and auxiliary percussion. |
| `Bass` | `808` `Sub` `Reese` `Wub` `Synth-Bass` `Bass` | Anything whose main job is the low end. |
| `Synths` | `Pad` `Lead` `Pluck` `Arp` `Stab` `Chords` `Synth` | Synthesized sounds. |
| `Instruments` | `Piano` `Keys` `Organ` `Guitar` `Sax` `Trumpet` `Brass` `Woodwind` `Strings` `Harp` `Mallets` `Accordion` | Acoustic and electric instruments, and sounds modeled on them. |
| `Melodic` | `Music` `Melody` `Stack` `Chords` `Beat` | Mixed or unspecified melodic material, when no single instrument fits. |
| `Vocals` | `Phrase` `Chop` `Chant` `Choir` `Adlib` `Harmony` `Melody` `Rap` `Acapella` `Vocal-FX` | Anything sung or spoken. |
| `FX` | `Riser` `Downlifter` `Impact` `Sub-Drop` `Sweep` `Atmosphere` `Vinyl` `Glitch` `Game` `Weapon` `Crowd` `Misc` | Effects and textures. |

Drum-based samples always go in `Drums`, while hand percussion such as bongos goes in `Percs`.

### Loop or one-shot

- **`Loop`**: a pattern or phrase meant to repeat or run in time. Multi-hit drum fills count as
  loops. Loops should have a BPM when the tempo is known.
- **`OneShot`**: a single event, played once.

Non-repeating events are always `OneShot`, even when tempo-synced or filed under "loops" in the
pack they came from: risers, downlifters, sweeps, impacts and sub drops. These may still
carry a BPM. A full-length acapella is also a `OneShot`.

## Examples

These lists are checked by the test suite against the parser, so they can't go out of date.

Valid names:

<!-- valid-names -->
```
Synths__Pad__Loop__080bpm__Dmin__24.00s__haze-lowest.wav
Drums__Kick__OneShot__NA__NA__0.25s__dark-subby-13.wav
Drums__Hat-Open__OneShot__NA__NA__0.48s__hh-v2.wav
Bass__808__OneShot__NA__F__1.16s__clean-dark-long.wav
FX__Riser__OneShot__170bpm__NA__11.29s__uplifter-6.wav
Instruments__Sax__Loop__120bpm__A#min__8.00s__echo.flac
Vocals__Acapella__OneShot__NA__NA__214.66s__scatman.mp3
```

Invalid names:

<!-- invalid-names -->
```
my kick.wav
Drums_Kick_OneShot_NA_NA_0.25s_dark.wav
Drums__Kick__OneShot__NA__NA__0.25s.wav
Drums__Kick__Shot__NA__NA__0.25s__dark.wav
Drums__Kick__OneShot__90bpm__NA__0.25s__dark.wav
Drums__Kick__OneShot__000bpm__NA__0.25s__dark.wav
Drums__Kick__OneShot__NA__Bbmin__0.25s__dark.wav
Drums__Kick__OneShot__NA__E#__0.25s__dark.wav
Drums__Kick__OneShot__NA__NA__0.3s__dark.wav
Drums__Kick__OneShot__NA__NA__0.25s__Dark.wav
Drums__Kick__OneShot__NA__NA__0.25s__dark--kick.wav
Drums__Kick__OneShot__NA__NA__0.25s__dark.WAV
Drums__Kick__OneShot__NA__NA__0.25s__dark.txt
```

## Parsing

This regular expression matches exactly the valid names:

<!-- name-regex -->
```regex
^(?P<main>[A-Za-z]+)__(?P<sub>[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)__(?P<type>Loop|OneShot)__(?P<bpm>(?!000)\d{3}bpm|NA)__(?P<key>(?:C#?|D#?|E|F#?|G#?|A#?|B)(?:maj|min)?|NA)__(?P<length>\d+\.\d{2})s__(?P<desc>[a-z0-9]+(?:-[a-z0-9]+)*)\.(?P<ext>wav|aif|aiff|flac|ogg|mp3)$
```

Or with `crate`:

```sh
crate ls key:Am bpm:120-130 --format json     # query the library as JSON
crate lint "some name.wav"                    # explain what's wrong with a name
```

## Adding new samples

1. Run `crate add` on a file, folder or zip, or press `A` in the app. Anything you already have is left out. The rest
   is named by this convention and filed automatically, using its filename, its folder within the pack, and its sound.
2. Check the result under **✚ Last upload** in the browse tree, and press `e` to correct anything.
3. Whatever `crate` couldn't work out waits in `Incoming`, listed under **! Needs fixing**. Press `e` to file those, with the
   fields it did work out already filled in.

You can also drop a file anywhere inside the library, under any name. It shows up under **! Needs fixing**, and `F` or
`crate file` will file it.

Or name and file it yourself following this document, and run `crate lint --deep` to check it.
`crate lint --fix` moves misfiled files and corrects wrong lengths in names. It can't guess a
category, so a name that doesn't parse has to be fixed by hand.

## What isn't in the name

Some information deliberately stays out of filenames, so the convention stays stable and editing it never renames a file:

- **Genres.** A sample can have any number of them, which doesn't fit seven fixed fields.
- **Measurements.** Audible length, Low/Mid/High range and the like are measured from the audio, and can always be measured again.

The Length field is always the length of the file, including any silence. For the length of the actual sound, use
`audible:` in a filter.

## Exported copies

Exports leave out a BPM or Key field that is `NA`, since it means nothing outside the library. A library file named
`Drums__Kick__OneShot__NA__NA__0.31s__subby-30.wav` is exported as `Drums__Kick__OneShot__0.31s__subby-30.wav`.
Those names have fewer fields, so they don't follow this convention, and only the copies are named this way.

A bank's export also starts each name with its pad number, as in `01__Drums__Kick__OneShot__0.31s__subby-30.wav`.

## Other files in the library

Both of these are optional. The library works without them, and `crate` ignores any file that isn't audio.

- **`_index.csv`** records where each sample came from: its original filename, its source paths,
  and whether each BPM and key was stated by the pack (`name`), estimated from the audio (`est-*`), or
  entered by hand (`manual`). `crate` keeps it up to date when it renames files, and shows it in the
  details panel. The filename still wins if the two ever disagree.
- **`.crate/`** holds `crate`'s favorites, genre tags, measurements, preferences and undo history. Favorites and
  genres exist nowhere else, so keep this folder if you move the library or back it up.
