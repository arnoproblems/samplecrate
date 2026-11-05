"""Working out where a new sample belongs, from its filename, its folder and how it sounds.

Filenames from sample packs are messy, so most of this is rules learned from sorting a real library: which
words name a sound and which merely describe one, and when a bare number or letter really is a tempo or key.
Each rule exists because something got misfiled without it. Anything that can't be worked out confidently
is left for a person, rather than guessed.
"""

from __future__ import annotations

import collections
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from samplecrate import audio, features as fx, naming
from samplecrate.features import Features
from samplecrate.naming import SampleName

AUDIO_EXT = re.compile(r'\.(wav|mp3|aif|aiff|flac|ogg|m4a)$', re.I)


# --------------------------------------------------------------------------------------- name normalisation

def strip_ext(b):
    return AUDIO_EXT.sub('', b)

def norm(name):
    n = strip_ext(name)
    n = re.sub(r'@[A-Za-z0-9]+', ' ', n)
    n = n.replace("'", '')
    n = n.replace('DnB', 'dnb').replace('LoFi', 'lofi')
    n = re.sub(r'([a-z])([A-Z])', r'\1 \2', n)
    n = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', n)
    n = re.sub(r'[^A-Za-z0-9#]+', ' ', n).lower().strip()
    n = re.sub(r'\bhihats?\b', 'hi hat', n)
    n = re.sub(r'\boneshots?\b', 'one shot', n)
    n = re.sub(r'\bdrumloops?\b', 'drum loop', n)
    n = re.sub(r'\bpercussion[st]+\b', 'percussion', n)
    n = re.sub(r'\bsubbass\b', 'sub bass', n)
    n = re.sub(r'\bopenhats?\b', 'open hat', n)
    n = re.sub(r'\bclosedhats?\b', 'closed hat', n)
    return n


# --------------------------------------------------------------------------------------- key parsing

ENH = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#', 'Cb': 'B', 'Fb': 'E', 'E#': 'F', 'B#': 'C'}
PC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def note(n):
    n = n[0].upper() + n[1:]
    return ENH.get(n, n)

def mode(m):
    if not m:
        return ''
    if m == 'M' or m.lower().startswith('maj'):
        return 'maj'
    return 'min'

BD = r'(?<![A-Za-z0-9#])'
BE = r'(?![A-Za-z0-9#])'
KEY_PREFIXED = re.compile(r'(?<![A-Za-z])[Kk][Ee][Yy][ _-]?([A-Ga-g](?:#|b)?)[ _-]?(major|minor|maj|min|Maj|Min|Major|Minor|m|M)?' + BE)
KEY_WORD = re.compile(BD + r'([A-G](?:#|b)?)[ _-]?(major|minor|maj|min|Major|Minor|Maj|Min|MAJ|MIN)' + BE)
KEY_M = re.compile(BD + r'([A-G](?:#|b)?)m' + BE)
KEY_BARE = re.compile(BD + r'([A-G](?:#|b)?)' + BE)

def parse_key(name):
    """Returns (key, span, strength) where strength is 'full' (mode stated) or 'bare' (note only)."""
    s = strip_ext(name)
    for rx in (KEY_PREFIXED, KEY_WORD, KEY_M):
        for m in rx.finditer(s):
            md = m.group(2) if rx is not KEY_M else 'm'
            # "I Am" is English, not A minor
            if rx is KEY_M and m.group(1) == 'A' and re.search(r'\b[Ii][ _-]$', s[:m.start()]):
                continue
            return note(m.group(1)) + mode(md), m.span(), ('full' if md else 'bare')
    last = None
    for m in KEY_BARE.finditer(s):
        last = m     # the note token nearly always trails the name
    if last:
        return note(last.group(1)), last.span(), 'bare'
    return None, None, None

def key_is_plausible(strength, span, name, main, sub):
    if strength == 'full':
        return True
    tok = strip_ext(name)[span[0]:span[1]]
    tail = strip_ext(name)[span[1]:]
    if main == 'Vocals':
        trailing = re.fullmatch(r'[ _-]*((\d+[ _-]*(bpm|bars?)?|wet|dry)[ _-]*)*', tail, re.I)
        if not trailing or re.search(r'\([^)]*$', strip_ext(name)[:span[0]]):
            return False
    if main in ('Drums', 'Percs'):
        # "Beat_09_A", "Kit B" are variation labels; only tuned kicks/toms carry real notes
        return sub in ('Kick', 'Tom') and tok not in ('A', 'B')
    if tok in ('A', 'B') and re.match(r'[ _-]*\d', tail):
        return False
    return True

# --------------------------------------------------------------------------------------- bpm parsing

BPM_AFTER = re.compile(r'(?<![\d.])(\d{2,3}(?:\.\d+)?)[ _-]*(?:bpm|pm)(?![A-Za-z])', re.I)
BPM_BEFORE = re.compile(r'(?<![A-Za-z])bpm[ _-]*(\d{2,3})(?!\d)', re.I)
BARE = re.compile(r'(?<![\dA-Za-z#.])([1-9]\d{1,2})(?![\dA-Za-z.])')
BARS = re.compile(r'(?<![\d])(\d{1,3})[ _-]?bars?(?![A-Za-z])', re.I)

def parse_bpm_explicit(name):
    s = strip_ext(name)
    for rx in (BPM_AFTER, BPM_BEFORE):
        m = rx.search(s)
        if m:
            v = float(m.group(1))
            if 40 <= v <= 300:
                return v, m.span()
            return None, m.span()     # e.g. "999bpm": remove from description but do not trust
    return None, None

def beat_fit(bpm, dur):
    """How close the sample length is to a whole number of beats at this tempo (0 = perfect)."""
    b = dur * bpm / 60.0
    return abs(b - round(b)) / max(round(b), 1), int(round(b))

def octave_close(a, b, tol=0.03):
    return any(abs(a * f / b - 1) < tol for f in (1, 2, 0.5))

GRID = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)


# --------------------------------------------------------------------------------------- category rules

def has(rx, n):
    return re.search(rx, n) is not None

R_VOCAL = (r'\bvocals?\b|\bvox\b|\bacapella\b|\bchoir\b|\bchants?\b|\badlibs?\b|\bv chop\b|\bharmony stack\b|\bbgvs\b'
           r'|\brap\b|\bwhoa\b|\bscatman\b')
ELEMENTS = [('Kick', r'\bkicks?\b|\bbd\b(?! \d)'), ('Snare', r'\bsnares?\b'), ('Clap', r'\bclaps?\b'),
            ('Hat', r'\bhi hats?\b|\bhats?\b|\bhh\b'), ('Cymbal', r'\bcymbals?\b|\bcrash(es)?\b'), ('Tom', r'\btoms?\b'), ('Rim', r'\brim ?shots?\b')]
PERC_ELEMENTS = [('Bongo', r'\bbongos?\b'), ('Conga', r'\bcongas?\b'), ('Shaker', r'\bshakers?\b|\bmaracas?\b'),
                 ('Tambourine', r'\btambourines?\b|\btambs?\b'), ('Cowbell', r'\bcowbells?\b'),
                 ('Clave', r'\bclaves?\b'), ('Woodblock', r'\bwood ?blocks?\b'), ('Triangle', r'\btriangles?\b'),
                 ('Log-Drum', r'\blog drums?\b'), ('Tabla', r'\btablas?\b|\bdjembes?\b'), ('Chimes', r'\bchimes?\b')]
INSTRUMENTS = [('Brass', r'\bbrass\b|\bbrass woodwind\b'), ('Sax', r'\bsax(ophones?)?\b'), ('Trumpet', r'\btrumpets?\b'),
               ('Brass', r'\btrombones?\b'), ('Brass', r'\bhorns?\b'),
               ('Woodwind', r'\bflutes?\b|\bwood ?winds?\b|\bclarinet\b|\boboe\b'),
               ('Strings', r'\bstrings?\b|\bviolins?\b|\bviola\b|\bcellos?\b|\bpizz\b|\borchestr'),
               ('Harp', r'\bharps?\b'),
               ('Mallets', r'\bkalimba\b|\bglockenspiel\b|\bvibraphone\b|\bmarimba\b|\bxylophone\b|\bbells?\b|\bcelesta\b'),
               ('Guitar', r'\bguitars?\b|\bgtr\b'),
               ('Piano', r'(?<!electric )\bpianos?\b'), ('Organ', r'\borgans?\b'), ('Accordion', r'\baccordi[ao]n\b'),
               ('Keys', r'\brhodes\b|\bwurl|\bwurty\b|\be ?pianos?\b|\belectric pianos?\b|\bkeys\b|\bkeyboards?\b')]
SYNTHS = [('Pad', r'\bpads?\b'), ('Arp', r'\barps?\b|\barpeggios?\b'), ('Pluck', r'\bpluck(s|ed)?\b'),
          ('Stab', r'\bstabs?\b'), ('Lead', r'\bleads?\b|\bsupersaw\b|\bhoover\b')]

def drum_type_elements(n):
    els = [e for e, rx in ELEMENTS if has(rx, n)]
    if 'Hat' in els and 'Cymbal' in els:      # "Cymbal_One_Shot_Hihat": hat is the specific element
        els.remove('Cymbal')
    return els

ESM_HEAD = re.compile(r'^esm (?:[a-z]{3} )?(?:\d+ )?((?:bass|drums?|synth|keys|percussion|melodic|vocals?)\b.*?)\b(?:loop|one shot)\b')

def classify_name(n):
    """Category from the filename alone. Returns (main, sub, confidence)."""
    m = ESM_HEAD.match(n)
    if m:
        r = classify_tokens(m.group(1).strip())
        if r[0]:
            return r
    return classify_tokens(n)

def classify_tokens(n):
    if has(R_VOCAL, n) or re.match(r'wt svc\b', n):
        for s_, rx in (('Acapella', r'\bacapella\b|\bscatman\b'), ('Choir', r'\bchoir\b'), ('Chant', r'\bchants?\b'),
                      ('Adlib', r'\badlibs?\b'), ('Chop', r'\bchops?\b|\bchopped\b'),
                      ('Harmony', r'\bharmony\b|\bbgvs\b'), ('Rap', r'\brap\b'),
                      ('Vocal-FX', r'\bfx\b|\bfoghorn\b|\bglitch\b'), ('Melody', r'\bmelod')):
            if has(rx, n):
                return 'Vocals', s_, 'high'
        return 'Vocals', 'Phrase', 'high'

    # FX named outright
    tonal_word = has(r'\bdrums?\b|\bbass\b|\bsynth\b|\bstabs?\b', n)
    if has(r'\breverse crash\b', n):
        return 'FX', 'Riser', 'high'
    if has(r'\bcrash\b', n) and has(r'\bfx\b', n):
        return 'Drums', 'Cymbal', 'high'
    for s_, rx in (('Riser', r'\brisers?\b|\buplifters?\b|\bupshift(er)?\b'),
                   ('Downlifter', r'\bdownlift(er|ers)?\b|\bdownshift(er)?\b|\bdowner\b'),
                   ('Sub-Drop', r'\bsub drop\b'),
                   ('Impact', r'\bimpacts?\b|\bsub boom\b'),
                   ('Glitch', r'\bsfx\b|\bcomputer glitch\b'),
                   ('Weapon', r'\bgun (automatic|retort)\b|\bmachine gun\b'),
                   ('Game', r'\bvgp\b')):
        if has(rx, n):
            return 'FX', s_, 'high'
    if has(r'\bsweeps?\b|\bwhoosh\b|\bswoosh\b', n) and not tonal_word:
        return 'FX', 'Sweep', 'high'
    if has(r'\batmos(phere)?s?\b|\bambien(ce|t)\b|\bdrones?\b', n) and not has(r'\bpiano\b', n):
        return 'FX', 'Atmosphere', 'high'
    if has(r'\bfx\b', n):
        if has(r'\bvinyl\b', n):
            return 'FX', 'Vinyl', 'high'
        return 'FX', ('Crowd' if has(r'\bcrowd\b', n) else 'Misc'), 'high'

    # Named instruments that can't be confused with anything else
    for s, rx in INSTRUMENTS[:9]:
        if has(rx, n):
            return 'Instruments', s, 'high'

    els = drum_type_elements(n)
    drumword = has(r'\bdrums?\b|\bbeats?\b|\bdrum loop\b|\bbreak(s|beat)?\b(?! chords?)|\bdr full\b', n)
    if drumword:
        # Named-key "beat loops" are full productions, not drums
        if has(r'\bbeat loop\b', n) and has(r'\bkey', n):
            return 'Melodic', 'Beat', 'med'
        if has(r'\bno kick\b|\bkickless\b|\btops?\b', n):
            return 'Drums', 'Top', 'high'
        if has(r'\bfills?\b', n):
            return 'Drums', 'Fill', 'high'
        if has(r'\bstripped\b', n):
            return 'Drums', 'Stripped', 'high'
        if has(r'\bfull\b', n):
            return 'Drums', 'Full', 'high'
        if len(els) == 1:
            return 'Drums', els[0], 'high'
        if len(els) > 1:
            return 'Drums', 'Full', 'high'
        if has(r'\bbreak(s|beat)?\b', n):
            return 'Drums', 'Break', 'high'
        for s_, rx in PERC_ELEMENTS:
            if has(rx, n):
                return 'Percs', s_, 'high'
        if has(r'\bperc(ussion|usion)?s?\b', n):
            return 'Percs', 'Perc', 'high'
        if has(r'\b808s?\b', n):
            return 'Bass', '808', 'high'
        if has(r'\brims?\b', n):
            return 'Drums', 'Rim', 'high'
        if has(r'\brides?\b', n):
            return 'Drums', 'Cymbal', 'high'
        return 'Drums', 'Full', 'med'

    for s, rx in PERC_ELEMENTS:
        if has(rx, n):
            return 'Percs', s, 'high'
    if len(els) == 1:
        return 'Drums', els[0], 'high'
    if len(els) > 1:
        return 'Drums', 'Full', 'high'
    if has(r'\bfills?\b', n):
        return 'Drums', 'Fill', 'high'
    if has(r'\btops?\b', n) and has(r'\bloops?\b|\b\d+ ?bars?\b', n):
        return 'Drums', 'Top', 'high'
    if has(r'\bperc(ussion|usion)?s?\b', n):
        return 'Percs', 'Perc', 'high'
    if has(r'\bsnaps?\b', n):
        return 'Drums', 'Snap', 'high'

    if has(r'\b808s?\b', n):
        return 'Bass', '808', 'high'
    if has(r'\breese\b|\breece\b', n):
        return 'Bass', 'Reese', 'high'
    if has(r'\bwubs?\b|\bwobbles?\b|\bgrowls?\b|\briddim\b', n):
        return 'Bass', 'Wub', 'high'
    if has(r'\bsynth bass\b|\bbass synth\b|\bmoog\b', n):
        return 'Bass', 'Synth-Bass', 'high'
    if has(r'\bsub\b|\bsubs\b', n):
        return 'Bass', 'Sub', 'high'
    if has(r'\bbass(line|lines|es)?\b', n):
        return 'Bass', 'Bass', 'high'

    for s, rx in INSTRUMENTS[9:]:
        if has(rx, n):
            return 'Instruments', s, 'high'
    for s, rx in SYNTHS:
        if has(rx, n):
            return 'Synths', s, 'high'
    if has(r'\bchords?\b', n):
        return ('Synths', 'Chords', 'high') if has(r'\bsynth\b', n) else ('Melodic', 'Chords', 'high')
    if has(r'\bsynths?\b|\bsaw\b', n):
        return 'Synths', 'Synth', 'high'

    if has(r'\bmelodic stack\b|\bstack\b', n):
        return 'Melodic', 'Stack', 'high'
    if has(r'\bmelody\b|\bmelodic\b', n):
        return 'Melodic', 'Melody', 'high'
    if has(r'\bsong ?starters?\b|\bmusic\b|\blayered\b|\bmulti instrument\b|\bstems\b', n):
        return 'Melodic', 'Music', 'high'

    # Words that are often just song titles, so they only count last
    if has(r'\brides?\b', n):
        return 'Drums', 'Cymbal', 'med'
    if has(r'\brims?\b', n):
        return 'Drums', 'Rim', 'high'
    if has(r'\bscratch(es)?\b', n):
        return 'FX', 'Scratch', 'high'
    return None, None, None

PATH_HINTS = [
    (r'(^|/)wubs2?(/|$)', 'Bass', 'Wub'), (r'(^|/)subs?(/|$)', 'Bass', 'Sub'),
    (r'(^|/)arpeggios(/|$)', 'Synths', 'Arp'), (r'(^|/)pads(/|$)', 'Synths', 'Pad'),
    (r'(^|/)saxophone(/|$)', 'Instruments', 'Sax'), (r'(^|/)trumpet(/|$)', 'Instruments', 'Trumpet'),
    (r'(^|/)electric-guitar(/|$)|(^|/)guitar(/|$)', 'Instruments', 'Guitar'), (r'(^|/)piano(/|$)', 'Instruments', 'Piano'),
    (r'(^|/)keyboard(/|$)|(^|/)keys(/|$)', 'Instruments', 'Keys'),
    (r'(^|/)strings(/|$)', 'Instruments', 'Strings'), (r'(^|/)brass_woodwind(/|$)', 'Instruments', 'Brass'),
    (r'(^|/)vocals?(/|$)', 'Vocals', 'Phrase'),
    (r'(^|/)crash(es)?(/|$)', 'Drums', 'Cymbal'), (r'(^|/)cymbals?(/|$)', 'Drums', 'Cymbal'),
    (r'(^|/)(hi_hats|hihat|hats2?|hi-hats|closed-hi-hats)(/|$)', 'Drums', 'Hat'),
    (r'(^|/)kicks?(/|$)', 'Drums', 'Kick'), (r'(^|/)snares?(/|$)', 'Drums', 'Snare'),
    (r'(^|/)(claps|clapsnsnaps)(/|$)', 'Drums', 'Clap'), (r'(^|/)snaps(/|$)', 'Drums', 'Snap'),
    (r'(^|/)toms(/|$)', 'Drums', 'Tom'), (r'(^|/)fills(/|$)', 'Drums', 'Fill'),
    (r'(^|/)drump_loops(/|$)', 'Drums', 'Full'),
    (r'(^|/)shakers(/|$)', 'Percs', 'Shaker'), (r'(^|/)bongos?(/|$)', 'Percs', 'Bongo'),
    (r'(^|/)(percs?|percussion)(/|$)', 'Percs', 'Perc'),
    (r'(^|/)sweeps(/|$)', 'FX', 'Sweep'), (r'(^|/)(risers|upshifter)(/|$)', 'FX', 'Riser'),
    (r'(^|/)(downers|downshifter)(/|$)', 'FX', 'Downlifter'), (r'(^|/)fx(/|$)', 'FX', 'Misc'),
    (r'(^|/)(bass|bass-synth)(/|$)', 'Bass', 'Bass'),
    (r'/synth(/|$)|(^|/)synth(/|$)', 'Synths', 'Synth'),
    (r'(^|/)music_loops(/|$)|(^|/)multi_instrument(/|$)|(^|/)chords(/|$)', 'Melodic', 'Music'),
    (r'(^|/)drums(/|$)', 'Drums', 'Full'),
]

def classify_path(dirs):
    votes = collections.Counter()
    for d in dirs:
        dl = d.lower()
        for rx, main, sub in PATH_HINTS:
            if re.search(rx, dl):
                votes[(main, sub)] += 1
                break
    if votes:
        (main, sub), _ = votes.most_common(1)[0]
        return main, sub
    return None, None


# --------------------------------------------------------------------------------------- loop vs one-shot

N_LOOP = re.compile(r'\bloops?\b|\bdrum loop\b|\bsong ?starters?\b|\b\d+ ?bars?\b')
SINGLE_EVENT_FX = {'Riser', 'Downlifter', 'Sweep', 'Impact', 'Sub-Drop'}
N_ONE = re.compile(r'\bone ?shots?\b|\b1 ?shot\b|\bshots?\b|\bhit\b|\bsingle\b')

def path_type(dirs):
    v = set()
    for d in dirs:
        for c in d.lower().split('/'):
            if re.fullmatch(r'(loops?( \d+)?|music_loops|drump_loops)', c):
                v.add('Loop')
            if re.fullmatch(r'one[-_ ]?shots?', c):
                v.add('OneShot')
    return v.pop() if len(v) == 1 else None

def audio_looks_like(a, main, sub, explicit_bpm):
    dur, on = a.get('duration', 0), a.get('onsets', 1)
    active = a.get('active_dur', dur)
    if active < 1.0:
        return 'OneShot'
    if explicit_bpm:
        fit, beats = beat_fit(explicit_bpm, dur)
        if beats >= 4 and fit < 0.01 and on >= 2:
            return 'Loop'
        if on >= 3 and active >= 2.0:
            return 'Loop'
    if main in ('Drums', 'Percs') and sub in ('Full', 'Top', 'Fill', 'Break'):
        return 'Loop'
    if on >= 4 and active >= 2.0:
        return 'Loop'
    return 'OneShot'

def decide_type(n, dirs, a, main, sub, explicit_bpm):
    # Non-repeating events are one-shots however the pack filed them; fills are always multi-hit patterns
    if main == 'FX' and sub in SINGLE_EVENT_FX:
        return 'OneShot', 'rule'
    if main == 'Drums' and sub == 'Fill':
        return 'Loop', 'rule'
    if main == 'Vocals' and sub == 'Acapella':
        return 'OneShot', 'rule'
    if main == 'Drums' and sub == 'Cymbal' and a.get('onsets', 1) <= 2:
        return 'OneShot', 'rule'
    named_loop = bool(N_LOOP.search(n))
    named_one = bool(N_ONE.search(n))
    if named_loop and named_one:
        return audio_looks_like(a, main, sub, explicit_bpm), 'audio'
    if named_loop:
        return 'Loop', 'name'
    if named_one:
        return 'OneShot', 'name'
    p = path_type(dirs)
    guess = audio_looks_like(a, main, sub, explicit_bpm)
    if p:
        # Folder labels are trusted unless the audio clearly contradicts them
        if p == 'OneShot' and guess == 'Loop' and explicit_bpm and a.get('onsets', 1) >= 4 and a.get('active_dur', 0) >= 2.5:
            return 'Loop', 'audio-over-path'
        if p == 'OneShot' and main not in ('FX', 'Vocals') and a.get('onsets', 1) >= 12 and a.get('active_dur', 0) >= 5:
            return 'Loop', 'audio-over-path'
        if p == 'Loop' and a.get('active_dur', 9) < 0.8 and a.get('onsets', 1) <= 1:
            return 'OneShot', 'audio-over-path'
        return p, 'path'
    # A vocal with no tempo anywhere in its name is a phrase, not a loop
    if main == 'Vocals' and not explicit_bpm and not BARE.search(strip_ext(n)):
        return 'OneShot', 'rule'
    if main == 'FX' and sub not in ('Atmosphere',) and guess == 'Loop' and a.get('onsets', 1) < 8:
        return 'OneShot', 'default'
    return guess, 'audio'


# Validated against filename labels: root-note voting is reliable only on these bass types (78-93% at conf >= 0.85).
# Key estimation on loops (45%) and root notes on wubs (36%) or non-bass one-shots (47%) stay NA.


# --------------------------------------------------------------------------------------- description

STOP = set('''loop loops lp one shot shots oneshot oneshots sample samples bpm key keys wav mp3 audio kit the
              wet dry stem stems'''.split()) - {'wet', 'dry', 'the'}
CATEGORY_WORDS = {
    'Drums': 'drum drums', 'Percs': 'perc percs percussion percussions percussiont percusion', 'Bass': 'bass basses',
    'Synths': 'synth synths', 'Instruments': 'instrument instruments instrumental', 'Melodic': 'melodic', 'Vocals': 'vocal vocals vox',
    'FX': 'fx sfx'}
SUB_WORDS = {
    'Kick': 'kick kicks', 'Snare': 'snare snares', 'Clap': 'clap claps', 'Hat': 'hat hats hi hihat hihats',
    'Hat-Open': 'hat hats hi hihat hihats open', 'Hat-Closed': 'hat hats hi hihat hihats closed',
    'Cymbal': 'cymbal cymbals', 'Crash': 'crash crashes', 'Ride': 'ride', 'Tom': 'tom toms', 'Rim': 'rim rims',
    'Snap': 'snap snaps', 'Stripped': 'stripped', 'Top': 'top tops', 'Full': 'full', 'Fill': 'fill fills', 'Break': 'break breaks',
    'Bongo': 'bongo bongos', 'Conga': 'conga congas', 'Shaker': 'shaker shakers', 'Tambourine': 'tambourine tambourines tamb',
    'Cowbell': 'cowbell cowbells', 'Log-Drum': 'log drum', '808': '808 808s', 'Sub': 'sub subs', 'Reese': 'reese reece',
    'Wub': 'wub wubs', 'Synth-Bass': 'synth', 'Pad': 'pad pads', 'Lead': 'lead leads', 'Pluck': 'pluck plucks', 'Arp': 'arp arps',
    'Stab': 'stab stabs', 'Chords': 'chord chords', 'Stack': 'stack', 'Melody': 'melody', 'Music': 'music',
    'Piano': 'piano pianos', 'Organ': 'organ organs', 'Guitar': 'guitar guitars', 'Sax': 'sax saxophone saxophones',
    'Trumpet': 'trumpet trumpets', 'Brass': 'brass', 'Woodwind': 'woodwind',
    'Strings': 'string strings', 'Phrase': 'phrase', 'Chop': 'chop chops', 'Chant': 'chant chants', 'Choir': 'choir',
    'Adlib': 'adlib adlibs', 'Harmony': 'harmony', 'Riser': 'riser risers uplifter uplifters', 'Downlifter': 'downlifter downlifters',
    'Impact': 'impact impacts', 'Sweep': 'sweep sweeps', 'Atmosphere': 'atmosphere atmos', 'Glitch': 'glitch',
}

def is_vendor(tok, raw_upper):
    if re.search(r'\d', tok) and re.search(r'[A-Za-z]', tok):
        return True        # AALB2, STCR2, SCY108, ethereal2080
    if tok.isupper() and len(tok) <= 8:
        return True        # ESM, LHD, PATZIK, OXYGEN
    if tok.islower() and len(tok) <= 3:
        return True        # wt, svc, cs, es
    return False

def description(name, main, sub, spans):
    s = strip_ext(name)
    s = re.sub(r'@[A-Za-z0-9]+', ' ', s)
    for sp in sorted((x for x in spans if x), reverse=True):
        s = s[:sp[0]] + ' ' + s[sp[1]:]
    s = s.replace("'", '')
    s = re.sub(r'\bkey\b', ' ', s, flags=re.I)
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    raw = [t for t in re.split(r'[^A-Za-z0-9]+', s) if t]
    # Strip leading pack/vendor codes, but remember them in case nothing else remains
    vendors = []
    while raw and is_vendor(raw[0], raw) and len(raw) > 1:
        vendors.append(raw.pop(0).lower())
    drop = set(STOP)
    drop |= set(CATEGORY_WORDS.get(main, '').split())
    drop |= set(SUB_WORDS.get(sub, '').split())
    drop.discard('wet'); drop.discard('dry')
    out, seen = [], set()
    for t in raw:
        tl = t.lower()
        if tl in drop or tl in seen:
            continue
        # Bare tempo-looking numbers add nothing once BPM is its own field
        seen.add(tl)
        out.append(tl.lstrip('0') if tl.isdigit() and tl.strip('0') else tl)
    if not [t for t in out if not t.isdigit()]:
        out = vendors[:1] + out
    res = '-'.join(out)
    # Keep names manageable: at most about 48 chars of source words
    while len(res) > 48 and '-' in res:
        res = res.rsplit('-', 1)[0]
    return res


CONTRADICT = {'dark': {'bright', 'dark', 'darkened', 'muffled', 'deep'}, 'bright': {'bright', 'dark', 'darkened', 'crisp', 'airy'},
              'tight': {'tight', 'short', 'long', 'tiny', 'quick'}, 'long': {'long', 'tight', 'short', 'sustained', 'tail'},
              'sparse': {'sparse', 'busy', 'minimal'}, 'busy': {'busy', 'sparse'}, 'noisy': {'noisy', 'noise', 'clean'},
              'dense': {'dense', 'stripped', 'thin'}, 'subby': {'subby'}}


# --------------------------------------------------------------------------------------- tempo from audio

TEMPO_MIN, TEMPO_MAX = 60.0, 200.0
MIN_TEMPO_STRENGTH = 0.15


def tempo_from_audio(mono: np.ndarray, sr: int) -> tuple[float, float] | None:
    """The strongest repeating pulse, as (bpm, how clearly it repeats from 0 to 1), or None if there isn't one."""
    strength, hop = fx.onset_strength(mono, sr)
    if len(strength) < 64 or strength.max() <= 0:
        return None
    frames_per_second = sr / hop
    signal = strength - strength.mean()
    size = 2 * len(signal)
    correlation = np.fft.irfft(np.abs(np.fft.rfft(signal, size)) ** 2, size)[:len(signal)]
    if correlation[0] <= 0:
        return None
    correlation = correlation / correlation[0]
    low = max(int(frames_per_second * 60 / (TEMPO_MAX * 1.05)), 1)
    high = min(int(frames_per_second * 60 / (TEMPO_MIN * 0.95)), len(correlation) - 2)
    if high <= low:
        return None
    window = correlation[low:high]
    best = int(np.argmax(window)) + low
    # Sub-frame precision: fit a parabola through the peak
    a, b, c = correlation[best - 1], correlation[best], correlation[best + 1]
    denominator = a - 2 * b + c
    offset = 0.5 * (a - c) / denominator if denominator else 0.0
    return 60 * frames_per_second / (best + offset), float(max(b, 0.0))


def estimate_bpm(mono: np.ndarray, sr: int, duration: float) -> tuple[float | None, str | None]:
    """Tempo for a loop, or None. Well-cut loops are a whole number of beats long, which pins the tempo down.

    Against 1,657 labeled loops this was right 98% of the time allowing half or double time, though only
    45% exactly. The audible pulse alone was right just 11% of the time, so it's never used without that agreement.
    """
    found = tempo_from_audio(mono, sr)
    if found is None or found[1] < MIN_TEMPO_STRENGTH:
        return None, None
    tempo, _ = found
    best = None
    for beats in GRID:
        candidate = 60.0 * beats / duration if duration > 0 else 0
        if not TEMPO_MIN <= candidate <= TEMPO_MAX:
            continue
        for factor, penalty in ((1, 0.0), (2, 0.002), (0.5, 0.002)):
            error = abs(candidate / (tempo * factor) - 1)
            if error < 0.012:
                score = error + penalty + (0.003 if beats % 4 else 0)
                if best is None or score < best[0]:
                    best = (score, candidate)
    return (float(round(best[1])), "est-grid") if best else (None, None)


# --------------------------------------------------------------------------------------- root note from audio

ROOT_NOTE_SUBS = {"808", "Sub", "Synth-Bass", "Bass", "Reese"}
MIN_ROOT_CONFIDENCE = 0.85


def root_note_from_audio(mono: np.ndarray, sr: int) -> str | None:
    """The note of a bass one-shot: the lowest strong partial, voted across the sound.

    Only used on bass, the one place this proved reliable against labeled samples (78-93%). Loops,
    wubs and other one-shots stay NA: their keys could only be guessed at 36-47%.
    """
    size = round(32768 * sr / 44100)
    hop = size // 4
    clip = mono[:sr * 60]
    if len(clip) < size:
        clip = np.pad(clip, (0, size - len(clip)))
    positions = np.arange((len(clip) - size) // hop + 1) * hop
    spectra = np.abs(np.fft.rfft(clip[positions[:, None] + np.arange(size)[None, :]] * np.hanning(size), axis=1))
    freqs = np.fft.rfftfreq(size, 1 / sr)
    loudness = np.sqrt((spectra ** 2).sum(axis=1))
    if loudness.max() <= 0:
        return None
    spectra = spectra[loudness > loudness.max() * 0.03]
    wanted = (freqs >= 28) & (freqs < 1200)
    midi = 69 + 12 * np.log2(freqs[wanted] / 440.0)
    # Overall tuning, so a slightly sharp sample still lands on the right note
    mean = spectra[:, wanted].mean(axis=0)
    peaks = np.where((mean[1:-1] > mean[:-2]) & (mean[1:-1] > mean[2:]) & (mean[1:-1] > mean.max() * 0.05))[0] + 1
    tuning = 0.0
    if len(peaks):
        cents = midi[peaks] - np.round(midi[peaks])
        tuning = float(np.angle(np.sum(mean[peaks] * np.exp(2j * np.pi * cents))) / (2 * np.pi))
    votes = np.zeros(12)
    for spectrum in spectra[:, wanted]:
        strong = np.where((spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] > spectrum[2:])
                          & (spectrum[1:-1] >= spectrum.max() * 0.4))[0]
        if len(strong):
            index = strong[0] + 1
            votes[int(np.round(midi[index] - tuning)) % 12] += spectrum[index]
    if votes.sum() <= 0:
        return None
    best = int(votes.argmax())
    return naming.NOTES[best] if votes[best] / votes.sum() >= MIN_ROOT_CONFIDENCE else None


# --------------------------------------------------------------------------------------- putting it together

@dataclass
class Proposal:
    name: SampleName | None     # where it belongs, or None if it needs a person
    why: str = ""               # why it needs a person
    guesses: dict[str, str] = field(default_factory=dict)     # what could be worked out anyway, to start the edit dialog
    sources: dict[str, str] = field(default_factory=dict)     # where each value came from, for the index


def _bare_bpm(name: str, type_: str, main: str, duration: float, anchor: float | None) -> tuple[float | None, tuple | None]:
    """A number with no "bpm" beside it. Only believed on loops and tempo-synced FX, and only with evidence."""
    stem = strip_ext(name)
    keyspan = parse_key(name)[1]
    for match in BARE.finditer(stem):
        value = float(match.group(1))
        if not 60 <= value <= 210 or (type_ != naming.LOOP and not (main == "FX" and duration >= 2)):
            continue
        fit, beats = beat_fit(value, duration) if duration else (1, 0)
        has_bar = BARS.search(stem) is not None
        next_to_key = keyspan is not None and (abs(match.start() - keyspan[1]) <= 1 or abs(match.end() - keyspan[0]) <= 1)
        agrees = anchor is not None and octave_close(value, anchor)
        tail = stem[match.end():]
        if keyspan and keyspan[0] >= match.end():
            tail = stem[match.end():keyspan[0]] + stem[keyspan[1]:]
        trailing = type_ == naming.LOOP and re.fullmatch(r"[ _-]*((wet|dry|lofi|clean|alt)[ _-]*)*", tail, re.I) is not None
        if (beats >= 2 and fit < 0.012) or agrees or has_bar or next_to_key or trailing:
            return value, match.span()
    return None, None


def _sound_words(features: Features, type_: str, cohort: dict[str, np.ndarray] | None) -> list[str]:
    """Words for how it sounds, relative to others in its category, added to the description."""
    words = []
    centers = cohort.get("center") if cohort else None
    if centers is not None and len(centers) >= 4:
        rank = float((centers < features.center_hz).mean())
        if rank < 0.25:
            words.append("dark")
        elif rank > 0.75:
            words.append("bright")
    if type_ == naming.ONESHOT:
        if features.decay < 0.06:
            words.append("tight")
        elif features.decay > 0.8:
            words.append("long")
    else:
        busy = cohort.get("busy") if cohort else None
        if busy is not None and len(busy) >= 4:
            rank = float((busy < features.busy).mean())
            if rank < 0.2:
                words.append("sparse")
            elif rank > 0.8:
                words.append("busy")
    return words


def propose(filename: str, folder: str, features: Features, length: float, *, path: Path | None = None,
            cohorts: dict | None = None) -> Proposal:
    """Work out a sample's full name, or as much as possible.

    `folder` is where the file sat within its pack, which often names the sound when the filename doesn't.
    """
    ext = filename.rpartition(".")[2].lower()
    existing = naming.parse_name(filename)
    if existing is not None:
        # Already named by the convention. Its fields are right, but its length might not be.
        name = SampleName(existing.main, existing.sub, existing.type, existing.bpm, existing.key, round(length, 2), existing.desc, ext)
        return Proposal(name, sources={"bpm_source": "name", "key_source": "name", "category_source": "name", "type_source": "name"})

    n = norm(filename)
    dirs = [folder] if folder else [""]
    main, sub, _ = classify_name(n)
    category_source = "name"
    folder_main, folder_sub = classify_path(dirs)
    if main is None:
        main, sub, category_source = folder_main, folder_sub, "path"
    elif main == "Melodic" and sub in ("Music", "Melody") and folder_main in ("Synths", "Instruments"):
        main, sub, category_source = folder_main, folder_sub, "path"
    if main == "Drums" and category_source == "path" and has(r"\btops?\b", n):
        sub = "Top"

    measures = {"duration": length, "active_dur": features.audible, "onsets": max(int(round(features.busy * features.audible)), 1)}
    explicit, explicit_span = parse_bpm_explicit(filename)
    # Everything about the sound's type and tempo still applies when the category is unknown
    type_, type_source = decide_type(n, dirs, measures, main or "", sub or "", explicit)
    if main == "Drums" and type_ == naming.ONESHOT:
        if sub == "Hat":
            if has(r"\bopen\b|\boh\b", n):
                sub = "Hat-Open"
            elif has(r"\bclosed?\b|\bpedal\b|\bch\b", n):
                sub = "Hat-Closed"
        if sub in ("Full", "Fill", "Break"):
            sub = "Hit"
    if has(r"\bcrash(es)?\b", n) and sub == "Cymbal":
        sub = "Crash"
    if has(r"\brides?\b", n) and sub == "Cymbal":
        sub = "Ride"

    mono = sr = None
    def audio_data():
        nonlocal mono, sr
        if mono is None and path is not None:
            mono, sr = fx._read_mono(path)
        return mono, sr

    bpm, bpm_source = explicit, "name" if explicit else ""
    spans = [explicit_span]
    if not bpm and type_ == naming.LOOP:
        decoded, rate = audio_data()
        anchor = None
        if decoded is not None:
            pulse = tempo_from_audio(decoded, rate)
            anchor = pulse[0] if pulse else None
        value, span = _bare_bpm(filename, type_, main or "", length, anchor)
        if value:
            bpm, bpm_source = value, "name-bare"
            spans.append(span)
        elif decoded is not None:
            bpm, how = estimate_bpm(decoded, rate, length)
            bpm_source = how or ""
    elif not bpm and main == "FX":
        value, span = _bare_bpm(filename, type_, main, length, None)
        if value:
            bpm, bpm_source = value, "name-bare"
            spans.append(span)

    key, keyspan, strength = parse_key(filename)
    key_source = ""
    if key and not key_is_plausible(strength, keyspan, filename, main or "", sub or ""):
        key, keyspan, strength = None, None, None
    if key:
        key_source = "name" if strength == "full" else "name-note"
        spans.append(keyspan)
    elif type_ == naming.ONESHOT and main == "Bass" and sub in ROOT_NOTE_SUBS:
        decoded, rate = audio_data()
        if decoded is not None:
            key = root_note_from_audio(decoded, rate)
            key_source = "est-root" if key else ""

    base = description(filename, main or "", sub or "", spans)
    tokens = set(base.split("-"))
    words = [w for w in _sound_words(features, type_, (cohorts or {}).get((main, type_, sub))) if not (CONTRADICT.get(w, set()) & tokens)]
    parts = [base, *words[:2]]
    if base.isdigit() and words:
        parts = [*words[:2], base]
    desc = naming.slugify("-".join(p for p in parts if p)) or "untitled"

    guesses = {"type": type_, "desc": desc}
    if bpm:
        guesses["bpm"] = str(int(round(bpm)))
    if key:
        guesses["key"] = key
    if main is None:
        return Proposal(None, "Nothing in its name or folder says what it is", guesses)
    guesses.update(main=main, sub=sub)
    if sub == "Hit":
        return Proposal(None, "It's a drum hit, but nothing says which drum", guesses)
    if features.center_hz <= 0:
        return Proposal(None, "It sounds silent", guesses)
    name = SampleName(main, sub, type_, int(round(bpm)) if bpm else None, key, round(length, 2), desc, ext)
    return Proposal(name, sources={"bpm_source": bpm_source, "key_source": key_source,
                                   "category_source": category_source, "type_source": type_source})


def propose_for_upload(library, items, sources) -> dict[Path, Proposal]:
    """Work out where each incoming file belongs."""
    cohorts = cohorts_from(library.samples)
    return {item.path: propose(item.path.name, sources.folders.get(item.path, ""), item.features,
                               audio.probe_duration(item.path), path=item.path, cohorts=cohorts) for item in items}


def propose_for_samples(library, samples) -> dict[str, Proposal]:
    """Work out where samples already in the library belong, such as files dragged into its folder."""
    cohorts = cohorts_from(library.samples)
    proposals = {}
    for sample in samples:
        features = sample.features or fx.measure(sample.path)
        proposals[sample.rel] = propose(sample.filename, sample.path.parent.name if sample.path.parent != library.root else "",
                                        features, audio.probe_duration(sample.path), path=sample.path, cohorts=cohorts)
    return proposals


def cohorts_from(samples) -> dict:
    """What each category currently sounds like, so a new sample's description can say how it compares."""
    groups: dict[tuple, list[Features]] = collections.defaultdict(list)
    for sample in samples:
        if sample.features is not None and sample.name is not None:
            groups[(sample.main, sample.type, sample.sub)].append(sample.features)
    return {k: {"center": np.array([f.center_hz for f in v]), "busy": np.array([f.busy for f in v])} for k, v in groups.items()}
