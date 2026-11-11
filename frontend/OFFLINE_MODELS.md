# On-device recognition

GroundInk performs image processing, word detection, character recognition, and
spelling feedback locally. It does not require a backend connection. ONNX
inference runs in a worker isolate so the interface can report progress while
the page is processed.

The app bundles these runtime assets in `assets/models/`:

- `word_detection_model.onnx`: YOLO26n Nano word detection (input size:
  `640 x 640` pixels).
- `char_model_nano_256.onnx`: YOLO26n Nano character segmentation: one box,
  class and outline (mask) per letter (input size: `256 x 256` pixels). It was
  trained on the ink outline of every letter.
- `english_lm.bin.gz`: the spelling model — the English word list (the same
  160,572 words as before), word frequencies, word-pair statistics, and the
  words that are never suggested.

Character regions are mapped back to the source image so spelling feedback can
highlight the corresponding handwritten character. The highlight colors the
dark pixels in the letter's box that the letter's predicted outline covers
(mask probability above 0.2 and higher than that of any other letter), so
where letter boxes overlap, the ink of the neighboring letters stays as it is
(`lib/services/letter_mask.dart`, `CharacterInk.extractWithMasks`). The ink color of each word
is measured as well: on red, pink or orange handwriting the suspect letters are
highlighted in blue instead of red.

## Spelling suggestions

A word is a mistake when it is not in the word list. Suggestions are ranked
with a noisy-channel model, `P(word | written, context)`:

- **Error model** (`lib/services/spelling/error_model.dart`): a weighted edit
  distance. Typical children's mistakes cost less than arbitrary edits
  (missing double letters, silent letters as in "wach"/"watch" or
  "scool"/"school", vowel and sound-alike confusions, swapped letters), and so
  do typical recognition errors (letters that look alike, "rn" read for "m",
  "0" for "o"). Letters the character model was unsure about are cheaper to
  replace.
- **Word frequency** `P(word)`: subtitle frequencies (OpenSubtitles, via
  pyspellchecker), mixed with Google Books frequencies.
- **Context**: how strongly a candidate is associated with the previous and
  the next word on the page (pointwise mutual information from Google Books
  word pairs). For example "whent" alone is ranked "when", but in "I whent to"
  it is "went".

Profanity and slurs (`tool/data/suggestion_blocklist.txt`) are never
suggested; subtitle frequencies would otherwise rank them very high. Written
words are still accepted if they are in the word list.

On the Holbrook corpus of children's writing (1,222 misspellings in their
sentences), the intended word is ranked first for 57.0% of the misspellings
(previously 17.0% with plain edit distance) and is among the first five for
75.9% (previously 43.5%). `tool/evaluate_spelling.dart` reproduces these
numbers and the weight search; the corpora are linked in that file.

`tool/build_language_model.py` rebuilds `english_lm.bin.gz` from its sources.

## Data sources

- Word list and subtitle word frequencies: pyspellchecker 0.9.0 (MIT License),
  built from OpenSubtitles2018 (P. Lison and J. Tiedemann, 2016).
- Google Books word and word-pair frequencies: SymSpell English frequency
  dictionaries (MIT License), derived from the Google Books Ngram data, which
  is licensed under Creative Commons Attribution 3.0
  (https://creativecommons.org/licenses/by/3.0/).
