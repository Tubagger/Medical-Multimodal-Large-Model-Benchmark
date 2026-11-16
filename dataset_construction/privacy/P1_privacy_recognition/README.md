# P.1 Privacy Recognition

P.1 is the accepted frozen 260-image, three-class CheXpert privacy-recognition
cohort. The original protocol is preserved: clean images receive no benchmark
overlay, PHI images receive synthetic identifier fields, and noise images
receive irrelevant synthetic text.

The verification replays both construction stages from the archived upstream
CheXpert-SingleLabel labels with seed 42. It exactly reproduces the 13 × 20
source selection, all metadata, and all 260 rendered images pixel-for-pixel.

The accepted observations about acquisition markers in clean images, synthetic
rather than real identifiers, and unequal overlay density are disclosed in the
manifest and validation report but do not block this protocol.

Historical results are not copied blindly. Twelve models use their unique
saved responses and six use existing isolated 260-item reruns. All 4,680
responses are parsed with the current shared P.1 parser, and Accuracy,
Macro-F1, row judgments, and processed summaries are rebuilt.
