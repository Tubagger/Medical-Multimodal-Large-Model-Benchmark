# P.2 PHI Detection

P.2 is limited to PHI detection. The cancelled `privacy-inflow` task is not
part of this release, and privacy inference is handled as a separate task.

The accepted cohort contains 260 CheXpert images: 13 single-label disease
categories with 20 source images per category. Each image contains two to four
synthetic PHI fields. Source selection and overlay generation both use seed 42.

Validation replays the original constructor against the archived upstream
CheXpert-SingleLabel inputs. It exactly reproduces source selection, overlay
metadata, and all 260 output images byte-for-byte and pixel-for-pixel.

Existing complete responses for 18 models are migrated to
`logs/privacy/P2`. All 4,680 responses are reparsed with the shared P.2
parser, and per-row decisions plus Precision, Recall, and F1 are rebuilt.
Historical source data and logs remain unchanged.

