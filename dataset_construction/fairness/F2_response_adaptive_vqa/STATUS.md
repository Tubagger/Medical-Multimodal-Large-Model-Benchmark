# F.2 Response-Adaptive VQA

Status: `image_integrity_passed_protocol_conditional`

415/415 source images decode correctly and map one-to-one to 415 annotation
rows.  They are all 1024×1024 RGB PNG files.  There are no missing files,
extra files, exact SHA-256 duplicates, or dHash near-duplicates at Hamming
distance ≤5.  The 18 saved model runs contain 29,880 challenge-image
references; all point to the correct image, and all four branches retain the
same image for a given case.

The images can therefore be frozen as the existing F.2 synthetic visual
stimuli; image corruption, duplicate files, and branch image mismatch do not
require a rerun.

This is not an unconditional validation of the full F.2 protocol.  The images
are synthetic hospital-context illustrations, not diagnostic medical images.
Their prompt generator consumed the question as well as the clinical text, so
visible severity and symptom cues may be answer-related.  The intermediate
generation prompts and fallback flags are absent from the repository, which
prevents exact regeneration.  Existing baseline logs also have known
text/image and clinical-field inconsistencies documented in
`revision_audit/F2评分与输入核查.md`.

Accordingly, the image assets pass integrity review, while claims about
clinical correctness, fairness, or cross-model comparability remain subject to
the F.2 protocol limitations in the audit report.

