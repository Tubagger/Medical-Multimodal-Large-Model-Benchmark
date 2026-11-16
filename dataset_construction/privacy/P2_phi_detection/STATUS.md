# P.2 status

Status: **passed**

- task scope is PHI detection only;
- 260 unique images and IDs;
- 13 CheXpert disease categories × 20 sources;
- two to four synthetic PHI fields per image;
- seed-42 selection and overlays exactly replayed;
- all 260 output images reproduced byte-for-byte and pixel-for-pixel;
- 18 models × 260 responses reparsed and rescored;
- `privacy-inflow` is cancelled and absent from active registration.

Four images have only boundary anti-aliasing pixels touching the central-region
check, and 133 generated MRN fields use the form `MRN-xxxxxx`. These are
recorded observations and do not invalidate the protocol.

