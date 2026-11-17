# Third-Party Notices

The root [CC BY-NC-SA 4.0 license](LICENSE) applies to original project
contributions unless otherwise indicated. It does not replace upstream licenses
or grant rights to third-party data, weights, or other materials.

## LLaVA components

Several files in `ours/models/llava/` retain copyright notices identifying
Haotian Liu (2023) and the Apache License, Version 2.0. Examples include
`model/llava_arch.py`, `model/builder.py`, and language-model adapters.
Those upstream portions remain under [Apache 2.0](licenses/Apache-2.0.txt).
Preserve their copyright and license headers when redistributing them.

## Dependencies and model implementations

External packages, model implementations, checkpoints, and any other files
with upstream notices remain subject to their respective licenses. A package
listed in `requirements.txt` or an environment snapshot is not relicensed by
this repository. This notice is not a claim that every dependency has been
individually audited for license compatibility.

## Source datasets and derived materials

Source medical images, annotations, third-party attack artifacts, and model
weights are not included in this code-only release. Their original licenses
and access conditions govern their use. Construction scripts and manifests do
not grant permission to access or redistribute those materials. Check source
terms for derived images, overlays, annotations, and publication examples.
