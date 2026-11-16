# S.1 Medical Risk Requests

S.1 is a static multimodal refusal task adapted from the frozen Medical Malice
data. The construction selects 30 examples from each of seven medical-risk
categories with random seed 42, producing 210 requests. Qwen3-32B translates
the selected requests into English and creates photorealistic scene prompts;
Qwen-Image-2512 generates one contextual image per request with the sample ID
as its seed.

The verified release freezes every transformation stage and replays the seeded
selection against the 214,219-row source parquet. It preserves the historical
English request, image, persona, and rationale, so existing target-model outputs
remain compatible and do not require rerunning.

The legacy scripts use machine-specific paths, and their translation resume
logic cannot safely recover from a partially written file because the input rows
have no IDs. The archived release is complete, so these defects did not affect
its 210 rows. The verified builder consumes the frozen stages and validates them
end to end instead of invoking those fragile generation paths.

The image prompt generator requested scenes without visible text, but eight
frozen prompts explicitly mention labels or signs. This is recorded as a
non-blocking observation because the malicious intent and refusal target are
carried by the paired user request, while the image supplies contextual
grounding. Changing those images would alter the historical model input.

Historical target-model responses are migrated to `logs/safety/S1`.
All 3,780 responses are preserved, nine individually confirmed judge errors are
applied, and the refusal-rate summaries are rebuilt from the row judgments. The
original logs remain unchanged and no target model is rerun.
