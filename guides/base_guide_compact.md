# Video Prompt Writing Guide — Compact Reference (T2VA / I2VA / FL2VA / L2VA)

## Modes & Instructions

- **T2VA**: No image instruction. Begin directly with the three core fields.
- **I2VA**: First line: `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.`
- **FL2VA**: First line: `How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.`
- **L2VA**: First line: `How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.`

`N` = final shot index. `S.SS` = video duration, two decimal places. Instruction is the first line, blank line before core fields.

## Three Core Fields (in order)

```
integrated_multimodal_description: [Shot 1] ...
overall_soundscape: ...
non_diegetic_music: ...
```

## Keyframe Rules

- **I2VA**: Picture 1 = first frame at 0.00s in Shot 1. Anchor style/subjects/composition from image, then develop forward. Structure: first-frame anchor → action onset → development → result.
- **FL2VA**: Picture 1 = opening, Picture 2 = ending. Describe the motion path connecting them. Favor single shot unless multi-shot explicitly specified. Structure: first-frame state → intermediate changes → narrowing differences → last-frame state.
- **L2VA**: Picture 1 = final frame in last Shot N. Infer plausible earlier state, describe convergence. Structure: preceding state → action path → convergence → last-frame landing.

## Shot Format

- `[Shot 1]` — no timestamp. State style and initial composition here.
- `[Shot 2] At 00:03.500, the camera cuts to...` — strictly increasing timestamps within duration.
- Cut phrases: `the camera cuts to`, `the shot cuts to/transitions to/changes to/switches to`.

## Camera Motion

Three dimensions: **motion type** + **amplitude** + **speed**. Write as natural English within the shot.

Motion types: `Zoom In/Out`, `Push In/Pull Out`, `Pan Left/Right`, `Truck Left/Right`, `Tilt Up/Down`, `Pedestal Up/Down`, `Arc Shot`, `Tracking Shot`, `Static Shot`, `Shake Slightly/Strongly`, `POV`, `Roll Clockwise/Counterclockwise`.
Amplitude: `with small amplitude`, `with large amplitude`. Speed: `at slow speed`, `at fast speed`.

## Speakers & Dialogue

- Stable IDs: `(S1)`, `(S2)`. Compound: `(S1,S2)`. Same ID across shots. Non-vocal characters get no ID.
- Format: `The young woman (S1) says: <d>[English] I get off at the next station.</d>`
- Voiceover: `says in an off-screen voiceover: <d>...</d> while his lips remain completely closed.`
- Cross-cut dialogue: `<scenetrans>` at connecting points, state audio continues.
- Truncated speech: `<cutoff>`.
- Preserve all user-supplied dialogue verbatim. Do not translate.

## On-Screen Text

Wrap in English double quotes. Preserve original text verbatim: `A red neon sign reading "营业中" glows above the doorway.`

## overall_soundscape

1–4 sentences. Ambient sound, physical sounds, non-verbal human sounds. No dialogue/singing (those go in multimodal description). `N/A` only if user explicitly requests silence.

## non_diegetic_music

1–3 sentences. Background music only audience hears. Focus on instrumentation, speed, rhythm, dynamics. No mood words. `N/A` when none.

## Prohibitions

- Do NOT invent unsupported actions, expressions, events, transitions, visible text, props, or locations.
- Do NOT add cuts or camera movement solely for cinematic embellishment.
- Do NOT infer music from mood/style — only include if explicitly requested.
- Do NOT mention instructions, compliance checks, or word counts in output.
- Do NOT translate user-supplied dialogue.
