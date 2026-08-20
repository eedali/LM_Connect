# Full-Reference Mode — Compact Reference

## Six Sections (exact order)

1. `subject_definitions` — Define each referenced content item with its label and role.
2. `summary` — One paragraph: `[task-type prefix] ...` summarizing target video and reference relationships.
3. `retention_analysis` — One line per label: where it appears, relationship marker, and explanation.
4. `detailed_description` — Shot-by-shot visuals, actions, sound, dialogue in playback order with reference labels.
5. `overall_soundscape` — Ambience and physical sounds (1–4 sentences).
6. `non_diegetic_music` — Audience-only background music (1–3 sentences) or `N/A`.

## Reference Labels

| Label | Meaning |
|---|---|
| `<Subject N>` | Reusable visible content (person, scene, clothing, action, style, etc.) |
| `<Picture N>` | Reference image used as a frame anchor or shot-planning reference |
| `<Video N>` | Reference video providing editing source, continuation, or temporal structure |
| `<Audio N>` | Audio signal that is copied or referenced |

Labels keep the same meaning across ALL six sections.

## subject_definitions Rules

- One line per item. State label, reference role, key features, source asset when needed.
- `<Subject N>`: for reusable visible content. One subject may come from multiple assets.
- `<Picture N>`: standalone entry ONLY when image is a frame anchor (first/last/keyframe). If image only defines a character/scene, cite it inside `<Subject N>` instead.
- `<Video N>`: for whole-video relationships (editing, continuation, temporal structure reference). Visible content from video → use `<Subject N>`.
- `<Audio N>`: audio roles (copy, timbre reference, rhythm). When linked to a speaker, reuse their `(Sx)` ID.
- `<Video N>` and `<Audio N>` numbered independently.

## summary

- Begin with square-bracketed task-type prefix.
- Task types: `keyframe completion`, `reference generation`, `video editing`, `video continuation`, `audio reuse`, `audio reference`. Combine with ` + `.
- Use previously defined labels. Do NOT introduce new labels here.

## retention_analysis

### Visual markers
`fully_preserved`, `partially_preserved`, `attribute_transfer`, `weak_reference`

### Audio markers
`fully_copy`, `partially_copy`, `reference`, `weak_reference`

Format: `<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - ...`

## detailed_description

- Style statement: 1–2 sentences BEFORE `[Shot 1]`.
- `[Shot 1]` no timestamp. Later shots: `[Shot N] At MM:SS.mmm, ...`
- Target length for generation tasks: 350–500 English words.
- Insert reference labels at first appearance and where roles apply.
- Camera, dialogue, `<d>`, `(Sx)`, `<scenetrans>`, `<cutoff>` rules: same as base guide.
- Speakers who are referenced subjects: `<Subject N> (Sx)`.
- Audio cues within reused BGM: use `<Audio N>` as source, no extra `(Sx)`.
- Preserve exact source dialogue words in `<d>`. Write `[unclear]` for unintelligible spans.

## Prohibitions

- Do NOT invent unsupported actions, expressions, events, transitions, visible text, props, locations, or details.
- Do NOT add cuts or camera movement solely for cinematic embellishment.
- Do NOT create one target shot per contact-sheet cell. Never mention contact sheet/cells/sampled frames in output.
- Do NOT infer music from mood/style — `N/A` unless explicitly requested.
- Do NOT classify ordinary full-reference images as keyframe-completion tasks.
- When user assigns a reference a specific role, transfer ONLY that role. Motion-only video must NOT contribute performer identity, clothing, location, background, lighting, or audio.
- Do NOT mention instructions, compliance checks, or word counts in the output.
- Do NOT translate user-supplied dialogue.
