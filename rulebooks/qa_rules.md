# QA Rules

- Extract frames at 10 FPS in E2B for verification before delivery:
  1. Animation loops must play twice without stutter.
  2. No sub-second flash frames.
  3. Captions must align to audio timestamps (within tolerance).
  4. Visual elements must not occlude safe-zone boundaries.
  5. Audio/video sync must be within acceptable tolerance.
- Auto-patch detected issues and re-verify before final AAC mux.
