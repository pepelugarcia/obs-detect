# Scope — obs-detect 0.0.22: per-frame crop glide

**Status:** implemented 2026-09-08 (flag default ON per Jose); CI build and rig soak pending. Install only with OBS closed, after wrap.
**Rig:** MOCAP-REMOTE, RTX 5060 Ti, four 4K DeckLink cams, OBS 32.1.2, canvas 2560×1440@30.
**Written:** 2026-09-08, from measurements taken on the live rig that morning.

---

## 1. The problem, in one sentence

The auto-zoom crop only moves **when the detector runs**, so at `detect_interval` 2 the pan updates at
15 Hz and at 3 it updates at 10 Hz — visibly stepped — while every-frame detection (interval 1), which
*does* pan smoothly, overloads the render thread and drops a quarter of all frames.

## 2. Evidence (measured 2026-09-08)

| setting | crop updates/s (poll of Detect Tracking insets) | OBS render | dropped frames |
|---|---|---|---|
| interval 1, all medium | ~30 (every frame) | 45 ms, 22 fps | **+204 in 25 s**; 09-02/09-04 sessions logged **24–28 % lagged** |
| interval 2, all medium | **14.5** (median dwell 67 ms) | 30.5 ms, 30 fps | 0 |
| interval 3, all medium | **9.7** (median dwell 104 ms) | 27 ms, 30 fps | 0 |
| interval 2, cam 3 small | 14.5 | 29.5 ms, 30 fps | 0 (current live state) |

GPU utilisation sits at ~35 % throughout: the ceiling is **latency** (four DirectML inferences run serially
on the render thread), not GPU capacity. No settings combination gives both a 30 Hz pan and zero drops —
small models on the tight cams bought ~4 ms, not the ~15 needed.

The code already documents the limitation — `detect-filter.cpp:476-480`:

> every-frame is what makes panning smooth: at interval > 1 the crop HOLDS between updates, which reads as
> choppy framing even while the output sits at a full 30fps.

## 3. Root cause in the code

`detect_filter_video_tick()` (line 855) is called by OBS **every rendered frame** with `seconds` (dt), but:

```
884:  if (!tf->inputFresh) return;          // nothing captured since last inference -> do nothing
```

Everything downstream — target selection, the median filter, **the easing step** and the crop push —
runs only when a fresh readback exists, i.e. once per `detect_interval` frames:

```
1414:  float posF   = tf->zoomSpeedFactor     * (lostTracking ? 0.2f : 1.0f);
1420:  float sizeInF  = tf->zoomSizeSpeedFactor * (lostTracking ? 0.2f : 1.0f);
1421:  float sizeOutF = std::min(sizeInF * 4.0f, 0.06f);
1434:  tf->trackingRect.height += (sizeErr > 0 ? sizeOutF : sizeInF) * sizeErr;   // ease size
1447:  tf->trackingRect.x += posF * (tx - tf->trackingRect.x);                    // ease position
1478:  obs_source_update(tf->trackingFilter, crop_pad_settings);                  // push crop
```

The *target* of that easing (`zh`, the box centre, `lostTracking`) is computed as locals and thrown
away; `FilterData` keeps only the smoothed `trackingRect`. So between inferences there is nothing to
glide toward, and the crop holds.

## 4. The change: decouple the glide from detection

Split the tracking math into two halves:

* **On inference frames (unchanged cadence):** pick the target, run the 5-sample median, compute the
  target height `zh` and centre `(cx, cy)`, and **store them** in `FilterData` instead of easing.
* **On every `video_tick` (new):** ease `trackingRect` toward the stored target and push the crop —
  with factors rescaled so the *feel per detection interval* is exactly what the sliders already say.

### 4.1 New `FilterData` state

```cpp
/* Per-frame crop glide (0.0.22). Inference sets a TARGET every detect_interval
   frames; video_tick eases trackingRect toward it EVERY rendered frame, so the
   pan moves at the render rate whatever the interval. */
bool  smoothEveryFrame;      // setting "smooth_every_frame" (default true)
bool  targetValid;           // a target exists since (re)start / model reload
float targetCX, targetCY;    // target window centre, READBACK space
float targetH;               // target window height after the median, READBACK space
bool  targetLost;            // lostTracking at the last inference (easing x0.2)
int   rawW, rawH;            // readback dimensions at the last inference (clamp space)
int   lastCropL, lastCropT, lastCropR, lastCropB;  // last ints pushed; skip identical pushes
```

### 4.2 The per-frame step — one function, called from one place

`apply_crop_glide(tf)` runs at the top of `video_tick`, **before** the `inputFresh` early return, when
`targetValid && trackingRect.width > 0`. It contains the existing math verbatim (deadband 8 %,
asymmetric widen ×4 capped 0.06, hug-edges, 15 % implosion floor, scale-out to source space) with
exactly two differences:

1. It reads the **stored** target instead of locals.
2. Its factors are **rescaled per frame** (§4.3).

Then it computes the four crop ints and calls `obs_source_update()` **only if they changed** since the
last push (`lastCrop*`). At slow speeds most frames round to the same ints, so churn stays low.

The inference branch keeps the initialisation case (`trackingRect.width == 0` → centre on target) and
otherwise **stops easing** — it only stores the target. This keeps the glide in a single place.

### 4.3 Factor rescaling — why the sliders do not need retuning

Today a factor `f` is applied once per `n` frames (`n` = the real inference cadence). After `n` frames
the error remaining is `(1 − f)`. Applying `f′` every frame for `n` frames leaves `(1 − f′)^n`. Equating:

```
f′ = 1 − (1 − f)^(1/n)        n = (detectInterval > 1 && !maskingEnabled) ? detectInterval : 1
```

Same convergence per detection interval, delivered as `n` small steps instead of one. At `n = 1` this
is **bit-identical to 0.0.21**, so the change is regression-safe by construction. `n` must mirror the
render gate's condition exactly (masking bypasses frame-skip, so inference is every frame there).
Applied to all three: `posF`, `sizeInF`, and the capped `sizeOutF`.

### 4.4 Feature flag — the live rollback lever

`smooth_every_frame` (bool, default **true**). When false the glide is invoked only on inference frames
with `n = 1`, which is 0.0.21 behaviour. Settable over obs-websocket (`obsctl setkv <cam>
smooth_every_frame false`) — so if anything looks wrong on stage it is reverted in seconds **without
reinstalling**. Not exposed in the OBS UI (like `locked_track_id`); a panel toggle can follow.

### 4.5 Behaviour on the edges

| case | today | after |
|---|---|---|
| source disabled / no model | tick returns early, crop holds | same (glide not reached) |
| before first detection | crop untouched | same (`targetValid` false) |
| target lost beyond `max_unseen` | ease toward last box at ×0.2, once per inference | same speed, smooth |
| target switch (stickiness handover) | crop jumps to the new target in one eased step per inference | glides there — an improvement |
| masking on | frame-skip already bypassed | `n = 1`, factors unchanged |
| preview on | full-res readback, frame-skip still applies | glide unaffected |
| readback size changes | — | `rawW/rawH` refresh at next inference; clamp uses stored dims |
| director lock | target selection only | unaffected |

### 4.6 Diagnostics

The throttled `[detect-diag]` line (line 1484, every 600 iterations) must count **inference** frames,
not ticks, so its cadence stays ~every 7 s per camera at interval 2. Add the glide's per-frame factors
to that line once, so a soak log shows the rescaling actually in force.

### 4.7 Cost

`obs_source_update()` on the Crop/Pad filter parses four ints — microseconds. Worst case 30 pushes/s
per camera, 120/s total, typically far fewer thanks to change detection. Render-time impact should be
unmeasurable; §6 checks it anyway.

## 5. Exact touch points

* `src/FilterData.h` — add the §4.1 members after `inputFresh` (line 36).
* `src/detect-filter.cpp`
  * defaults (~481): `obs_data_set_default_bool(settings, "smooth_every_frame", true)`; drop the
    "every-frame is what makes panning smooth" comment (no longer true) and say why.
  * `update()` (~512): read `smoothEveryFrame`; reset `targetValid` on model reload.
  * `video_tick()` (855): call `apply_crop_glide(tf)` before the `inputFresh` return.
  * inference branch (1391–1449): store target; keep init case; remove the easing lines.
  * factor out 1451–1479 (clamp + scale-out + push) into `apply_crop_glide`; add change detection.
  * diag (1483–1493): count inference frames; log `n` and the per-frame factors.
* `buildspec.json` — version `0.0.21` → `0.0.22`.
* `C:\Users\Public\do_install_22.ps1` — copy of `do_install_19.ps1` with the staged path and the
  `FileVersion == '0.0.22'` check.

Estimated diff: ~90 lines changed/added. No new dependencies. Passes the CI clang-format gate.

## 6. Acceptance criteria (all measurable with existing scripts)

1. **Crop update rate** (`croprate.py`, high-speed inset poll): ≥ 28 changes/s at interval 2 **and** at
   interval 3 (today 14.5 / 9.7). Median step ≈ today's ÷ n (≈ 4 px at interval 2).
2. **Render budget** (`GetStats` every 15 s for 30 min, cams live, interval 2): `activeFps` 30.0,
   average render time within +1 ms of 0.0.21 (~30 ms), render + output skipped frames ≤ 5 total.
3. **Motion quality** (`hfreq.py`, 60 s): pan reversals ≤ 0.0.21's (≤ 15/min); zoom step median ≤ today's.
4. **Detection unchanged**: tracking hit-rate per camera (tracks_json polls) equal to 0.0.21's.
5. **Flag-off equivalence**: with `smooth_every_frame=false`, criterion 1 returns to ~15/s at interval 2.
6. **Interval-1 equivalence**: at interval 1 the numbers match 0.0.21's (the math is identical there).
7. **Visual sign-off** by Jose on the multiview during real movement.
8. No new `LOG_ERROR`/`LOG_WARNING` lines; `[detect-diag]` cadence unchanged.

## 7. Test & soak protocol (the rig-soak rule applies — no unvalidated build on the stage PC)

1. Build via CI (push to `master` runs check-format + build-project; fetch the `windows-x64-<sha>`
   artifact through the REST API — `gh run view --json artifacts` returns empty for this repo).
2. Stage as `C:\Users\Public\obs-detect-0.0.22-stage.dll`. **Do not install during a shoot**: the
   installer requires OBS closed. Window: right after wrap, or before the next shoot starts.
3. `do_install_22.ps1` — backs up the live DLL as `obs-detect.dll.0.0.21.bak` (instant rollback).
4. Start OBS (via the launcher — autostart brings the panel server up), confirm all four cams live.
5. Run `soak022.py` (to be written from today's scripts): `GetStats` every 15 s for 30 min; `croprate`
   at 5/15/25 min; `hfreq` for 60 s at 10 and 20 min; tracking hit-rate every 5 min; flag off→on once
   at 12 min to record the equivalence. Someone walks the volume for at least 10 of those minutes
   (or play back a HyperDeck recording into the cards).
6. Review the report against §6. Any failure → `smooth_every_frame=false` (seconds), or `.bak` rollback
   (a minute, OBS closed).
7. Only then: File → Exit + `backup_obs.bat` so the flag default and any tuning persist.

The DLL lives in `Program Files`, so one install covers both Windows accounts; the flag default needs
no per-account config.

## 8. Rollout & rollback summary

* **Rollout:** bump → commit → CI → stage → install (OBS closed) → soak → go/no-go before a shoot.
* **Rollback A (seconds, live):** `smooth_every_frame=false` on the four Detect filters.
* **Rollback B (a minute, OBS closed):** copy `obs-detect.dll.0.0.21.bak` back; the installer keeps it.

## 9. Effort

| step | time |
|---|---|
| implementation + clang-format + comments | 1.5–2 h |
| CI build + artifact fetch | ~20 min |
| stage + install (OBS closed) | 10 min |
| soak + review | 45 min |

About half a day, mostly waiting. Only the install step touches the stage PC, and only with OBS closed.

## 10. Risks and how each is closed

| risk | mitigation |
|---|---|
| the feel changes (pan too fast/slow) | §4.3 rescaling keeps per-interval convergence; interval 1 is bit-identical; flag off = 0.0.21 |
| extra `obs_source_update` load | change detection; §6.2 measures render time; expected unmeasurable |
| `n` mismatched to the real cadence (masking/preview) | `n` uses the gate's exact condition; unit-checked in the soak at interval 1/2/3 |
| a bad build reaches the stage | soak rule: staged file, version check, `.bak`, flag |
| readback-vs-source scaling mistakes | the scale-out code is moved, not rewritten; `[detect-diag]` prints both spaces |

## 11. Out of scope (follow-ups, not blockers)

* **Staggered or batched inference across cameras** — the real headroom fix (GPU is at 35 %); would make
  interval 2 cheap and maybe interval 1 viable. Separate change, separate soak.
* **dt-based easing** using `video_tick`'s `seconds` — would make the feel frame-rate-independent; kept
  out so slider semantics stay stable for now.
* **Panel toggle** for `smooth_every_frame` — websocket-only in 0.0.22.

## 12. Decisions needed from Jose

1. Default `smooth_every_frame` **on**? (recommended — it is the point of the release; the flag is the safety net)
2. Install window: after today's wrap, or before the next shoot?
3. Go for implementation?
