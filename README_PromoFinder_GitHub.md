# PromoFinder AI — Kuku AI Generalist, Problem 1

A rules-first prototype that finds and ranks candidate 20–60 second promotional segments in timestamped microdrama screenplays.

## What it does

1. Finds `episode-*.txt` files recursively under the Kaggle input directory (or inspects ZIP files if extracted text files are not present).
2. Parses episode metadata, speaker labels, dialogue/actions/sound/camera beats and source timestamps.
3. Builds continuous candidate windows using screenplay event boundaries and enforces the configured 20–60 second duration.
4. Scores candidates using opening hook, cliffhanger, standalone context, dramatic progression and duration fit.
5. Limits overlapping recommendations and retains up to three candidates per episode.
6. Exports candidate CSV/JSON, episode-processing summary, cross-episode montage suggestions, human-label template and run summary.
7. Provides an optional provider-neutral model adapter. The default run is deterministic and does not call an external model.

## Run in Kaggle

1. Create or open a Kaggle Notebook.
2. Add the authorised screenplay input using **Add Input**. The expected file pattern is `episode-*.txt`.
3. Paste the code from `PromoFinder_Any_AI_Kaggle_SingleCell.py` into one code cell, or import the `.ipynb` file.
4. Keep `USE_LLM = False` for the no-API baseline.
5. Run the cell. Results are written to `/kaggle/working/promofinder_results/` and bundled as `/kaggle/working/promofinder_results.zip`.

## Optional model integration

Set `USE_LLM = True` only after configuring a provider endpoint, model ID and API key securely through Kaggle Secrets. The generic adapter supports OpenAI-compatible Chat Completions APIs. Providers with a different API contract can be connected using `MODEL_ADAPTER(prompt, model_name) -> (response_text, usage_dict)`.

## Quality evaluation

The run creates `human_labels.csv` as an annotation template. Independently select a best promo interval for each sampled episode and enter start/end timestamps in `MM:SS.cc` format. Rerun the notebook to calculate top-1 temporal agreement using interval IoU >= 0.50. This measures overlap with human labels, not audience-performance accuracy.

## Known limitations

- Heuristic scores are ranking signals, not probabilities.
- Screenplay text does not verify the final video's acting, framing, sound, pacing or visual continuity.
- Cross-episode montage suggestions are experimental chronological combinations and require editorial review for continuity and spoiler risk.
- The current baseline has not yet been validated against completed human labels or live audience results.

## Confidentiality

This repository contains code and instructions only. Do not upload Kuku's screenplay input pack, screenplay excerpts, raw output files containing transcript excerpts, or other confidential materials to a public repository. Use the assignment submission channel for raw outputs.
