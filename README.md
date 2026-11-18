# PromoFinder AI — Kuku AI Generalist, Problem 1

A rules-first prototype that finds and ranks candidate 20–60-second promotional segments in timestamped microdrama screenplays.

## What it does

1. Finds `episode-*.txt` files recursively under the Kaggle input directory. If extracted text files are not present, it inspects ZIP files for matching episode files.
2. Parses episode metadata, speaker labels, dialogue, actions, sound and camera beats, and source timestamps.
3. Builds continuous candidate windows from screenplay event boundaries and enforces the configured 20–60-second duration range.
4. Scores candidates using five signals: opening hook, cliffhanger strength, standalone context, dramatic progression and duration fit.
5. Limits overlapping recommendations and retains up to three candidates per episode.
6. Exports ranked candidate CSV, episode-processing summary, cross-episode montage suggestions, a human-label template, human-evaluation JSON and a run summary.
7. Provides an optional provider-neutral model adapter. The default run is deterministic and does not call an external model.

## Run in Kaggle

1. Create or open a Kaggle Notebook.
2. Add the authorised screenplay input using **Add Input**. The expected file pattern is `episode-*.txt`.
3. Paste the code from `PromoFinder_Any_AI_Kaggle_SingleCell.py` into one code cell, or import and run `PromoFinder_Any_AI_Kaggle_SingleCell.ipynb`.
4. Keep `USE_LLM = False` for the no-API baseline.
5. Run the cell. Results are written to `/kaggle/working/promofinder_results/` and bundled as `/kaggle/working/promofinder_results.zip`.

## Optional model integration

Set `USE_LLM = True` only after configuring a provider endpoint, model ID and API key securely through Kaggle Secrets. The generic adapter supports OpenAI-compatible Chat Completions APIs. Providers with a different API contract can be connected using `MODEL_ADAPTER(prompt, model_name) -> (response_text, usage_dict)`.

## Baseline execution results

The prototype was run in Kaggle on the supplied screenplay dataset. The recorded run produced:

| Metric | Result |
|---|---:|
| Episodes processed | 50 |
| Timestamped beats extracted | 3,727 |
| Promo candidates generated | 150 |
| Cross-episode montage suggestions | 10 |
| Measured execution time | 10.270 seconds |
| External AI API calls | 0 |

The execution time is from one Kaggle session and may vary with the runtime environment. The 150 candidates and 10 montage suggestions are automated recommendations, not editor-approved promotional assets.

## Quality evaluation

The run creates `human_labels.csv` as an annotation template. Independently select a preferred promo interval for sampled episodes and enter start/end timestamps in `MM:SS.cc` format. Rerun the notebook to calculate top-1 temporal agreement using interval IoU ≥ 0.50.

This measures overlap with the supplied human labels; it is not audience-performance accuracy. **Human quality agreement has not yet been measured in the reported baseline run**, because the human-label template was still unfilled.

## Known limitations

- Heuristic scores are ranking signals, not probabilities or validated measures of promo quality.
- The scoring weights are initial assumptions and should be calibrated using editor feedback and human-labelled examples.
- Screenplay text does not verify the final video's acting, framing, sound, pacing or visual continuity.
- Cross-episode montage suggestions are experimental chronological combinations and require editorial review for narrative continuity and spoiler risk.
- The baseline has not yet been validated against completed human labels or live audience results.
- The baseline execution did not call an external AI model; the model adapter is optional and must be configured before use.

## Confidentiality

This public repository contains code and instructions only. Do not upload Kuku's screenplay input pack, screenplay excerpts, raw output files containing transcript excerpts, API keys or other confidential materials to a public repository. Submit raw outputs separately through the authorised assignment submission channel.
