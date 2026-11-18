# PromoFinder AI — Microdrama Promo Discovery

## Project Overview

PromoFinder AI is a screenplay intelligence tool developed for Kuku's Problem 1: Find the Promos.

It processes timestamped screenplay files and generates ranked promotional clip recommendations of 20–60 seconds.

## Features

- Automatic screenplay discovery and parsing.
- Dynamic cast extraction.
- Hook and cliffhanger identification.
- Rules-based promotional clip ranking.
- Experimental cross-episode montage suggestions.
- Human evaluation using temporal Intersection over Union (IoU).
- Optional provider-independent LLM integration.

## How to Run on Kaggle

1. Open the supplied Kaggle notebook.
2. Attach the `scripts_dataset` through Add Input.
3. Keep `USE_LLM = False` for the initial rules-based run.
4. Click Run All.
5. Find generated outputs under `/kaggle/working/promofinder_results/`.

## Generated Outputs

- `promo_candidates.csv`
- `cross_episode_montages.csv`
- `episode_processing_summary.csv`
- `human_labels.csv`
- `human_evaluation.json`
- `run_summary.json`

## Baseline Execution

Tested on 50 supplied episodes:

- 50 episodes processed.
- 3,727 timestamped beats extracted.
- 150 promotional candidates generated.
- 10 experimental montage suggestions.

The measured runtime was 10.270 seconds in one Kaggle execution. Runtime depends on the environment.

The initial run uses deterministic rules and does not make external AI API calls.

## Limitations

The scores are heuristic ranking signals, not probabilities. Recommendations require editorial review. Human quality evaluation must be completed using independently selected timestamps.

## Confidentiality

The original screenplay dataset and screenplay-derived raw outputs are not included in this public repository.
