# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a frame interpolation service built on Google's FILM (Frame Interpolation for Large Motion) model. It generates a middle frame between two input images using a TensorFlow 2 saved model.

The project has two execution modes:
1. **Worker service** (`worker.py`): AWS-integrated long-running worker that polls SQS, downloads frames from S3, runs interpolation, uploads results to S3, and emails the user via SES.
2. **Cog predictor** (`frame-interpolation/predict.py`): Replicate.com-compatible predictor for on-demand inference.

## Running the Worker

Copy `.env.example` to `.env` and fill in your AWS values, then:

```bash
pip install -r requirements_worker.txt tensorflow numpy
python worker.py
```

The worker requires these environment variables (see `.env.example`):
- `AWS_REGION`, `SQS_QUEUE_URL`, `S3_BUCKET`, `MODEL_PATH`, `SES_SENDER_EMAIL`
- `PRESIGNED_URL_EXPIRY_SECONDS` (optional, defaults to 604800 / 7 days)

## Model

Three pretrained FILM model variants are in `pretrained_models/film_net/`:
- `Style/saved_model` — default used by both the worker and Cog predictor
- `L1/saved_model`
- `VGG/saved_model` (requires `pretrained_models/vgg/imagenet-vgg-verydeep-19.mat`)

`MODEL_PATH` in the worker should point to one of these `saved_model` directories.

## Architecture

### `worker.py`
- `InterpolationWorker` loads the FILM model at startup via `Interpolator` from `frame-interpolation/eval/interpolator.py`
- Long-polls SQS with 20-second wait; processes one message at a time
- Message schema: `{ JobId, Email, StartFrameKey, EndFrameKey }` (S3 object keys)
- On failure, leaves the message in the queue for SQS retry (no explicit DLQ handling in code)

### `frame-interpolation/eval/interpolator.py`
- `Interpolator` wraps the TF2 saved model
- Input images are `np.float32` in `[0, 1]` range with shape `(B, H, W, C)`
- `align=64` pads images so dimensions are divisible by 64 before inference
- `block_shape=[1, 1]` means no patch subdivision (used in worker); set higher values to process large images in tiles

### `frame-interpolation/eval/util.py`
- `interpolate_recursively_from_files` — recursive bisection to produce `2^n + 1` frames between two images

## Key Dependency

The `frame-interpolation/` directory is Google's open-source FILM repo (Apache 2.0). `worker.py` inserts it into `sys.path` at runtime and imports `eval.interpolator.Interpolator` from it directly — there is no pip-installable package.
