# Cross-modal Feature Graph Modeling for CT VQA

Implementation of **Cross-modal Feature Graph Modeling for Computed Tomography Visual Question Answering**, accepted at **CHIP 2026**.

The model encodes CT slices and question tokens, applies bilinear attention over a cross-modal graph, and projects graph nodes into soft prompts for Qwen2-VL (2B). Training uses answer-token cross-entropy and updates the participating encoders, graph, projection, and language decoder. Prediction uses greedy decoding.

## Installation

Use Python 3.12 and install `requirements.txt` in a dedicated environment.

```bash
python -m pip install -r requirements.txt
```

## Data and models

Obtain M3D-VQA from the [official M3D project](https://github.com/BAAI-DCAI/M3D#data), follow its data preparation instructions, and retain its separate train, validation, and test CSV files. The loader uses the original `Image Path`, `Question`, `Answer`, and `Question Type` columns. Image paths resolve under `--data-root` to normalized NumPy volumes shaped `(1, D, H, W)` in `[0, 1]`. The entry point uses open-ended questions and answers and preserves all rows of the selected split.

Provide local or Hugging Face paths for the vision, text, and Qwen2-VL checkpoints in the configuration. The encoder interface uses ViT patch states and BERT token states of equal hidden width. The [official Qwen2-VL collection](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) provides checkpoint access and model usage terms. Set the exact checkpoint variant and revision explicitly.

## Configuration

Pass a JSON configuration with these required fields:

- `vision_checkpoint`, `text_checkpoint`, `language_checkpoint`, and their `vision_revision`, `text_revision`, `language_revision` values.
- `graph_layers`, `dtype` (`float32` or `bfloat16`), `seed`.
- `graph_special_tokens`, `question_special_tokens`, `answer_eos`: explicit booleans controlling token boundaries. Question strings are encoded directly.
- `preprocessing`: `slice_stride`, `image_size`, `resize_mode` (`nearest`, `bilinear`, or `bicubic`), and three-element `mean` and `std` lists. Slices retain depth order; grayscale is repeated across three channels. Bilinear and bicubic resizing use `align_corners=False`.
- For training, `training`: `optimizer` (`Adam` or `AdamW`), `learning_rate`, two-element `betas`, `epsilon`, `weight_decay`, `batch_size`, `epochs`, and `schedule` (`constant`). Variable-size examples accumulate gradients sequentially within each batch; answer-token loss is averaged per example, then across the batch. Each epoch saves model weights and configuration.
- For prediction, `max_new_tokens` and `eos_ids`. Use the same complete configuration used for training.

Choose values from the experiment configuration. All scientific choices are explicit inputs. The model preserves every graph node as a soft-prompt token. Its graph has bidirectional neighboring-slice edges and all slice-token edges, with no self-loops or question-question edges. Each layer implements the paper's attention-weighted affine aggregation. The LLM receives graph tokens, question tokens, and teacher-forced answer tokens in that order. Position indices run consecutively across this sequence. Generation ends at a configured EOS token or length limit.

## Run

Run commands from this directory. Store data, checkpoints, and outputs outside the source directory.

```bash
python run.py train --config experiment.json --csv data/train.csv --data-root data --output outputs/model.pt --device cuda
python run.py predict --config experiment.json --csv data/test.csv --data-root data --checkpoint outputs/model.pt --output outputs/predictions.jsonl --device cuda
```

Predictions retain the row index, clinical category, question, gold answer, and generated answer for evaluation. `--checkpoint` loads model weights; training creates a fresh optimizer.

## Structure

- `model.py`: slice/token encoding, graph construction, attentive graph layers, answer loss, and greedy decoding.
- `data.py`: M3D-VQA CSV and CT volume loading.
- `run.py`: training and prediction entry point.

## Citation
