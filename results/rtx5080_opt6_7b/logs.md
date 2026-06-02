# RTX 5080 OPT-6.7B Run Log

## Model Preparation

Metadata and tokenizer files were already present under `/data/models/opt-6.7b`.
The sharded PyTorch weights were downloaded from `facebook/opt-6.7b`:

```text
pytorch_model-00001-of-00002.bin: 9960750957 bytes
pytorch_model-00002-of-00002.bin: 3356360185 bytes
pytorch_model.bin.index.json: 41937 bytes
```

The first snapshot download produced a corrupt first shard with a zero-filled file header. It was moved aside as:

```text
/data/models/opt-6.7b/pytorch_model-00001-of-00002.bin.corrupt
```

The first shard was then replaced by direct download:

```sh
curl -L --fail --progress-bar \
  -o /data/models/opt-6.7b/pytorch_model-00001-of-00002.bin \
  https://huggingface.co/facebook/opt-6.7b/resolve/main/pytorch_model-00001-of-00002.bin
```

After replacement, both active shards had a ZIP checkpoint header.

## Tiny CUDA Gate

```sh
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.zeros(1, device="cuda"))
PY
```

Output:

```text
True
NVIDIA GeForce RTX 5080
tensor([0.], device='cuda:0')
```

## Original Evaluation

Baseline without allocator tuning failed:

```text
Failure: CUDA OOM during first loss calculation
Wall time: 11 seconds
Peak VRAM: 15662 MiB
```

The successful run used:

```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval_perplexity.py \
  -m opt \
  -s 6.7b \
  -t wikitext \
  --quant-method none \
  --gpu-start-idx 0 \
  --gpu-count 1
```

Observed:

```text
Original PPL: tensor(10.8594, device='cuda:0', dtype=torch.float16)
Wall time: 47 seconds
Peak VRAM: 15806 MiB
```

## Oaken Offline Profiling

```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python oaken_preprocess_activation.py \
  -m opt \
  -s 6.7b \
  -t wikitext \
  -f 0.04 0.9 0.06 \
  -o quantizer/oaken/opt-6.7b.json \
  --gpu-start-idx 0 \
  --gpu-count 1
```

Observed:

```text
Profiling pass PPL: tensor(10.8594, device='cuda:0', dtype=torch.float16)
Elapsed time: 108.85710959299831 seconds
Wall time: 113 seconds
Peak VRAM: 15806 MiB
```

## Oaken Evaluation

```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval_perplexity.py \
  -m opt \
  -s 6.7b \
  -t wikitext \
  -q quantizer/oaken/opt-6.7b.json \
  --quant-method oaken \
  --gpu-start-idx 0 \
  --gpu-count 1
```

Observed:

```text
Failure: CUDA OOM during loss calculation after 6/141 Wikitext chunks
Wall time before failure: 12 seconds
Peak VRAM: 15826 MiB
```
