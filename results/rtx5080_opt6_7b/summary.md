# RTX 5080 OPT-6.7B Oaken Boundary Result

## Result

| Metric | Value |
| --- | ---: |
| Original Wikitext PPL | 10.8594 |
| Oaken Wikitext PPL | OOM |
| PPL delta | N/A |
| Original eval wall time | 47s |
| Original eval peak VRAM | 15806 MiB |
| Profiling time | 108.86s |
| Profiling wall time | 113s |
| Profiling peak VRAM | 15806 MiB |
| Oaken eval wall time before OOM | 12s |
| Oaken eval peak VRAM | 15826 MiB |
| Idle VRAM after runs | 38 MiB |
| Status | Boundary: original eval and profiling OK; Oaken eval OOM |

## Artifacts

- `oaken-quantizer.json`: copied from `quantizer/oaken/opt-6.7b.json`.
- `original.log`: original FP16 Wikitext perplexity run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- `profile.log`: Oaken offline profiling run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- `eval-oom.log`: Oaken Wikitext perplexity run that failed with CUDA OOM.
- `original-no-expandable-oom.log`: baseline original eval attempt without allocator tuning; it failed with CUDA OOM.
- `*-vram.csv`: 1s `nvidia-smi` samples for each run.
- `hardware.txt`: host and CUDA environment snapshot.

## Notes

Before each major CUDA run, the tiny CUDA allocation gate passed:

```text
True
NVIDIA GeForce RTX 5080
tensor([0.], device='cuda:0')
```

The initial OPT-6.7B download produced a corrupt first shard whose first bytes were zero-filled. The shard was preserved as `/data/models/opt-6.7b/pytorch_model-00001-of-00002.bin.corrupt` and replaced by a direct download from the official Hugging Face resolve URL. Model weights and Hugging Face cache files are not part of this result artifact.

Without `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, original FP16 eval loaded the checkpoint but failed at the first loss calculation with CUDA OOM at 15662 MiB peak VRAM. With the allocator setting, original eval and profiling completed, but Oaken eval still failed near the 16GB boundary.
