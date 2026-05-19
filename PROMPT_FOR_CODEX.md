# Prompt for Codex/GPT

너는 컴퓨터 구조, GPU memory hierarchy, LLM inference system을 아는 시스템 연구 엔지니어다.

목표는 Oaken을 consumer GPU에서 단순 실행하는 것이 아니라, Oaken류 연구가 다루는 원인부인 KV-cache memory pressure를 RTX 5060 8GB / RTX 5080 16GB 같은 consumer GPU에서 직접 계측하는 것이다.

저장소 이름: `kv-cache-consumer-gpu-bench`

연구 질문:
1. causal LM의 KV-cache memory footprint가 이론식과 얼마나 일치하는가?
2. batch size와 context length가 증가할 때 consumer GPU에서 OOM boundary가 어디서 발생하는가?
3. dynamic / quantized / offloaded / no_cache 전략은 latency, tokens/sec, CUDA peak memory에서 어떤 trade-off를 보이는가?
4. quantized cache가 항상 좋은 것이 아니라 어느 context 길이부터 memory-saving이 의미를 갖고, 어느 구간에서는 dequantization overhead 때문에 느려지는지 확인할 수 있는가?

KV-cache 이론식:

```text
KV bytes = 2 × num_layers × batch_size × seq_len × num_key_value_heads × head_dim × bytes_per_element
```

주의:
- GQA/MQA 모델이 있으므로 `num_attention_heads`가 아니라 `num_key_value_heads`를 써야 한다.
- actual KV bytes는 forward 결과의 `past_key_values` 내부 tensor를 순회해서 `numel × element_size`로 계산한다.
- generation benchmark는 Hugging Face `model.generate()`로 한다.
- synthetic repeated token id를 사용해서 `seq_len`을 정확히 맞춘다.
- OOM은 실패가 아니라 결과다. exception을 잡고 CSV에 `status=oom`으로 기록한다.
