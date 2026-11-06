#!/usr/bin/env bash
# Merge the adapter, convert to GGUF, quantize to Q4_K_M and upload it to the Hub.
# Run from anywhere. Needs HF_USERNAME and HF_TOKEN. The upload makes the GGUF public.
set -euo pipefail
cd "$(dirname "$0")/.."

setting() { python -c "import config; print(config.$1)"; }
export_dir="$(setting EXPORT_DIR)"
merged_dir="$(setting MERGED_DIR)"
gguf_path="$export_dir/$(setting GGUF_FILENAME)"
f16_path="$export_dir/commitcraft-f16.gguf"
llama_dir="$export_dir/llama.cpp"

mkdir -p "$export_dir"
python -m train.export merge

if [ ! -d "$llama_dir" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$llama_dir"
fi
python "$llama_dir/convert_hf_to_gguf.py" "$merged_dir" --outtype f16 --outfile "$f16_path"

cmake -S "$llama_dir" -B "$llama_dir/build" -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build "$llama_dir/build" --target llama-quantize -j "$(nproc)"
"$llama_dir/build/bin/llama-quantize" "$f16_path" "$gguf_path" Q4_K_M

python -m train.export upload
