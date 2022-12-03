OUTPUT_DIR="models/xls-r_300m"

mkdir -p $OUTPUT_DIR
pushd $OUTPUT_DIR
# Get XLS-R model weights
wget https://huggingface.co/facebook/wav2vec2-xls-r-300m/resolve/main/pytorch_model.bin
# Use model config from w2v2 model known to work with pretraining script
wget https://huggingface.co/patrickvonplaten/wav2vec2-large-repro-960h-libri-120k-steps/raw/main/config.json
wget https://huggingface.co/patrickvonplaten/wav2vec2-large-repro-960h-libri-120k-steps/raw/main/preprocessor_config.json
popd
