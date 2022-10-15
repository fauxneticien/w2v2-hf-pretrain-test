# w2v2-hf-pretrain-test
Testing wav2vec 2.0 pre-training with HuggingFace

## Set up

## Environment

```
pip install -r requirements.txt
```

## GitHub (only if you intend to push changes back up to GitHub repo)

```bash
# Install sudo (Jarvis Debian doesn't come with sudo)
apt-get update && apt-get install -y sudo

# Install GitHub cli (so you can push changes made from Jarvis instance)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y

# Authenticate with personal access token for pushing changes
gh auth login

# Clone repo using GitHub CLI
gh repo clone fauxneticien/w2v2-hf-pretrain-test

# Set up credentials
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```

## Stage 1 (Sept 27, 2022):

Attempt to set up correct enviroment and replicate demo run from:
https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-pretraining#demo

Run command (note: config set for 1-GPU instance with A6000, 48 GB VRAM):

```bash
accelerate launch run_wav2vec2_pretraining_no_trainer.py \
	--dataset_name="librispeech_asr" \
	--dataset_config_names clean clean \
	--dataset_split_names validation test \
	--model_name_or_path="patrickvonplaten/wav2vec2-base-v2" \
	--output_dir="./wav2vec2-pretrained-demo" \
	--max_train_steps="20000" \
	--num_warmup_steps="32000" \
	--gradient_accumulation_steps="3" \
	--learning_rate="0.005" \
	--weight_decay="0.01" \
	--max_duration_in_seconds="20.0" \
	--min_duration_in_seconds="2.0" \
	--logging_steps="1" \
	--saving_steps="10000" \
	--per_device_train_batch_size="20" \
	--per_device_eval_batch_size="20" \
	--adam_beta1="0.9" \
	--adam_beta2="0.98" \
	--adam_epsilon="1e-06" \
	--gradient_checkpointing
```

W&B link: https://wandb.ai/fauxneticien/w2v2-pretrain/runs/2n8n06z5?workspace=user-fauxneticien

## Stage 2 (Octobert 3, 2022):

- Swap out `librispeech_asr` dataset with dataset of interest using HF Datasets library 'audiofolder' feature: https://huggingface.co/docs/datasets/audio_dataset#audiofolder (create dataset from a bunch of audio files in folder). 
- Use 2.7 hours of Nasal audio for pre-training (split into 90/10 train/test using `--validation_split_percentage="10"`)

```bash
accelerate launch run_wav2vec2_pretraining_no_trainer_audiofolder.py \
	--data_dir="nasal" \
	--validation_split_percentage="10" \
	--model_name_or_path="patrickvonplaten/wav2vec2-base-v2" \
	--output_dir="./wav2vec2-nasal2.7h" \
	--max_train_steps="150000" \
	--num_warmup_steps="32000" \
	--gradient_accumulation_steps="3" \
	--learning_rate="0.005" \
	--weight_decay="0.01" \
	--max_duration_in_seconds="10.0" \
	--min_duration_in_seconds="2.0" \
	--logging_steps="1" \
	--saving_steps="10000" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--adam_beta1="0.9" \
	--adam_beta2="0.98" \
	--adam_epsilon="1e-06" \
	--gradient_checkpointing
```

## Stage 3 (October 14, 2022):

- Change dataset to 4.5 hour dataset (instead of 2.7 hour dataset)
	- Minimum wav duration is now 1.5s and maximum wav duration is 2.5s
	- Smaller range around the median of wav durations allows us to 1) use as much data as possible and 2) use a bigger batch size without worrying about accidental OOM issues (i.e. 1 very long wav file causes batch to be bigger than what GPU memory can fit)

- WHOOPS: turns out the old code was randomly initializing a new model (updated code to load pre-trained weights):

	```python
	# initialize random model
    # model = Wav2Vec2ForPreTraining(config)

    # load pre-trained model
    model = Wav2Vec2ForPreTraining.from_pretrained(args.model_name_or_path)
	```	

	- But the way wav2vec 2 models are pre-trained has changed since the original paper so not all checkpoints can be continued to be pre-trained (at least using HF code, see https://discuss.huggingface.co/t/pretrain-facebook-wav2vec2-base/14121/3?u=fauxneticien and https://github.com/facebookresearch/fairseq/issues/3277)

- Try continue pre-training from one of the latest official model checkpoints (XLS-R 300m) using model weights from https://huggingface.co/facebook/wav2vec2-xls-r-300m and pre-training config from https://huggingface.co/patrickvonplaten/wav2vec2-large-repro-960h-libri-120k-steps

```bash
accelerate launch run_wav2vec2_pretraining_no_trainer_audiofolder.py \
	--data_dir="20221014_nasal/data" \
	--validation_split_percentage="10" \
	--model_name_or_path="xls-r_300m" \
	--output_dir="./xls-r0.3b_nasal4.5h" \
	--max_train_steps="75000" \
	--num_warmup_steps="32000" \
	--gradient_accumulation_steps="2" \
	--learning_rate="0.0005" \
	--weight_decay="0.01" \
	--max_duration_in_seconds="3" \
	--min_duration_in_seconds="1" \
	--logging_steps="1" \
	--saving_steps="10000" \
	--per_device_train_batch_size="64" \
	--per_device_eval_batch_size="64" \
	--adam_beta1="0.9" \
	--adam_beta2="0.98" \
	--adam_epsilon="1e-06" \
	--gradient_checkpointing
```
