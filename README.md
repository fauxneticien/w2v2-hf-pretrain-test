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
