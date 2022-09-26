# w2v2-hf-pretrain-test
Testing wav2vec 2.0 pre-training with HuggingFace

## Set up

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
```

## Stage 1 (Sept 27, 2022):

Attempt to set up correct enviroment and replicate demo run from:
https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-pretraining#demo
