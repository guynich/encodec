Neural codecs are encoder–decoder models that convert audio into compact numeric
representations (e.g.: quantized latent tokens or continuous latents) which can
be ingested and produced by large language models (LLMs). By letting an LLM
predict sequences of these audio tokens, the model can generate or reconstruct
audible sound.

This script plays the original and reconstructed audio for a chosen bandwidth
using [`EnCodec`](https://github.com/facebookresearch/encodec).

- [1. Setup](#1-setup)
- [2. Run script](#2-run-script)
- [Optional clean-up](#optional-clean-up)
- [References](#references)

## 1. Setup

These steps tested on:

> * MacBook Air M3 (16GB)
> * macOS 15.7
> * Python 3.12.6
>
> I used `brew` on macOS to install `git` and `pyenv`.

1. Clone this repo.
```bash
git clone git@github.com:guynich/encodec.git
```

2. Install dependencies

Create a Python environment and activate it.
```bash
cd
python3 -m venv .venv_encodec
source ./.venv_encodec/bin/activate
```

Install package requirements.
```bash
python3 -m pip install --upgrade pip
cd encodec
python3 -m pip install -r requirements.txt
```

3. Build FFMPEG for torchcodec

The Pytorch package `torchcodec` installed in the last step supports only
versions 4-7 of FFMPEG.  On macOS there is
[reported incompatibility](https://github.com/pytorch/torchcodec/issues/570)
with `brew` installed FFMPEG versions.  In this section we'll build FFMPEG
version 7.1.2.

Install build tools using `brew`.
```bash
brew update
brew install autoconf cmake
```
Clone the FFmpeg repo (official upstream).
```bash
cd
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg
```
Checkout version-7.1.2 branch (or any version supported by PyTorch `torchcodec`).
```bash
git checkout n7.1.2
```
Configure minimal build to produce `.dylib` shared libraries.
```bash
./configure \
  --prefix=/usr/local/ffmpeg7 \
  --enable-shared \
  --disable-static \
  --enable-pthreads
```
Build.
```bash
make -j$(nproc)
```
Install.
```bash
sudo make install
```
The shared `.dylib` library needed by PyTorch `torchcodec` is in this folder.
```bash
ls /usr/local/ffmpeg7/lib | grep libavutil
```

## 2. Run script

Run in the virtual environment and share the library path.
```bash
source ~/.venv_encodec/bin/activate

export DYLD_FALLBACK_LIBRARY_PATH=/usr/local/ffmpeg7/lib

cd ~/encodec
python3 main.py
```

Example console.
```console
$ python3 main.py
Fetching 1 files: 100%|█████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2589.08it/s]
Fetching 1 files: 100%|█████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2165.36it/s]
Fetching 1 files: 100%|█████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1883.39it/s]
Playing original audio ...
Playing reconstructed audio ...

Bandwith:              6 kbps
Input values shape:    torch.Size([1, 1, 140520]) (batch_size, channels, samples)
Encoder outputs shape: torch.Size([1, 1, 8, 440]) (batch_size, channels, number_of_codebooks, frames)
Reconstructed shape:   torch.Size([1, 1, 140520]) (batch_size, channels, samples)
```

## Optional clean-up
Remove cloned repos, virtual environment and FFMPEG files.
```bash
cd
rm -rf encodec
rm -rf ffmpeg

deactivate
rm -rf .venv_encodec

sudo rm -rf /usr/local/ffmpeg7
```

## References

The code in `main.py` script is adapted from the EnCodec repo example.

* EnCodec repo: https://github.com/facebookresearch/encodec
* FFMPEG repo: https://github.com/FFmpeg/FFmpeg
