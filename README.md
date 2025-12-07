<h1 align="center">LatentSync</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.09262)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/ByteDance/LatentSync-1.6)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/fffiloni/LatentSync)
<a href="https://replicate.com/lucataco/latentsync"><img src="https://replicate.com/lucataco/latentsync/badge" alt="Replicate"></a>

</div>

## 📖 Introduction

We present *LatentSync*, an end-to-end lip-sync method based on audio-conditioned latent diffusion models without any intermediate motion representation, diverging from previous diffusion-based lip-sync methods based on pixel-space diffusion or two-stage generation. Our framework can leverage the powerful capabilities of Stable Diffusion to directly model complex audio-visual correlations.

## 🔧 Setting up the Environment

Install the required packages and download the checkpoints via:

```bash
pip install -r requirements.txt
huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints
huggingface-cli download ByteDance/LatentSync-1.6 tiny.pt --local-dir checkpoints/whisper
```

## 🚀 Inference API

To start the inference server, run the following command:

```bash
python app.py
```

The server will start on `http://0.0.0.0:5000`. You can then send a POST request to the `/generate` endpoint with a video and audio file to perform lip-syncing.

### Request

- **Method:** `POST`
- **Endpoint:** `/generate`
- **Form Data:**
  - `video`: The video file to process.
  - `audio`: The audio file to use for lip-syncing.
  - `guidance_scale` (optional): The guidance scale for the diffusion model. Default is `2.0`.
  - `inference_steps` (optional): The number of inference steps. Default is `20`.
  - `seed` (optional): The random seed for the generation. Default is a random integer.

### Example

```bash
curl -X POST -F "video=@/path/to/video.mp4" -F "audio=@/path/to/audio.wav" http://0.0.0.0:5000/generate > output.mp4
```

## 🙏 Acknowledgement

- Our code is built on [AnimateDiff](https://github.com/guoyww/AnimateDiff). 
- Some code are borrowed from [MuseTalk](https://github.com/TMElyralab/MuseTalk), [StyleSync](https://github.com/guanjz20/StyleSync), [SyncNet](https://github.com/joonson/syncnet_python), [Wav2Lip](https://github.com/Rudrabha/Wav2Lip).

Thanks for their generous contributions to the open-source community!

## 📖 Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@article{li2024latentsync,
  title={LatentSync: Taming Audio-Conditioned Latent Diffusion Models for Lip Sync with SyncNet Supervision},
  author={Li, Chunyu and Zhang, Chao and Xu, Weikai and Lin, Jingyu and Xie, Jinghui and Feng, Weiguo and Peng, Bingyue and Chen, Cunjian and Xing, Weiwei},
  journal={arXiv preprint arXiv:2412.09262},
  year={2024}
}
```
