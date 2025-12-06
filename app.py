import os
import tempfile
from flask import Flask, request, jsonify, send_file
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DContitionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from io import BytesIO

app = Flask(__name__)

# Global variables for the model
pipeline = None
config = None
dtype = torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7 else torch.float32

def load_model():
    global pipeline, config

    config = OmegaConf.load("configs/unet/stage2.yaml")
    inference_ckpt_path = "checkpoints/latentsync_unet.pt"

    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DContitionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        inference_ckpt_path,
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

@app.route('/generate', methods=['POST'])
def generate():
    if 'video' not in request.files or 'audio' not in request.files:
        return jsonify({'error': 'video and audio files are required'}), 400

    video_file = request.files['video']
    audio_file = request.files['audio']

    video_path = None
    audio_path = None
    output_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            video_file.save(temp_video_file.name)
            video_path = temp_video_file.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            audio_file.save(temp_audio_file.name)
            audio_path = temp_audio_file.name

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        guidance_scale = float(request.form.get('guidance_scale', 2.0))
        inference_steps = int(request.form.get('inference_steps', 20))
        seed = int(request.form.get('seed', 0))

        if seed <= 0:
            seed = int.from_bytes(os.urandom(2), "big")
        set_seed(seed)

        pipeline(
            video_path=video_path,
            audio_path=audio_path,
            video_out_path=output_path,
            num_frames=config.data.num_frames,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
            mask_image_path=config.data.mask_image_path,
            temp_dir="temp",
        )

        buffer = BytesIO()
        with open(output_path, 'rb') as f:
            buffer.write(f.read())
        buffer.seek(0)

        return send_file(buffer, mimetype='video/mp4', as_attachment=True, download_name='output.mp4')

    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        if output_path and os.path.exists(output_path):
            os.remove(output_path)


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000)