from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import time
import traceback
import base64
import json
from functools import cache
from io import BytesIO
from PIL import Image

import spaces
import gradio as gr
from gradio import Server
from fastapi.responses import HTMLResponse

MODEL_REPO = os.environ.get("H3_MODEL_REPO", "MiniMaxAI/MiniMax-H3")
CONDITIONER_SPACE = os.environ.get("H3_CONDITIONER", "multimodalart/qwen3vl-conditioner")
PLACEMENT = os.environ.get("H3_PLACEMENT", "pack").lower()
ATTENTION = os.environ.get("H3_ATTENTION", "_native_cudnn").lower()
GPU_SIZE = os.environ.get("H3_GPU_SIZE", "xlarge")

CANVASES = {
    "960x544 · 16:9 fast": (544, 960),
    "1024x576 · 16:9 fast": (576, 1024),
    "1152x640 · 16:9": (640, 1152),
    "1280x704 · 16:9": (704, 1280),
    "1344x768 · 16:9 full": (768, 1344),
    "544x960 · 9:16 fast": (960, 544),
    "640x1152 · 9:16": (1152, 640),
    "768x1344 · 9:16 full": (1344, 768),
    "544x544 · 1:1 fast": (544, 544),
    "768x768 · 1:1 full": (768, 768),
    "768x576 · 4:3 fast": (576, 768),
    "1024x768 · 4:3 full": (768, 1024),
    "576x768 · 3:4 fast": (768, 576),
    "768x1024 · 3:4 full": (1024, 768),
    "1152x512 · 21:9 fast": (512, 1152),
    "1536x672 · 21:9 full": (672, 1536),
}
DEFAULT_CANVAS = "960x544 · 16:9 fast"

FPS, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 24, 17, 5

MIN_UI_DURATION = float(os.environ.get("H3_MIN_DURATION", "2"))
MAX_UI_DURATION = float(os.environ.get("H3_MAX_DURATION", "20"))
PIPELINE_MAX_DURATION = MAX_UI_DURATION + 1.0
CHUNK_SECONDS = float(os.environ.get("H3_CHUNK_SECONDS", "14"))
CONDITIONER_MAX_FRAMES = int(os.environ.get("H3_CONDITIONER_MAX_FRAMES", "345"))

GPU_DURATION_CHOICES = (60, 90, 120, 150, 180, 250)
DEFAULT_GPU_DURATION = 120

OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "h3-outputs")

PIPE = None
MUTE_PIPE = None
MANAGER = None
LOAD_ERROR: str | None = None
MUTE_ERROR: str | None = None
LOADED_IN: float | None = None
LORA_STATUS: str | None = None


def snap_frames(seconds: float) -> int:
    frames = max(1, round(float(seconds) * FPS))
    while frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        frames += 1
    return frames


def patch_duration_window(minimum: float, maximum: float) -> None:
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
    MiniMaxH3ModularPipeline.min_duration = property(lambda self: float(minimum))
    MiniMaxH3ModularPipeline.max_duration = property(lambda self: float(maximum))


def _apply_lora(transformer) -> str | None:
    try:
        import h3_lora
        return h3_lora.apply_lora(transformer)
    except ImportError:
        return "⚠️ `h3_lora.py` is missing — running without the Turbo LoRA."
    except Exception as error:
        traceback.print_exc()
        return f"⚠️ LoRA loading failed ({type(error).__name__}: {error})"


def _set_lora(transformer, name: str) -> str:
    try:
        import h3_lora
        return h3_lora.set_active(transformer, name)
    except Exception:
        return "off"


def load_models() -> str | None:
    global PIPE, MUTE_PIPE, MANAGER, LOAD_ERROR, MUTE_ERROR, LOADED_IN, LORA_STATUS
    if PIPE is not None or LOAD_ERROR is not None:
        return LOAD_ERROR

    started = time.time()
    try:
        import torch
        from diffusers import ComponentsManager
        from h3_split_blocks import MiniMaxH3GeneratorBlocks

        patch_duration_window(MIN_UI_DURATION, PIPELINE_MAX_DURATION)
        manager = ComponentsManager()
        blocks = MiniMaxH3GeneratorBlocks()
        print(f"[gen] loading {[c.name for c in blocks.expected_components]} from {MODEL_REPO} ...", flush=True)
        pipe = blocks.init_pipeline(MODEL_REPO, components_manager=manager, collection="h3")
        pipe.load_components(dtype=torch.bfloat16)

        LORA_STATUS = _apply_lora(pipe.transformer)
        if LORA_STATUS:
            print(f"[gen] {LORA_STATUS}", flush=True)

        pipe.transformer.set_attention_backend(ATTENTION)

        if PLACEMENT == "pack":
            pipe.transformer.to("cuda")
        if PLACEMENT == "offload":
            manager.enable_auto_cpu_offload(device="cuda")
            _arm_decode_hooks(pipe)

        PIPE, MANAGER = pipe, manager
        MUTE_PIPE, MUTE_ERROR = _build_mute_pipe(pipe)
        LOADED_IN = time.time() - started
        print(f"[gen] ready in {LOADED_IN:.0f}s", flush=True)
    except Exception as error:
        traceback.print_exc()
        LOAD_ERROR = (
            f"**Loading `{MODEL_REPO}` failed** after {time.time() - started:.0f}s: "
            f"`{type(error).__name__}: {error}`"
        )
    return LOAD_ERROR


def _build_mute_pipe(pipe):
    try:
        from h3_split_blocks import MiniMaxH3MuteGeneratorBlocks
        blocks = MiniMaxH3MuteGeneratorBlocks()
        mute = blocks.init_pipeline(MODEL_REPO)
        shared = {}
        for spec in blocks.expected_components:
            component = getattr(pipe, spec.name, None)
            if component is not None:
                shared[spec.name] = component
        mute.update_components(**shared)
        print(f"[gen] video-only pipeline over {sorted(shared)}", flush=True)
        return mute, None
    except Exception as error:
        traceback.print_exc()
        return None, f"{type(error).__name__}: {error}"


def _arm_decode_hooks(pipe):
    for name in ("vae", "audio_vae"):
        module = getattr(pipe, name, None)
        if module is None:
            continue
        inner = module.decode

        def armed(*args, _module=module, _decode=inner, **kwargs):
            hook = getattr(_module, "_hf_hook", None)
            if hook is not None:
                hook.pre_forward(_module)
            return _decode(*args, **kwargs)

        module.decode = armed


@cache
def conditioner():
    from gradio_client import Client
    return Client(CONDITIONER_SPACE)


def conditioner_client(ip_token):
    if not ip_token:
        return conditioner()
    from gradio_client import Client
    return Client(CONDITIONER_SPACE, headers={"x-ip-token": ip_token})


def encode_remote(prompt, image_path, last_image_path, canvas, num_frames, rewrite_prompt=False, ip_token=None):
    from gradio_client import handle_file
    from safetensors import safe_open

    requested = int(num_frames)
    wire_frames = min(requested, CONDITIONER_MAX_FRAMES)
    path, plan = conditioner_client(ip_token).predict(
        prompt=prompt,
        image_path=handle_file(image_path) if image_path else None,
        last_image_path=handle_file(last_image_path) if last_image_path else None,
        canvas=canvas,
        num_frames=wire_frames,
        rewrite_prompt=bool(rewrite_prompt),
        api_name="/encode",
    )
    with safe_open(path, framework="pt") as handle:
        metadata = dict(handle.metadata())
        metadata["num_frames"] = str(requested)
        return handle.get_tensor("prompt_embeds"), handle.get_tensor("text_token_tags"), metadata, plan


_DUR_B, _DUR_C = 1.1745e-4, 3.8396e-9
_DECODE_BASE, _DECODE_PER_DEFAULT_CANVAS, _DEFAULT_CANVAS_PIXELS = 15, 15, 960 * 544 * 124
_PLACEMENT_ALLOWANCE, _PAD = 12, 10


def estimate_seconds(height, width, num_frames, steps, num_keyframes=0, with_audio=True) -> int:
    height, width, num_frames, steps = int(height), int(width), int(num_frames), int(steps)
    latent_frames = (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2
    patches = (height // 32) * (width // 32)
    rows = latent_frames * patches + int(num_keyframes) * patches
    denoise = steps * (_DUR_B * rows + _DUR_C * rows**2)
    decode = _DECODE_BASE + _DECODE_PER_DEFAULT_CANVAS * (height * width * num_frames) / _DEFAULT_CANVAS_PIXELS
    if not with_audio:
        decode *= 0.85
    return max(60, int(denoise + decode) + _PLACEMENT_ALLOWANCE + _PAD)


def get_duration(
    prompt_embeds, text_token_tags, image, last_image, height, width, num_frames, steps, seed,
    lora="lightx", with_audio=True, gpu_duration=DEFAULT_GPU_DURATION, *a, **k
):
    if gpu_duration:
        try:
            value = int(gpu_duration)
        except (TypeError, ValueError):
            value = 0
        if value in GPU_DURATION_CHOICES:
            return value
    keyframes = int(image is not None) + int(last_image is not None)
    return estimate_seconds(height, width, num_frames, steps, keyframes, with_audio)


@spaces.GPU(duration=get_duration, size=GPU_SIZE)
def _generate(
    prompt_embeds, text_token_tags, image, last_image, height, width, num_frames, steps, seed,
    lora="lightx", with_audio=True, gpu_duration=DEFAULT_GPU_DURATION,
):
    import torch
    active_lora = _set_lora(PIPE.transformer, lora)
    pipe = PIPE if (with_audio or MUTE_PIPE is None) else MUTE_PIPE
    decoded_audio = with_audio or MUTE_PIPE is None

    if PLACEMENT == "lazy":
        PIPE.to("cuda")
    elif PLACEMENT == "pack":
        PIPE.vae.to("cuda")
        if decoded_audio:
            PIPE.audio_vae.to("cuda")

    state = pipe(
        prompt_embeds=prompt_embeds.to("cuda"),
        text_token_tags=text_token_tags,
        image=image,
        last_image=last_image,
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        num_inference_steps=int(steps) + 1,
        generator=torch.Generator("cpu").manual_seed(int(seed)),
    )
    frames = state.get("videos")[0]
    if not decoded_audio:
        return frames, None, None, active_lora
    return frames, state.get("audio")[0].cpu(), state.get("sampling_rate"), active_lora


def _fit_keyframe(image_path, current_canvas):
    from PIL import Image as _Image
    img = _Image.open(image_path)
    aspect = img.width / img.height
    fastest = {}
    for label, (h, w) in CANVASES.items():
        r = w / h
        if r not in fastest or w * h < fastest[r][1][0] * fastest[r][1][1]:
            fastest[r] = (label, (h, w))
    ratio = min(fastest, key=lambda r: abs(r - aspect))
    label, (h, w) = fastest[ratio]

    cur_h, cur_w = CANVASES[current_canvas]
    if abs(cur_w / cur_h - aspect) <= abs(ratio - aspect):
        label = current_canvas
        h, w = cur_h, cur_w

    target = w / h
    if abs(img.width / img.height - target) > 1e-3:
        if img.width / img.height > target:
            new_w = int(img.height * target)
            left = (img.width - new_w) // 2
            img = img.crop((left, 0, left + new_w, img.height))
        else:
            new_h = int(img.width / target)
            top = (img.height - new_h) // 2
            img = img.crop((0, top, img.width, top + new_h))
        img.save(image_path)
    return image_path, label


LORA_NAMES = ("larry", "lightx", "lightx8", "realism", "joyfox", "H3-Facial-Realism-CloseUp", "off")


def _resolve_lora(lora) -> str:
    if not isinstance(lora, str) or not lora.strip():
        return "lightx"
    value = lora.strip().lower()
    lowered = {name.lower(): name for name in LORA_NAMES}
    if value in lowered:
        return lowered[value]
    matches = [name for lower, name in lowered.items() if lower.startswith(value)]
    if len(matches) == 1:
        return matches[0]
    raise gr.Error(f"Unknown LoRA `{lora}`. Allowed: {', '.join(LORA_NAMES)}.")


def keyframe(path):
    from PIL import Image, ImageOps
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB") if path else None


def _render_clip(prompt, first, last, canvas, duration, steps, seed, upsample, lora, with_audio, ip_token, work,
                 gpu_duration=DEFAULT_GPU_DURATION):
    from diffusers.utils import encode_video

    num_frames = snap_frames(duration)
    conditioned = time.time()
    prompt_embeds, text_token_tags, metadata, plan = encode_remote(
        prompt, first, last, canvas, num_frames, rewrite_prompt=upsample, ip_token=ip_token
    )
    condition_seconds = time.time() - conditioned
    height, width, num_frames = (int(metadata[key]) for key in ("height", "width", "num_frames"))
    refined = plan.get("refined_prompt") or ""

    started = time.time()
    frames, audio, sampling_rate, active_lora = _generate(
        prompt_embeds, text_token_tags, keyframe(first), keyframe(last),
        height, width, num_frames, steps, seed, lora, with_audio, gpu_duration,
    )
    generate_seconds = time.time() - started

    path = os.path.join(work, f"clip-{int(time.time() * 1000)}.mp4")
    if audio is not None:
        encode_video(frames, fps=FPS, output_path=path, audio=audio, audio_sample_rate=sampling_rate)
    else:
        encode_video(frames, fps=FPS, output_path=path)

    report = (
        f"{width}x{height} · {num_frames} frames ({num_frames / FPS:.3f} s) · {int(steps)} steps · "
        f"conditioner {condition_seconds:.0f}s ({plan['num_text_tokens']} tokens"
        f"{', upsampled' if refined else ''}) · denoise+decode {generate_seconds:.0f}s "
        f"({generate_seconds / int(steps):.1f} s/step) · LoRA {active_lora} · "
        f"audio {'yes' if audio is not None else 'no'} · seed {int(seed)} · GPU booking {int(gpu_duration)}s"
    )
    print(f"[gen] {report}", flush=True)
    return path, frames[-1], report, refined


def _concat(paths, output_path):
    listing = os.path.join(os.path.dirname(output_path), "concat.txt")
    with open(listing, "w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"file '{os.path.abspath(path)}'\n")

    base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listing]
    proc = subprocess.run(base + ["-c", "copy", output_path], capture_output=True, text=True)
    if proc.returncode != 0:
        proc = subprocess.run(
            base + ["-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-movflags", "+faststart", output_path],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise gr.Error(f"ffmpeg concat failed: {proc.stderr[-1500:]}")
    return output_path


def plan_clips(total: float, chunk: float) -> list[float]:
    count = max(1, math.ceil(total / chunk))
    each = total / count
    if each < MIN_UI_DURATION:
        count = max(1, int(total // MIN_UI_DURATION))
        each = total / count
    return [each] * count


CHAIN_SUFFIX = (
    "\n\nContinue seamlessly from the provided first frame. Preserve the same subject identity, clothing, "
    "environment, lighting and camera style; do not cut to a new shot."
)


def generate(
    prompt, first_frame=None, last_frame=None, canvas=DEFAULT_CANVAS, duration=3.0, mode="single",
    steps=5, seed=42, upsample=False, lora="lightx", with_audio=True, ip_token=None, progress=None,
    gpu_duration=DEFAULT_GPU_DURATION,
):
    if LOAD_ERROR:
        raise gr.Error(LOAD_ERROR.replace("**", "").replace("`", ""))
    if PIPE is None:
        raise gr.Error("The denoiser is still loading — check the Space logs and try again in a moment.")
    if not prompt or not str(prompt).strip():
        raise gr.Error("MiniMax-H3 always needs a prompt, with or without keyframes.")

    prompt = str(prompt).strip()
    lora = _resolve_lora(lora)
    canvas = canvas if canvas in CANVASES else DEFAULT_CANVAS
    duration = max(MIN_UI_DURATION, min(float(duration), MAX_UI_DURATION))
    with_audio = bool(with_audio)

    try:
        gpu_duration = int(gpu_duration)
    except (TypeError, ValueError):
        gpu_duration = DEFAULT_GPU_DURATION
    if gpu_duration not in GPU_DURATION_CHOICES:
        gpu_duration = min(GPU_DURATION_CHOICES, key=lambda v: abs(v - gpu_duration))

    first, last = first_frame, last_frame
    if first:
        first, canvas = _fit_keyframe(first, canvas)
    if last:
        last, canvas = _fit_keyframe(last, canvas)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    work = tempfile.mkdtemp(prefix="h3-", dir=OUTPUT_DIR)
    durations = [duration] if mode == "single" else plan_clips(duration, CHUNK_SECONDS)

    try:
        segments, reports, refined_all = [], [], []
        carry = first
        for index, clip_duration in enumerate(durations):
            is_last = index == len(durations) - 1
            if progress is not None:
                progress(index / len(durations), desc=f"Clip {index + 1}/{len(durations)} ({clip_duration:.1f}s)")
            clip_prompt = prompt if index == 0 else prompt + CHAIN_SUFFIX
            path, tail, report, refined = _render_clip(
                clip_prompt, carry, last if is_last else None, canvas, clip_duration,
                steps, int(seed) + index, upsample and index == 0, lora, with_audio, ip_token, work, gpu_duration,
            )
            segments.append(path)
            reports.append(f"Clip {index + 1}: {report}")
            if refined:
                refined_all.append(refined)
            if not is_last:
                carry = os.path.join(work, f"handoff-{index}.png")
                tail.save(carry)

        final = os.path.join(OUTPUT_DIR, f"h3-{int(time.time() * 1000)}.mp4")
        if len(segments) == 1:
            shutil.move(segments[0], final)
        else:
            _concat(segments, final)

        summary = (
            f"**{mode}** · {len(durations)} clip(s) · target {duration:.2f}s · "
            f"actual {sum(snap_frames(d) for d in durations) / FPS:.2f}s · GPU booking {gpu_duration}s/clip\n\n"
            + "\n\n".join(reports)
        )
        if mode == "single" and duration > 15:
            summary += "\n\n> Beyond 15 s is outside the H3 training distribution. Try chain mode if the video drifts."
        return final, summary, "\n\n---\n\n".join(refined_all)
    except gr.Error:
        raise
    except Exception as error:
        message = str(error).lower()
        if any(hint in message for hint in ("gpu limit", "quota", "could not allocate", "too many", "concurrent")):
            raise gr.Error(
                "The shared ZeroGPU pool is currently at its limit — this is not a problem with your inputs. "
                "Wait a minute and try again."
            ) from error
        traceback.print_exc()
        raise gr.Error(f"Generation failed: {type(error).__name__}: {error}") from error
    finally:
        shutil.rmtree(work, ignore_errors=True)


def save_b64_image(b64_str):
    if not b64_str: return None
    if "," in b64_str:
        _, data = b64_str.split(",", 1)
    else:
        data = b64_str
    img_data = base64.b64decode(data)
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_file.write(img_data)
    temp_file.close()
    return temp_file.name

# ── Gradio Server (Server mode): FastAPI + Gradio queue/API engine ────────────
app = Server(title="MiniMax-H3-Turbo-LoRA-Fast")

@app.mcp.tool(name="generate_video")
@app.api(name="generate_video")
def infer(
    prompt: str,
    images_b64_json: str,
    canvas: str,
    duration: float,
    mode: str,
    steps: int,
    seed: int,
    upsample: bool,
    lora: str,
    with_audio: bool,
    gpu_duration: int,
) -> dict:
    """Generates a video using MiniMax-H3-Turbo-LoRA-Fast."""
    images = json.loads(images_b64_json) if images_b64_json else []
    first_frame = save_b64_image(images[0]) if len(images) > 0 else None
    last_frame = save_b64_image(images[1]) if len(images) > 1 else None
    
    final_video_path, summary, refined = generate(
        prompt, first_frame, last_frame, canvas, duration, mode, steps, seed, upsample, lora, with_audio,
        ip_token=None, progress=None, gpu_duration=gpu_duration
    )
    
    with open(final_video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()
    os.remove(final_video_path)
    
    return {
        "video": f"data:video/mp4;base64,{video_b64}",
        "report": summary,
        "refined": refined
    }

@app.get("/api/config")
def client_config():
    return {"examples": []}

@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

load_models()

if __name__ == "__main__":
    app.launch(show_error=True, mcp_server=True, allowed_paths=[OUTPUT_DIR])