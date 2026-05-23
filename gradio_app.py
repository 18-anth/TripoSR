import argparse

from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
from tsr.system import TSR
from functools import partial
from PIL import Image
import torch
import rembg
import numpy as np
import gradio as gr
import logging
import os
import tempfile
import time

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(8192)
model.to(device)

rembg_session = rembg.new_session()


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background, foreground_ratio):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + \
            (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image


def generate(image, mc_resolution, decimate_target, formats=["obj", "glb"]):
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    if decimate_target > 0 and len(mesh.faces) > decimate_target:
        import trimesh

        mesh = mesh.simplify_quadric_decimation(decimate_target)
    rv = []
    for format in formats:
        mesh_path = tempfile.NamedTemporaryFile(
            suffix=f".{format}", delete=False)
        mesh.export(mesh_path.name)
        rv.append(mesh_path.name)
    return rv


def run_example(image_pil):
    preprocessed = preprocess(image_pil, False, 0.9)
    mesh_name_obj, mesh_name_glb = generate(
        preprocessed, 320, 0, ["obj", "glb"])
    return preprocessed, mesh_name_obj, mesh_name_glb


CSS = """
:root {
    --primary: #A89CEC;
    --primary-dark: #9285D8;
    --primary-glow: rgba(168, 156, 236, 0.28);
    --accent: #F9A8D4;
    --accent-glow: rgba(249, 168, 212, 0.2);
    --bg-deep: #FAF8FF;
    --bg-card: #FFFFFF;
    --bg-card2: #F3F0FB;
    --bg-input: #F7F5FF;
    --border: rgba(168, 156, 236, 0.3);
    --border-hover: rgba(168, 156, 236, 0.7);
    --text-primary: #3D3659;
    --text-secondary: #7B72A3;
    --text-muted: #B0A8D0;
    --success: #86EFAC;
    --radius: 18px;
    --radius-sm: 12px;
    --shadow: 0 4px 24px rgba(168, 156, 236, 0.15);
}

body, .gradio-container, footer { background: var(--bg-deep) !important; color: var(--text-primary) !important; }
footer { display: none !important; }

/* ── Hero ── */
#hero { background: linear-gradient(135deg, #EDE9FF 0%, #FDE8F4 55%, #E8F4FF 100%);
  border: 1px solid var(--border); border-radius: var(--radius);
  padding: 44px 40px 32px; margin-bottom: 20px;
  position: relative; overflow: hidden;
  box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.7); }
#hero::before { content:''; position:absolute; top:-90px; right:-90px;
  width:320px; height:320px;
  background: radial-gradient(circle, var(--primary-glow) 0%, transparent 70%); pointer-events:none; }
#hero::after { content:''; position:absolute; bottom:-60px; left:25%;
  width:200px; height:200px;
  background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%); pointer-events:none; }
#hero h1 { font-size:2.6rem!important; font-weight:800!important; letter-spacing:-0.5px;
  background: linear-gradient(120deg, var(--primary-dark) 0%, #C084FC 50%, var(--accent) 100%);
  -webkit-background-clip:text!important; -webkit-text-fill-color:transparent!important;
  background-clip:text!important; margin:0 0 10px!important; }
#hero p { color:var(--text-secondary)!important; font-size:.97rem!important; line-height:1.65!important; }
#hero a { color:var(--accent)!important; text-decoration:none!important; }
#hero a:hover { text-decoration:underline!important; }
.badges { display:flex; gap:8px; flex-wrap:wrap; margin-top:16px; }
.badge { display:inline-flex; align-items:center; gap:5px;
  background:rgba(168,156,236,.12); border:1px solid var(--border);
  border-radius:20px; padding:4px 13px; font-size:.76rem; color:var(--text-secondary); font-weight:500; }

/* ── Cards ── */
#left-panel, #right-panel, #examples-panel {
  background:var(--bg-card); border:1px solid var(--border);
  border-radius:var(--radius); padding:22px;
  box-shadow:var(--shadow); transition:border-color .3s; }
#left-panel:hover, #right-panel:hover { border-color:var(--border-hover); }

.section-label { font-size:.7rem!important; font-weight:700!important;
  text-transform:uppercase!important; letter-spacing:1.8px!important;
  color:var(--text-muted)!important; margin-bottom:10px!important; display:block; }

/* ── Upload zone ── */
#content_image { border:2px dashed var(--border)!important;
  border-radius:var(--radius-sm)!important; background:var(--bg-input)!important;
  min-height:260px!important; transition:border-color .3s, box-shadow .3s!important; }
#content_image:hover { border-color:var(--primary)!important; box-shadow:0 0 22px var(--primary-glow)!important; }

/* ── Controls ── */
#controls { background:var(--bg-card2); border:1px solid var(--border);
  border-radius:var(--radius-sm); padding:18px; margin-top:14px; }
label span { color:var(--text-secondary)!important; font-size:.84rem!important; font-weight:500!important; }
input[type=range] { accent-color:var(--primary)!important; }
input[type=checkbox] { accent-color:var(--primary)!important; }

/* ── Generate button ── */
#gen-btn {
  background:linear-gradient(135deg,var(--primary) 0%,var(--primary-dark) 100%)!important;
  border:none!important; border-radius:var(--radius-sm)!important;
  color:#3D3659!important; font-size:1rem!important; font-weight:700!important;
  padding:14px!important; width:100%!important;
  box-shadow:0 4px 24px var(--primary-glow)!important;
  transition:all .25s!important; letter-spacing:.4px!important; margin-top:10px!important; }
#gen-btn:hover { transform:translateY(-2px)!important; box-shadow:0 8px 36px var(--primary-glow)!important; filter:brightness(1.1)!important; }
#gen-btn:active { transform:translateY(0)!important; }

/* ── Tips ── */
#tips { background:rgba(168,156,236,.08); border:1px solid rgba(168,156,236,.3);
  border-radius:var(--radius-sm); padding:13px 16px; margin-top:14px;
  font-size:.83rem; color:var(--text-secondary); line-height:1.75; }
#tips b { color:var(--primary)!important; }

/* ── Tabs ── */
.tab-nav { border-bottom:1px solid var(--border)!important; }
.tab-nav button { background:transparent!important; border:none!important;
  color:var(--text-muted)!important; font-weight:600!important; padding:10px 18px!important;
  border-bottom:2px solid transparent!important; transition:all .2s!important; }
.tab-nav button.selected { color:var(--primary)!important; border-bottom-color:var(--primary)!important; }
.tab-nav button:hover { color:var(--text-primary)!important; }

/* ── 3D Viewer ── */
canvas, .model3D-viewer { border-radius:var(--radius-sm)!important; background:var(--bg-input)!important; min-height:320px!important; }
.output-note { background:rgba(249,168,212,.08); border:1px solid rgba(249,168,212,.3);
  border-radius:var(--radius-sm); padding:9px 13px; font-size:.8rem;
  color:var(--text-secondary); margin-top:8px; }

/* ── Examples ── */
.examples-table img { border-radius:8px!important; border:1px solid var(--border)!important;
  transition:transform .2s, border-color .2s!important; }
.examples-table img:hover { transform:scale(1.06)!important; border-color:var(--primary)!important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg-deep); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--primary); }
"""

with gr.Blocks(title="TripoSR · 3D Reconstruction", css=CSS, theme=gr.themes.Base()) as interface:

    # ── HERO ──
    with gr.Group(elem_id="hero"):
        gr.Markdown("""
# TripoSR
**Fast AI-powered 3D reconstruction from a single image.**
Developed by [Tripo AI](https://www.tripo3d.ai/) &amp; [Stability AI](https://stability.ai/) · Open source on [GitHub](https://github.com/VAST-AI-Research/TripoSR)

<div class="badges">
<span class="badge">⚡ Single Image → 3D Mesh</span>
<span class="badge">🎨 OBJ &amp; GLB Export</span>
<span class="badge">🧠 AI Background Removal</span>
<span class="badge">🔓 Open Source</span>
</div>
""")

    # ── MAIN ROW ──
    with gr.Row(equal_height=False):

        # LEFT – Input & Controls
        with gr.Column(scale=5, elem_id="left-panel"):
            gr.Markdown("<span class='section-label'>📥 Input Image</span>")
            with gr.Row(equal_height=True):
                input_image = gr.Image(
                    label="Upload Image",
                    image_mode="RGBA",
                    sources="upload",
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(
                    label="Preprocessed Preview",
                    interactive=False,
                )

            with gr.Group(elem_id="controls"):
                gr.Markdown("<span class='section-label'>⚙️ Options</span>")
                with gr.Row():
                    do_remove_background = gr.Checkbox(
                        label="✂️  Remove Background (AI)",
                        value=True,
                        info="Auto-removes background using rembg",
                    )
                    foreground_ratio = gr.Slider(
                        label="Foreground Ratio",
                        minimum=0.5, maximum=1.0, value=0.85, step=0.05,
                        info="How much of the frame the subject fills",
                    )
                with gr.Row():
                    mc_resolution = gr.Slider(
                        label="Mesh Resolution",
                        minimum=32, maximum=5000, value=320, step=32,
                        info="Marching Cubes resolution · higher = more detail & slower",
                    )
                    decimate_target = gr.Slider(
                        label="Polygon Limit  (0 = off)",
                        minimum=0, maximum=2000000, value=0, step=50000,
                        info="Reduce polygon count after generation to shrink file size",
                    )

            gr.Markdown("""
<div id="tips">
<b>Tips</b><br>
• Disable <i>Remove Background</i> if your image already has a transparent background.<br>
• Subject should be centered and fill &gt;70% of the image for best results.<br>
• Use resolution <b>256–320</b> for fast previews; increase for final exports.
</div>
""")
            submit = gr.Button("⚡  Generate 3D Model",
                               elem_id="gen-btn", variant="primary")

        # RIGHT – 3D Output
        with gr.Column(scale=6, elem_id="right-panel"):
            gr.Markdown("<span class='section-label'>🧊 3D Output</span>")
            with gr.Tabs():
                with gr.Tab("GLB  (recommended)"):
                    output_model_glb = gr.Model3D(
                        label="Preview — GLB", interactive=False)
                    gr.Markdown(
                        "<div class='output-note'>ℹ️ GLB preserves materials and is web/AR ready. Download for accurate colors.</div>")
                with gr.Tab("OBJ"):
                    output_model_obj = gr.Model3D(
                        label="Preview — OBJ", interactive=False)
                    gr.Markdown(
                        "<div class='output-note'>ℹ️ OBJ is ideal for Blender & 3D editors. Preview is flipped — download for correct orientation.</div>")

    # ── EXAMPLES ──
    with gr.Group(elem_id="examples-panel"):
        gr.Markdown(
            "<span class='section-label'>🖼️ Examples — click to try</span>")
        gr.Examples(
            examples=[
                "examples/hamburger.png",
                "examples/poly_fox.png",
                "examples/robot.png",
                "examples/teapot.png",
                "examples/tiger_girl.png",
                "examples/horse.png",
                "examples/flamingo.png",
                "examples/unicorn.png",
                "examples/chair.png",
                "examples/iso_house.png",
                "examples/marble.png",
                "examples/police_woman.png",
                "examples/captured.jpeg",
            ],
            inputs=[input_image],
            outputs=[processed_image, output_model_obj, output_model_glb],
            cache_examples=False,
            fn=partial(run_example),
            label="",
            examples_per_page=20,
        )

    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background, foreground_ratio],
        outputs=[processed_image],
    ).success(
        fn=generate,
        inputs=[processed_image, mc_resolution, decimate_target],
        outputs=[output_model_obj, output_model_glb],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--username", type=str, default=None, help="Username for authentication"
    )
    parser.add_argument(
        "--password", type=str, default=None, help="Password for authentication"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run the server listener on"
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="use share=True for gradio and make the UI accessible through their site",
    )
    parser.add_argument(
        "--queuesize", type=int, default=1, help="launch gradio queue max_size"
    )
    args = parser.parse_args()
    interface.queue(max_size=args.queuesize)
    interface.launch(
        auth=(
            (args.username, args.password)
            if (args.username and args.password)
            else None
        ),
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.port,
        show_error=True,
    )
