import torch
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
import gradio as gr
import os

# Creamos la carpeta para que no llore el sistema
if not os.path.exists("offload"):
    os.makedirs("offload")

dtype = torch.float16 

# Carga con carpeta de descarga (offload_folder)
transformer = QwenImageTransformer2DModel.from_pretrained(
    "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V4", 
    torch_dtype=dtype,
    device_map="auto",
    load_in_4bit=True,
    offload_folder="offload"  # <--- ESTO ARREGLA EL ERROR
)

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    transformer=transformer,
    torch_dtype=dtype,
    device_map="auto",
    offload_folder="offload" # <--- TAMBIÉN AQUÍ POR SI LAS DUDAS
)

def edit(image, prompt):
    if image is None: return None
    image = image.convert("RGB").resize((512, 512))
    result = pipe(prompt=prompt, image=image).images[0]
    return result

demo = gr.Interface(
    fn=edit,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Prompt")],
    outputs=gr.Image(),
    title="Qwen Ultra Lite"
)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
