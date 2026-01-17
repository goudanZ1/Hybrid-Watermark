import argparse
import os
from typing import List

import torch
from diffusers import DPMSolverMultistepScheduler
from stable_diffusion.inverse_stable_diffusion import InversableStableDiffusionPipeline


def _load_prompts(args) -> List[str]:
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f.readlines()]
        return [p for p in prompts if p]
    if args.prompt:
        return [args.prompt]
    raise ValueError("Please provide --prompt or --prompt_file.")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=device).manual_seed(args.gen_seed)

    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_path,
        scheduler=scheduler,
        dtype=torch.float16,
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    prompts = _load_prompts(args)
    os.makedirs(args.output_path, exist_ok=True)

    image_index = 0
    for prompt_idx, prompt in enumerate(prompts):
        for i in range(args.num):
            seed = args.gen_seed + image_index
            g.manual_seed(seed)
            outputs = pipe(
                prompt,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                generator=g,
            )
            image = outputs.images[0]
            image.save(f"{args.output_path}/{prompt_idx}_{i}.png")
            image_index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="No Watermark (Custom Prompts)")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--num", default=1, type=int, help="Images per prompt.")
    parser.add_argument("--image_length", default=512, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--gen_seed", default=0, type=int)
    parser.add_argument("--output_path", default="./output_no_wm/")
    parser.add_argument(
        "--model_path",
        default="/home/jiazhao/Hybrid-Watermark/modelscope/AI-ModelScope/stable-diffusion-2-1-base",
    )

    args = parser.parse_args()
    main(args)
