import argparse
from tqdm import tqdm
import torch
from diffusers import DPMSolverMultistepScheduler
from stable_diffusion.inverse_stable_diffusion import InversableStableDiffusionPipeline
from utils.optim_utils import get_dataset
from utils.image_utils import set_random_seed, image_distortion
import os


def _has_distortion(args):
    return any(
        getattr(args, name) is not None
        for name in [
            "jpeg_ratio",
            "random_crop_ratio",
            "random_drop_ratio",
            "gaussian_blur_r",
            "median_blur_k",
            "resize_ratio",
            "gaussian_std",
            "sp_prob",
            "brightness_factor",
            "rotate",
        ]
    )


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

    dataset, prompt_key = get_dataset(args)
    os.makedirs(args.output_path, exist_ok=True)

    for i in tqdm(range(args.num)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]

        set_random_seed(seed)
        base_latent = pipe.get_random_latents()

        outputs = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=base_latent,
            generator=g,
        )

        image = outputs.images[0]
        image.save(f"{args.output_path}/{i}.png")

        if _has_distortion(args):
            image_distorted = image_distortion(image, seed, args)
            image_distorted.save(f"{args.output_path}/{i}_distorted.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="No Watermark Baseline")
    parser.add_argument("--num", default=20, type=int)
    parser.add_argument("--image_length", default=512, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--gen_seed", default=0, type=int)
    parser.add_argument("--output_path", default="./output/")
    parser.add_argument("--dataset_path", default="Gustavosta/Stable-Diffusion-Prompts")
    parser.add_argument(
        "--model_path",
        default="/home/jiazhao/Hybrid-Watermark/modelscope/AI-ModelScope/stable-diffusion-2-1-base",
    )

    # image distortion options (same as run_gaussian_shading.py)
    parser.add_argument("--jpeg_ratio", default=None, type=int)
    parser.add_argument("--random_crop_ratio", default=None, type=float)
    parser.add_argument("--random_drop_ratio", default=None, type=float)
    parser.add_argument("--gaussian_blur_r", default=None, type=int)
    parser.add_argument("--median_blur_k", default=None, type=int)
    parser.add_argument("--resize_ratio", default=None, type=float)
    parser.add_argument("--gaussian_std", default=None, type=float)
    parser.add_argument("--sp_prob", default=None, type=float)
    parser.add_argument("--brightness_factor", default=None, type=float)
    parser.add_argument("--rotate", default=None, type=float)

    args = parser.parse_args()
    main(args)
