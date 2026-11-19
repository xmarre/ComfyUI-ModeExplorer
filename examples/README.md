# Example workflows

Open a JSON file in the ComfyUI canvas and select your installed Krea 2 Turbo diffusion model, Qwen3-VL-4B encoder, and Qwen Image VAE. These graphs use Euler, the simple scheduler, CFG 1 and eight steps for the initial pass. The model filename defaults match the ComfyUI Krea 2 blueprint.

| Workflow | Setup |
| --- | --- |
| [A/B](ab.json) | Runs native and ModeExplorer at the same fixed seed. Change both seeds together for each pair. |
| [Turbo](turbo.json) | One empty latent through the regular KSampler. |
| [Batch](batch.json) | Four samples under the same prompt, with no reference forwards. |
| [Sequential](sequential.json) | One image per queue submission. The native KSampler seed increments when queued. |
| [Multistage](multistage.json) | A 1024px guided composition pass, latent resize, and a 1536px native refinement pass. |

To test initialization alone, select `native` under **feature** and enable **initialization**. Keep schedule, seed and all other graph inputs identical between compared runs. For Raw select Raw weights, encode a true negative prompt, and use your usual Raw step count and CFG. Insert an identity LoRA upstream of the ModeExplorer node and branch both variants from the same LoRA output.
