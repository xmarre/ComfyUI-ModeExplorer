"""ReDiverse for native Krea 2: first-step adapter, teacher directions and archives.

Method and reference configuration: "Restoring Diversity in Distilled Image and Video
Generators by Probing Activations" (arXiv:2611.18550) and its Krea 2 port
(github.com/H-furkan/rediverse, krea2/). This module holds the framework-independent
parts: adapter parsing, the first-evaluation schedule contract, the reference teacher
schedule, coefficient seeds and the direction archive format.
"""

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass

import torch

UPSTREAM_REPOSITORY = "https://github.com/H-furkan/rediverse"
UPSTREAM_REVISION = "0a39a7ce38c783038e2394c72d1b877c0129552b"
ADAPTER_RELEASE = "ReDiverse/Krea-2-ReDiverse"
ADAPTER_RELEASE_REVISION = "e13314703bf844e06c8caf42055add96288f480e"
ADAPTER_FILE = "krea-2-rediverse-comfy.safetensors"
TEACHER = ("krea/Krea-2-Raw", "6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603")
STUDENT = ("krea/Krea-2-Turbo", "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7")

ANCHOR_BLOCK = 26
TEACHER_STEPS = 52
TEACHER_FORWARD = 5
ROLLOUTS = 25
DIRECTIONS = 8
PAPER_BAND = 2.0
RELEASE_BAND = 4.0
ROLLOUT_SAMPLE_BASE = 100          # estimation latents j = 100..124, disjoint from evaluation samples
TF_ALPHA_SEED0 = 7_000_000
TF_ALPHA_STRIDE = 1009

ATTENTION = ("wq", "wk", "wv", "gate", "wo")
MLP = ("gate", "up", "down")
COMFY_MODULES = tuple(f"attn.{name}" for name in ATTENTION) + tuple(f"mlp.{name}" for name in MLP)
DIFFUSERS_MODULES = {
    "attn.to_q": "attn.wq", "attn.to_k": "attn.wk", "attn.to_v": "attn.wv", "attn.to_gate": "attn.gate",
    "attn.to_out.0": "attn.wo", "ff.gate": "mlp.gate", "ff.up": "mlp.up", "ff.down": "mlp.down",
}
_COMFY_KEY = re.compile(r"^(?:diffusion_model\.)?blocks\.(\d+)\.((?:attn|mlp)\.[a-z]+)\.(lora_A|lora_B|alpha)(?:\.weight)?$")
_DIFFUSERS_KEY = re.compile(r"^(?:transformer\.)?transformer_blocks\.(\d+)\.(.+?)\.(lora_A|lora_B)(?:\.[A-Za-z_]+)?\.weight$")
MAX_ADAPTER_BYTES = 2 << 30
MAX_ARCHIVE_BYTES = 8 << 30
ARCHIVE_SCHEMA = 1
ARCHIVE_KEY = "modeexplorer_rediverse"
SCHEDULES = ("first_evaluation", "first_interval")


# ------------------------------------------------------------------ adapter --
@dataclass
class Adapter:
    """Parsed low-rank adapter: {block: {module: (A [r, in], B [out, r], scale)}} on CPU, float32."""

    path: str
    modules: dict
    rank: int
    metadata: dict
    sha256: str
    parameters: int

    @property
    def blocks(self):
        return sorted(self.modules)

    def describe(self):
        return {"file": os.path.basename(self.path), "sha256": self.sha256, "rank": self.rank,
                "blocks": f"{self.blocks[0]}..{self.blocks[-1]}" if self.blocks else "",
                "modules": sum(len(m) for m in self.modules.values()), "parameters": self.parameters,
                "method": self.metadata.get("method", "")}


def _file_sha256(path, limit=MAX_ADAPTER_BYTES):
    size = os.path.getsize(path)
    if size > limit:
        raise ValueError(f"{os.path.basename(path)} is larger than the {limit >> 20} MiB limit.")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_adapter(path):
    """Strictly parse a ReDiverse-FT adapter in ComfyUI (original Krea 2) or diffusers names."""
    from safetensors import safe_open

    digest = _file_sha256(path)
    entries = {}
    with safe_open(path, framework="pt", device="cpu") as package:
        metadata = dict(package.metadata() or {})
        keys = list(package.keys())
        if not keys:
            raise ValueError("The adapter file holds no tensors.")
        for key in keys:
            match = _COMFY_KEY.match(key)
            if match:
                block, module, part = int(match.group(1)), match.group(2), match.group(3)
                if module not in COMFY_MODULES:
                    raise ValueError(f"Unsupported adapter module {key!r}.")
            else:
                match = _DIFFUSERS_KEY.match(key)
                if not match or match.group(2) not in DIFFUSERS_MODULES:
                    raise ValueError(f"Unrecognized adapter tensor {key!r}; expected Krea 2 single-stream LoRA keys.")
                block, module, part = int(match.group(1)), DIFFUSERS_MODULES[match.group(2)], match.group(3)
            slot = entries.setdefault((block, module), {})
            if part in slot:
                raise ValueError(f"Duplicate adapter tensor for block {block} {module} {part}.")
            slot[part] = package.get_tensor(key)
    modules, rank, parameters = {}, None, 0
    for (block, module), parts in sorted(entries.items()):
        if set(parts) - {"alpha"} != {"lora_A", "lora_B"}:
            raise ValueError(f"Block {block} {module} needs both lora_A and lora_B.")
        down, up = parts["lora_A"], parts["lora_B"]
        if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0] or down.shape[0] < 1:
            raise ValueError(f"Block {block} {module} has inconsistent LoRA shapes {tuple(down.shape)} / {tuple(up.shape)}.")
        if rank is None:
            rank = down.shape[0]
        elif down.shape[0] != rank:
            raise ValueError("The adapter mixes LoRA ranks.")
        down, up = down.float().contiguous(), up.float().contiguous()
        if not (torch.isfinite(down).all() and torch.isfinite(up).all()):
            raise ValueError(f"Block {block} {module} holds nonfinite adapter weights.")
        scale = 1.0
        if "alpha" in parts:
            alpha = float(parts["alpha"].float().reshape(-1)[0])
            if not math.isfinite(alpha) or alpha <= 0:
                raise ValueError(f"Block {block} {module} has an invalid alpha.")
            scale = alpha / rank
        elif metadata.get("alpha") and metadata.get("rank"):
            # The released file records alpha = rank (scale 1) in its metadata.
            alpha, recorded = float(metadata["alpha"]), int(metadata["rank"])
            if recorded != rank or not math.isfinite(alpha) or alpha <= 0:
                raise ValueError("Adapter metadata disagrees with its tensors.")
            scale = alpha / rank
        modules.setdefault(block, {})[module] = (down, up, scale)
        parameters += down.numel() + up.numel()
    if not any(float(m[1].abs().sum()) > 0 for b in modules.values() for m in b.values()):
        raise ValueError("The adapter's lora_B tensors are all zero.")
    return Adapter(path, modules, rank, metadata, digest, parameters)


def verify_adapter(adapter, diffusion_model):
    """Match every adapter module with a native Krea 2 linear of the loaded model."""
    blocks = diffusion_model.blocks
    for block, modules in adapter.modules.items():
        if not 0 <= block < len(blocks):
            raise ValueError(f"Adapter block {block} is outside the loaded model's {len(blocks)} blocks.")
        for name, (down, up, _) in modules.items():
            layer = resolve_module(blocks[block], name)
            inputs, outputs = getattr(layer, "in_features", None), getattr(layer, "out_features", None)
            if inputs is None or outputs is None:
                raise ValueError(f"Block {block} {name} is not a native linear layer.")
            if (down.shape[1], up.shape[0]) != (inputs, outputs):
                raise ValueError(f"Adapter block {block} {name} expects {down.shape[1]}->{up.shape[0]} features; "
                                 f"the loaded model has {inputs}->{outputs}.")


def resolve_module(block, name):
    module = block
    for part in name.split("."):
        module = getattr(module, part)
    return module


# ------------------------------------------------------------------ schedule --
class FirstEvaluations:
    """Admit the opening model evaluations of one sampler invocation.

    ``first_evaluation`` (reference) admits every evaluation at the first ``count``
    distinct evaluation sigmas, in order, and closes at the first evaluation at any
    later sigma. All CFG branches and nested auxiliary forwards at an admitted sigma
    share the adapted model. With Euler and one sigma this is exactly the upstream
    "adapter active in sampling step 1" rule. ``first_interval`` additionally admits
    every evaluation strictly above the second schedule sigma, which includes the
    intermediate stages of multistage solvers within step 1.
    """

    def __init__(self, schedule="first_evaluation", count=1, boundary=None):
        if schedule not in SCHEDULES or type(count) is not int or not 1 <= count <= 4:
            raise ValueError("Unknown ReDiverse schedule.")
        self.schedule = schedule
        self.count = count
        self.boundary = boundary
        self.seen = []
        self.closed = False
        self.admitted = 0

    def pending(self, sigma):
        if self.closed:
            return False
        if self.schedule == "first_interval" and self.boundary is not None:
            return sigma > self.boundary
        return len(self.seen) < self.count or sigma == self.seen[-1]

    def admit(self, sigma):
        if self.closed:
            return False
        if self.schedule == "first_interval" and self.boundary is not None:
            if sigma > self.boundary:
                if not self.seen or self.seen[-1] != sigma:
                    self.seen.append(sigma)
                self.admitted += 1
                return True
            self.closed = True
            return False
        if self.seen and sigma == self.seen[-1]:
            self.admitted += 1
            return True
        if len(self.seen) < self.count:
            self.seen.append(sigma)
            self.admitted += 1
            return True
        self.closed = True
        return False


def flow_shift(mu, t):
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))


def image_mu(tokens):
    """Krea2Pipeline / FLUX calculate_shift (base 256 / 0.5, max 6400 / 1.15)."""
    m = (1.15 - 0.5) / (6400 - 256)
    return tokens * m + (0.5 - m * 256)


def reference_sigmas(steps, tokens=None, mu=None):
    """FlowMatchEuler sigmas of the Krea 2 reference schedules, terminal zero included."""
    if steps < 1:
        raise ValueError("A schedule needs at least one step.")
    mu = image_mu(tokens) if mu is None else mu
    base = [1.0 - i * (1.0 - 1.0 / steps) / max(steps - 1, 1) for i in range(steps)]
    return [flow_shift(mu, t) for t in base] + [0.0]


# ------------------------------------------------------------------ coefficients --
def coefficient_seed(seed, row, trial=0):
    """Deterministic ReDiverse-TF coefficient seed for a noise seed and batch row.

    Mirrors the upstream rule 7_000_000 + i * 1009 + s with the ComfyUI noise seed in
    place of the prompt index and the batch row as the sample index.
    """
    value = TF_ALPHA_SEED0 + (int(seed) % (1 << 40)) * TF_ALPHA_STRIDE + int(row) + int(trial) * 1_000_003
    return value % ((1 << 63) - 1)


def coefficients(seed, rows, k, band, trial=0):
    """[rows, k] float32 coefficients alpha ~ U[-band, band] from independent CPU generators."""
    out = torch.empty(rows, k, dtype=torch.float32)
    for row in range(rows):
        generator = torch.Generator("cpu").manual_seed(coefficient_seed(seed, row, trial))
        out[row].uniform_(-float(band), float(band), generator=generator)
    return out


def packed_noise(seed, height, width, channels=16, patch=2):
    """Upstream packed-latent draw randn([tokens, C*p*p]) on a CPU generator, unpacked to [C, H, W].

    The packed column order is (channel, patch row, patch column), as in Krea2Pipeline and
    ComfyUI's ``process_img``, so a calibration rollout starts from the same tensor the
    reference implementation draws for this seed.
    """
    if height % patch or width % patch:
        raise ValueError("Reference packed noise needs a latent divisible by the patch size.")
    h, w = height // patch, width // patch
    generator = torch.Generator("cpu").manual_seed(int(seed))
    packed = torch.randn((h * w, channels * patch * patch), generator=generator)
    return packed.reshape(h, w, channels, patch, patch).permute(2, 0, 3, 1, 4).reshape(channels, height, width)


# ------------------------------------------------------------------ fingerprints --
def tensor_fingerprint(tensor):
    """SHA-256 of a tensor's float32 values and shape (dtype-normalized, device independent)."""
    value = tensor.detach().to("cpu", torch.float32).contiguous()
    digest = hashlib.sha256(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_fingerprint(diffusion_model, samples=4096):
    """Stable identity of loaded transformer weights from fixed slices of named tensors.

    Reads raw storage, so quantized checkpoints are identified by their stored
    values. This distinguishes checkpoints and precisions; it is not a full hash.
    """
    digest = hashlib.sha256()
    names = ("first.weight", "blocks.0.attn.wq.weight", f"blocks.{len(diffusion_model.blocks) // 2}.mlp.down.weight",
             f"blocks.{len(diffusion_model.blocks) - 1}.attn.wo.weight", "last.linear.weight")
    state = dict(diffusion_model.named_parameters())
    state.update(dict(diffusion_model.named_buffers()))
    for name in names:
        tensor = state.get(name)
        if tensor is None:
            digest.update(f"{name}:absent".encode())
            continue
        flat = tensor.detach().reshape(-1)[:samples]
        raw = flat.to("cpu").contiguous()
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        try:
            digest.update(raw.view(torch.uint8).numpy().tobytes())
        except (RuntimeError, TypeError):
            digest.update(raw.float().numpy().tobytes())
    return digest.hexdigest()


def patch_fingerprint(patcher):
    """Identity of a ModelPatcher's weight patches: keys, strengths and patch-tensor samples."""
    digest = hashlib.sha256()
    count = 0
    for key in sorted(getattr(patcher, "patches", {})):
        for entry in patcher.patches[key]:
            count += 1
            strength_patch, data = entry[0], entry[1]
            strength_model = entry[2] if len(entry) > 2 else 1.0
            digest.update(f"{key}:{strength_patch}:{strength_model}:{type(data).__name__}".encode())
            tensors = getattr(data, "weights", data)
            for item in tensors if isinstance(tensors, (tuple, list)) else (tensors,):
                if torch.is_tensor(item):
                    digest.update(item.detach().reshape(-1)[:256].to("cpu", torch.float32).numpy().tobytes())
    return {"weight_patches": count, "sha256": digest.hexdigest()}


# ------------------------------------------------------------------ archives --
REQUIRED_METADATA = (
    "schema", "implementation", "coordinate", "block", "token_grid", "d_model", "teacher_model_sha256",
    "teacher_patches", "conditioning_sha256", "negative_conditioning_sha256", "sigma_in", "sigma_out",
    "teacher_forward", "teacher_steps", "teacher_sigmas", "guidance", "seeds", "rollouts", "initial_noise_sha256",
    "normalization", "basis_dtype", "directions", "valid_directions", "identity_sha256",
)
COORDINATE = "block_output_image_rows"
# Tolerances for a bf16-stored basis: rounding perturbs each unit row by about 2^-9 relative.
BASIS_TOLERANCE = 2e-2
SPREAD_TOLERANCE = 1e-3


@dataclass
class Directions:
    """A validated ReDiverse-TF direction archive held on CPU."""

    path: str
    U: torch.Tensor
    s: torch.Tensor
    eigenvalues: torch.Tensor
    mean: torch.Tensor | None
    metadata: dict
    sha256: str = ""

    @property
    def block(self):
        return int(self.metadata["block"])

    @property
    def grid(self):
        return tuple(self.metadata["token_grid"])

    @property
    def d_model(self):
        return int(self.metadata["d_model"])

    def describe(self):
        keys = ("block", "token_grid", "d_model", "teacher_forward", "sigma_in", "rollouts", "directions",
                "valid_directions", "guidance", "conditioning_sha256", "teacher_model_sha256", "label")
        return {"file": os.path.basename(self.path), "sha256": self.sha256,
                **{key: self.metadata.get(key) for key in keys}}


def save_directions(path, U, s, eigenvalues, metadata, mean=None):
    """Write a direction archive atomically (temporary file, then rename)."""
    from safetensors.torch import save_file

    missing = [key for key in REQUIRED_METADATA if key not in metadata]
    if missing:
        raise ValueError("Direction metadata is missing " + ", ".join(missing))
    tensors = {"U": U.contiguous(), "s": s.float().contiguous(), "eigenvalues": eigenvalues.float().contiguous()}
    if mean is not None:
        tensors["mean"] = mean.contiguous()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + f".tmp{os.getpid()}"
    save_file(tensors, temporary, metadata={ARCHIVE_KEY: json.dumps(metadata, sort_keys=True)})
    os.replace(temporary, path)
    return path


def load_directions(path, max_bytes=MAX_ARCHIVE_BYTES):
    """Load and validate a direction archive. Never unpickles data."""
    from safetensors import safe_open

    if not os.path.isfile(path):
        raise ValueError(f"Direction archive {os.path.basename(path)} does not exist.")
    digest = _file_sha256(path, max_bytes)
    with safe_open(path, framework="pt", device="cpu") as package:
        raw = (package.metadata() or {}).get(ARCHIVE_KEY)
        if raw is None:
            raise ValueError("This file is not a ModeExplorer ReDiverse direction archive.")
        try:
            metadata = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("The direction archive metadata is malformed.") from exc
        if not isinstance(metadata, dict) or metadata.get("schema") != ARCHIVE_SCHEMA:
            raise ValueError("Unsupported direction archive schema.")
        missing = [key for key in REQUIRED_METADATA if key not in metadata]
        if missing:
            raise ValueError("The direction archive is incomplete: " + ", ".join(missing))
        keys = set(package.keys())
        if not {"U", "s", "eigenvalues"} <= keys or keys - {"U", "s", "eigenvalues", "mean"}:
            raise ValueError("The direction archive holds unexpected tensors.")
        U, s, eigenvalues = package.get_tensor("U"), package.get_tensor("s"), package.get_tensor("eigenvalues")
        mean = package.get_tensor("mean") if "mean" in keys else None
    grid = metadata["token_grid"]
    if (not isinstance(grid, list) or len(grid) != 2 or any(type(v) is not int or v < 1 for v in grid)
            or type(metadata["d_model"]) is not int or type(metadata["block"]) is not int):
        raise ValueError("The direction archive geometry is malformed.")
    features = grid[0] * grid[1] * metadata["d_model"]
    k = metadata["directions"]
    if (U.ndim != 2 or U.shape != (k, features) or s.shape != (k,) or eigenvalues.ndim != 1
            or U.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or str(U.dtype).removeprefix("torch.") != metadata["basis_dtype"]
            or (mean is not None and mean.shape != (features,))):
        raise ValueError("The direction archive tensors do not match their recorded shapes.")
    if not (torch.isfinite(U.float()).all() and torch.isfinite(s).all() and torch.isfinite(eigenvalues).all()
            and (mean is None or torch.isfinite(mean.float()).all())):
        raise ValueError("The direction archive holds nonfinite values.")
    if metadata["coordinate"] != COORDINATE:
        raise ValueError(f"Unsupported direction coordinate {metadata['coordinate']!r}; expected {COORDINATE}.")
    usable = s > 0
    if (s < 0).any() or int(usable.sum()) < 1 or int(usable.sum()) != metadata["valid_directions"]:
        raise ValueError("The direction archive holds no usable directions or disagrees with its valid count.")
    rows = U.float()
    norms = rows.norm(dim=1)
    if bool(norms[~usable].any()):
        raise ValueError("The direction archive holds nonzero basis rows without spread.")
    valid = rows[usable]
    gram = valid @ valid.T
    if not torch.allclose(gram, torch.eye(len(valid)), atol=BASIS_TOLERANCE):
        raise ValueError("The direction archive basis is not unit-normalized and mutually orthogonal.")
    rollouts = metadata["rollouts"]
    if (type(rollouts) is not int or eigenvalues.shape != (rollouts,) or len(metadata["seeds"]) != rollouts
            or k > rollouts):
        raise ValueError("The direction archive eigenvalues disagree with its rollout count.")
    leading = float(eigenvalues.max()) if eigenvalues.numel() else 0.0
    if leading <= 0 or bool((eigenvalues[:-1] < eigenvalues[1:] - 1e-6 * leading).any()) \
            or bool((eigenvalues < -1e-6 * leading).any()):
        raise ValueError("The direction archive eigenvalues are not a nonnegative descending spectrum.")
    implied = (eigenvalues[:k].double().clamp_min(0) / rollouts).sqrt()
    spreads = s.double()
    if not torch.allclose(spreads[usable], implied[usable], rtol=SPREAD_TOLERANCE, atol=1e-12):
        raise ValueError("The direction archive spreads disagree with sqrt(eigenvalue / rollouts).")
    return Directions(path, U, s.float(), eigenvalues.float(), mean, metadata, digest)


def archive_root():
    import folder_paths

    return os.path.join(folder_paths.get_output_directory(), "modeexplorer_rediverse")


def direction_files():
    root = os.path.join(archive_root(), "directions")
    if not os.path.isdir(root):
        return []
    return sorted(name for name in os.listdir(root) if name.endswith(".safetensors"))
