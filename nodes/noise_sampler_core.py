import hashlib
import struct

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.utils
import latent_preview


MAX_SEED = 0xffffffffffffffff
NOISE_TYPES = [
    "gaussian",
    "uniform",
    "laplacian",
    "pink",
    "brown",
    "blue",
    "violet",
    "pyramid",
    "perlin",
    "studentt",
    "white",
    "grey",
    "velvet",
    "green_test",
    "highres_pyramid",
    "pyramid_discount5",
    "pyramid_mix",
    "rainbow_mild",
    "rainbow_intense",
    "wavelet",
]
ILLUSTRIOUS_MODES = ["balanced", "texture", "composition", "wild"]
PONY_V7_PROFILES = ["stage1_gaussian", "stage2_violet", "balanced", "soft", "graphic"]
UNIVERSAL_NOISE_PROFILES = (
    NOISE_TYPES
    + [f"illustrious_{mode}" for mode in ILLUSTRIOUS_MODES]
    + [f"pony_v7_{profile}" for profile in PONY_V7_PROFILES]
)
PREVIEW_METHODS = ["default", "latent2rgb", "taesd", "none"]
SPECTRAL_NOISE_ALPHA = {
    "brown": 2.0,
    "pink": 1.0,
    "blue": -1.0,
    "violet": -2.0,
}


class OnekoEmptyNoise:
    def __init__(self):
        self.seed = 0

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        return _zeros_like_noise(latent_image, "cpu")


class OnekoRandomNoise:
    def __init__(self, seed, noise_device, noise_type="gaussian", noise_strength=1.0):
        self.seed = seed
        self.noise_device = noise_device
        self.noise_type = noise_type
        self.noise_strength = noise_strength

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        device = _resolve_noise_device(self.noise_device)
        return _prepare_noise(latent_image, self.seed, batch_inds, device, self.noise_type, self.noise_strength)


class OnekoCompositeNoise:
    def __init__(self, seed, noise_device, family, profile, noise_strength=1.0, detail_bias=0.35):
        self.seed = seed
        self.noise_device = noise_device
        self.family = family
        self.profile = profile
        self.noise_strength = noise_strength
        self.detail_bias = detail_bias

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        device = _resolve_noise_device(self.noise_device)
        return _prepare_composite_noise(
            latent_image,
            self.seed,
            batch_inds,
            device,
            self.family,
            self.profile,
            self.noise_strength,
            self.detail_bias,
        )


class OnekoUniversalNoise:
    def __init__(self, seed, noise_device, noise_profile="gaussian", noise_strength=1.0, detail_bias=0.35):
        self.seed = seed
        self.noise_device = noise_device
        self.noise_profile = noise_profile
        self.noise_strength = noise_strength
        self.detail_bias = detail_bias

    def generate_noise(self, input_latent):
        noise = make_noise_generator(
            self.seed,
            self.noise_device,
            self.noise_profile,
            self.noise_strength,
            self.detail_bias,
        )
        return noise.generate_noise(input_latent)


def make_noise_generator(seed, noise_device, noise_profile="gaussian", noise_strength=1.0, detail_bias=0.35):
    if noise_profile not in UNIVERSAL_NOISE_PROFILES:
        raise ValueError(f"Unknown Oneko noise profile: {noise_profile!r}")
    if noise_profile in NOISE_TYPES:
        return OnekoRandomNoise(seed, noise_device, noise_profile, noise_strength)

    if noise_profile.startswith("illustrious_"):
        return OnekoCompositeNoise(
            seed,
            noise_device,
            "illustrious",
            noise_profile.removeprefix("illustrious_"),
            noise_strength,
            detail_bias,
        )

    if noise_profile.startswith("pony_v7_"):
        return OnekoCompositeNoise(
            seed,
            noise_device,
            "pony_v7",
            noise_profile.removeprefix("pony_v7_"),
            noise_strength,
            detail_bias,
        )



def sample_custom_advanced(
    guider,
    sampler,
    sigmas,
    latent_image,
    noise,
    noise_seed,
    start_at_step=0,
    end_at_step=10000,
    return_with_leftover_noise="disable",
    preview_method="default",
):
    latent = latent_image
    latent_samples = latent["samples"]
    latent = latent.copy()
    latent_samples = comfy.sample.fix_empty_latent_channels(
        guider.model_patcher,
        latent_samples,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    latent["samples"] = latent_samples
    sigmas, return_latent = sigmas_for_step_range(
        sigmas,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
    )
    if return_latent:
        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        return (out, out, noise_seed)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    x0_output = {}
    override_preview = preview_method not in (None, "", "default")
    if override_preview:
        latent_preview.set_preview_method(preview_method)
    try:
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(
            noise.generate_noise(latent),
            latent_samples,
            sampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise_seed,
        )
    finally:
        if override_preview:
            latent_preview.set_preview_method(None)
    samples = samples.to(comfy.model_management.intermediate_device())

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    if "x0" in x0_output:
        x0_out = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        if samples.is_nested:
            latent_shapes = [sample.shape for sample in samples.unbind()]
            x0_out = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0_out, latent_shapes))
        out_denoised = latent.copy()
        out_denoised["samples"] = x0_out
    else:
        out_denoised = out

    return (out, out_denoised, noise_seed)


def sigmas_for_step_range(sigmas, start_at_step=0, end_at_step=10000, return_with_leftover_noise="disable"):
    start_at_step = max(0, int(start_at_step))
    end_at_step = max(0, int(end_at_step))
    force_full_denoise = return_with_leftover_noise != "enable"

    if sigmas.shape[-1] <= 1:
        return sigmas, True

    ranged_sigmas = sigmas
    if end_at_step < (ranged_sigmas.shape[-1] - 1):
        ranged_sigmas = ranged_sigmas[: end_at_step + 1]
        if force_full_denoise:
            ranged_sigmas = ranged_sigmas.clone()
            ranged_sigmas[-1] = 0

    if start_at_step < (ranged_sigmas.shape[-1] - 1):
        ranged_sigmas = ranged_sigmas[start_at_step:]
    else:
        return ranged_sigmas, True

    return ranged_sigmas, False


def generate_oneko_noise_for_tensor(tensor, seed, noise_type="gaussian", noise_strength=1.0):
    return _prepare_noise(tensor, seed, None, tensor.device, noise_type, noise_strength)


def _resolve_noise_device(requested_device):
    if requested_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _prepare_noise(latent_image, seed, noise_inds=None, device="cpu", noise_type="gaussian", noise_strength=1.0):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    if latent_image.is_nested:
        noises = [
            _prepare_noise_inner(tensor, generator, noise_inds, device, noise_type, noise_strength)
            for tensor in latent_image.unbind()
        ]
        return comfy.nested_tensor.NestedTensor(noises)

    return _prepare_noise_inner(latent_image, generator, noise_inds, device, noise_type, noise_strength)


def _prepare_noise_inner(latent_image, generator, noise_inds, device, noise_type, noise_strength):
    if noise_inds is None:
        return _generate_noise_by_type(latent_image.size(), latent_image, generator, device, noise_type, noise_strength)

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for index in range(unique_inds[-1] + 1):
        noise = _generate_noise_by_type(
            [1] + list(latent_image.size())[1:],
            latent_image,
            generator,
            device,
            noise_type,
            noise_strength,
        )
        if index in unique_inds:
            noises.append(noise)
    noises = [noises[index] for index in inverse]
    return torch.cat(noises, axis=0)


def _generate_noise_by_type(shape, latent_image, generator, device, noise_type, noise_strength):
    if float(noise_strength) == 0.0:
        return torch.zeros(shape, dtype=latent_image.dtype, layout=latent_image.layout, device=device)

    if noise_type == "gaussian":
        noise = _randn_for_latent(shape, latent_image, generator, device)
        return _apply_noise_strength(noise, noise_strength)

    if noise_type == "uniform":
        noise = _uniform_noise(shape, latent_image, generator, device)
    elif noise_type == "laplacian":
        noise = _laplacian_noise(shape, latent_image, generator, device)
    elif noise_type in SPECTRAL_NOISE_ALPHA:
        noise = _spectral_noise(shape, latent_image, generator, device, SPECTRAL_NOISE_ALPHA[noise_type])
    elif noise_type == "pyramid":
        noise = _pyramid_noise(shape, latent_image, generator, device)
    elif noise_type == "studentt":
        noise = _studentt_noise(shape, latent_image, generator, device)
    elif noise_type == "white":
        noise = _power_law_noise(shape, latent_image, generator, device, alpha=0.0, use_sign=True)
    elif noise_type == "grey":
        noise = _power_law_noise(shape, latent_image, generator, device, alpha=0.0, use_sign=False)
    elif noise_type == "velvet":
        noise = _power_law_noise(shape, latent_image, generator, device, alpha=1.0, use_sign=True)
    elif noise_type == "green_test":
        noise = _green_test_noise(shape, latent_image, generator, device)
    elif noise_type == "highres_pyramid":
        noise = _highres_pyramid_noise(shape, latent_image, generator, device)
    elif noise_type == "pyramid_discount5":
        noise = _pyramid_noise(shape, latent_image, generator, device, discount=0.5)
    elif noise_type == "pyramid_mix":
        noise = _pyramid_mix_noise(shape, latent_image, generator, device)
    elif noise_type == "rainbow_mild":
        noise = _rainbow_noise(shape, latent_image, generator, device, intensity="mild")
    elif noise_type == "rainbow_intense":
        noise = _rainbow_noise(shape, latent_image, generator, device, intensity="intense")
    elif noise_type == "wavelet":
        noise = _wavelet_noise(shape, latent_image, generator, device)
    elif noise_type == "perlin":
        noise = _perlin_like_noise(shape, latent_image, generator, device)
    else:
        noise = _randn_for_latent(shape, latent_image, generator, device)

    noise = _normalize_tensor(noise)
    noise = _apply_noise_strength(noise, noise_strength)
    return noise.to(dtype=latent_image.dtype)


def _prepare_composite_noise(
    latent_image,
    seed,
    noise_inds=None,
    device="cpu",
    family="illustrious",
    profile="balanced",
    noise_strength=1.0,
    detail_bias=0.35,
):
    if float(noise_strength) == 0.0:
        return _zeros_like_noise(latent_image, device)

    recipe = _composite_recipe(family, profile, detail_bias)
    components = []
    for index, (noise_type, weight) in enumerate(recipe):
        component_seed = _derive_component_seed(seed, family, profile, noise_type, index)
        component = _component_noise(latent_image, component_seed, noise_inds, device, noise_type)
        components.append((component, weight))

    combined = _combine_components(components)
    combined = _normalize_noise_like(combined)
    combined = _scale_noise_like(combined, noise_strength)
    return _cast_noise_like(combined, latent_image)


def _composite_recipe(family, profile, detail_bias):
    if family == "pony_v7":
        return _pony_v7_recipe(profile, detail_bias)
    return _illustrious_recipe(profile, detail_bias)


def _illustrious_recipe(variation_mode, detail_bias):
    detail_bias = max(0.0, min(1.0, float(detail_bias)))
    low = 1.0 - detail_bias
    high = detail_bias

    if variation_mode == "texture":
        return [
            ("gaussian", 0.42),
            ("blue", 0.22 + 0.34 * high),
            ("violet", 0.10 + 0.25 * high),
            ("pyramid", 0.28 + 0.22 * low),
        ]
    if variation_mode == "composition":
        return [
            ("gaussian", 0.38),
            ("pink", 0.25 + 0.30 * low),
            ("brown", 0.12 + 0.28 * low),
            ("pyramid", 0.30 + 0.20 * low),
            ("blue", 0.10 * high),
        ]
    if variation_mode == "wild":
        return [
            ("gaussian", 0.35),
            ("pyramid", 0.32 + 0.20 * low),
            ("pink", 0.22 + 0.18 * low),
            ("blue", 0.18 + 0.28 * high),
            ("violet", 0.08 + 0.18 * high),
        ]

    return [
        ("gaussian", 0.52),
        ("pyramid", 0.26 + 0.22 * low),
        ("pink", 0.18 + 0.18 * low),
        ("blue", 0.12 * high),
    ]


def _pony_v7_recipe(v7_profile, detail_bias):
    detail_bias = max(0.0, min(1.0, float(detail_bias)))
    low = 1.0 - detail_bias
    high = detail_bias

    if v7_profile == "stage2_violet":
        return [
            ("gaussian", 0.48),
            ("violet", 0.22 + 0.34 * high),
            ("blue", 0.08 + 0.18 * high),
            ("pyramid", 0.20 + 0.14 * low),
        ]
    if v7_profile == "balanced":
        return [
            ("gaussian", 0.52),
            ("pyramid", 0.24 + 0.20 * low),
            ("pink", 0.16 + 0.18 * low),
            ("blue", 0.08 + 0.12 * high),
        ]
    if v7_profile == "soft":
        return [
            ("gaussian", 0.58),
            ("pyramid", 0.24 + 0.28 * low),
            ("pink", 0.18 + 0.24 * low),
        ]
    if v7_profile == "graphic":
        return [
            ("gaussian", 0.46),
            ("blue", 0.20 + 0.30 * high),
            ("violet", 0.16 + 0.28 * high),
            ("pyramid", 0.18 + 0.12 * low),
        ]

    return [("gaussian", 1.0)]


def _component_noise(latent_image, seed, noise_inds, device, noise_type):
    if latent_image.is_nested:
        noises = [
            _component_noise(tensor, seed, noise_inds, device, noise_type)
            for tensor in latent_image.unbind()
        ]
        return comfy.nested_tensor.NestedTensor(noises)

    latent_ref = torch.empty(
        latent_image.shape,
        dtype=torch.float32,
        layout=latent_image.layout,
        device=latent_image.device,
    )
    return _prepare_noise(latent_ref, seed, noise_inds, device, noise_type, 1.0).float()


def _combine_components(components):
    first_noise = components[0][0]
    if _is_nested_noise(first_noise):
        nested = []
        unbound_components = [(noise.unbind(), weight) for noise, weight in components]
        for sample_index in range(len(unbound_components[0][0])):
            combined = None
            for noises, weight in unbound_components:
                current = noises[sample_index].float() * float(weight)
                combined = current if combined is None else combined + current
            nested.append(combined)
        return comfy.nested_tensor.NestedTensor(nested)

    combined = None
    for noise, weight in components:
        current = noise.float() * float(weight)
        combined = current if combined is None else combined + current
    return combined


def _normalize_noise_like(noise):
    if _is_nested_noise(noise):
        return comfy.nested_tensor.NestedTensor([_normalize_tensor(tensor) for tensor in noise.unbind()])
    return _normalize_tensor(noise)


def _normalize_tensor(tensor):
    tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
    tensor = tensor - tensor.mean()
    std = tensor.std()
    if not torch.isfinite(std) or std <= 1e-8:
        return torch.zeros_like(tensor)
    return tensor / std


def _scale_noise_like(noise, strength):
    strength = float(strength)
    if strength == 1.0:
        return noise
    if _is_nested_noise(noise):
        return comfy.nested_tensor.NestedTensor([tensor * strength for tensor in noise.unbind()])
    return noise * strength


def _cast_noise_like(noise, latent_image):
    if _is_nested_noise(noise):
        latent_tensors = latent_image.unbind()
        return comfy.nested_tensor.NestedTensor(
            [
                tensor.to(dtype=latent_tensors[index].dtype, device=tensor.device)
                for index, tensor in enumerate(noise.unbind())
            ]
        )
    return noise.to(dtype=latent_image.dtype)


def _zeros_like_noise(latent_image, device):
    if latent_image.is_nested:
        return comfy.nested_tensor.NestedTensor(
            [
                torch.zeros(tensor.shape, dtype=tensor.dtype, layout=tensor.layout, device=device)
                for tensor in latent_image.unbind()
            ]
        )
    return torch.zeros(
        latent_image.shape,
        dtype=latent_image.dtype,
        layout=latent_image.layout,
        device=device,
    )


def _is_nested_noise(noise):
    return getattr(noise, "is_nested", False)


def _derive_component_seed(base_seed, family, profile, noise_type, index):
    person = b"NukunV7" if family == "pony_v7" else b"NukunIL"
    digest = hashlib.blake2b(digest_size=8, person=person)
    digest.update(struct.pack("<Q", int(base_seed) & MAX_SEED))
    digest.update(profile.encode("utf-8"))
    digest.update(b"\0")
    digest.update(noise_type.encode("utf-8"))
    digest.update(struct.pack("<I", int(index)))
    return int.from_bytes(digest.digest(), "little", signed=False)


def _randn_for_latent(shape, latent_image, generator, device):
    if torch.device(device).type == "cpu":
        return torch.randn(
            shape,
            dtype=torch.float32,
            layout=latent_image.layout,
            generator=generator,
            device="cpu",
        ).to(dtype=latent_image.dtype)

    return torch.randn(
        shape,
        dtype=latent_image.dtype,
        layout=latent_image.layout,
        generator=generator,
        device=device,
    )


def _uniform_noise(shape, latent_image, generator, device):
    noise = torch.rand(
        shape,
        dtype=torch.float32,
        layout=latent_image.layout,
        generator=generator,
        device=device,
    )
    return (noise - 0.5) * 3.4641016151377544


def _laplacian_noise(shape, latent_image, generator, device):
    uniform = torch.rand(
        shape,
        dtype=torch.float32,
        layout=latent_image.layout,
        generator=generator,
        device=device,
    ) - 0.5
    uniform = uniform.clamp(min=-0.499999, max=0.499999)
    return -torch.sign(uniform) * torch.log1p(-2.0 * uniform.abs()) / 1.4142135623730951


def _spectral_noise(shape, latent_image, generator, device, alpha):
    if len(shape) < 4:
        return _randn_float32(shape, latent_image, generator, device)

    noise = _randn_float32(shape, latent_image, generator, device)
    height = shape[-2]
    width = shape[-1]
    y_freq = torch.fft.fftfreq(height, device=device, dtype=torch.float32)
    x_freq = torch.fft.fftfreq(width, device=device, dtype=torch.float32)
    freq = torch.sqrt(y_freq[:, None] ** 2 + x_freq[None, :] ** 2).clamp(min=1e-6)
    scale = torch.pow(freq, -alpha / 2.0)
    scale[0, 0] = 0.0
    scale = scale.reshape((1,) * (len(shape) - 2) + (height, width))
    noise_fft = torch.fft.fftn(noise, dim=(-2, -1))
    return torch.fft.ifftn(noise_fft * scale, dim=(-2, -1)).real


def _studentt_noise(shape, latent_image, generator, device):
    normal = _randn_float32(shape, latent_image, generator, device)
    chi = sum(_randn_float32(shape, latent_image, generator, device).square() for _ in range(2))
    noise = normal / torch.sqrt((chi / 2.0).clamp(min=1e-6))

    if noise.ndim > 1:
        flat = noise.flatten(start_dim=1).abs()
        quantile = torch.quantile(flat, 0.75, dim=-1)
        quantile = quantile.reshape(*quantile.shape, *((1,) * (noise.ndim - quantile.ndim)))
        noise = noise.clamp(-quantile, quantile)

    return torch.sign(noise) * torch.sqrt(noise.abs())


def _power_law_noise(shape, latent_image, generator, device, alpha=0.0, use_sign=False):
    noise = _randn_float32(shape, latent_image, generator, device)
    modulation = noise.abs().pow(float(alpha))
    result = torch.sign(noise) if use_sign else noise
    result = result * modulation

    if len(shape) >= 3:
        dims = tuple(range(max(1, len(shape) - 3), len(shape)))
        denom = result.abs().amax(dim=dims, keepdim=True).clamp(min=1e-6)
        result = result / denom

    return result


def _green_test_noise(shape, latent_image, generator, device):
    if len(shape) < 4:
        return _randn_float32(shape, latent_image, generator, device)

    noise = _randn_float32(shape, latent_image, generator, device)
    height = int(shape[-2])
    width = int(shape[-1])
    y_freq = torch.fft.fftfreq(height, device=device, dtype=torch.float32).square()
    x_freq = torch.fft.fftfreq(width, device=device, dtype=torch.float32).square()
    power = torch.sqrt((y_freq[:, None] + x_freq[None, :]).clamp(min=1e-8))
    power[0, 0] = 1.0
    power = power.reshape((1,) * (len(shape) - 2) + (height, width))
    filtered = torch.fft.ifft2(torch.fft.fft2(noise, dim=(-2, -1)) / torch.sqrt(power), dim=(-2, -1)).real
    return filtered


def _pyramid_noise(shape, latent_image, generator, device, discount=0.7, levels=5, mode="bilinear"):
    if len(shape) < 4:
        return _randn_float32(shape, latent_image, generator, device)

    base_shape = list(shape)
    height = int(base_shape[-2])
    width = int(base_shape[-1])
    noise = _randn_float32(shape, latent_image, generator, device)

    for level in range(1, int(levels) + 1):
        scale = 2 ** level
        low_h = max(1, height // scale)
        low_w = max(1, width // scale)
        if low_h == height and low_w == width:
            continue

        low_shape = base_shape[:-2] + [low_h, low_w]
        low_noise = _randn_float32(low_shape, latent_image, generator, device)
        upsampled = _resize_last_two_dims(low_noise, height, width, mode=mode)
        noise = noise + upsampled * (float(discount) ** level)

        if low_h == 1 and low_w == 1:
            break

    return noise


def _highres_pyramid_noise(shape, latent_image, generator, device):
    if len(shape) < 4:
        return _randn_float32(shape, latent_image, generator, device)

    base_shape = list(shape)
    height = int(base_shape[-2])
    width = int(base_shape[-1])
    noise = _uniform_noise(shape, latent_image, generator, device)
    discount = 0.7

    for level in range(1, 5):
        scale = level + 1
        high_h = min(height * scale, 1024)
        high_w = min(width * scale, 1024)
        high_shape = base_shape[:-2] + [high_h, high_w]
        high_noise = _randn_float32(high_shape, latent_image, generator, device)
        downsampled = _resize_last_two_dims(high_noise, height, width, mode="bilinear")
        noise = noise + downsampled * (discount ** level)

    return noise


def _pyramid_mix_noise(shape, latent_image, generator, device):
    first = _pyramid_noise(shape, latent_image, generator, device, discount=0.6, levels=5)
    second = _pyramid_noise(shape, latent_image, generator, device, discount=0.6, levels=5)
    return first * 0.2 - second * 0.8


def _rainbow_noise(shape, latent_image, generator, device, intensity="mild"):
    green = _green_test_noise(shape, latent_image, generator, device)
    perlin = _perlin_like_noise(shape, latent_image, generator, device)
    gaussian = _randn_float32(shape, latent_image, generator, device)

    if intensity == "intense":
        return green * 0.65 + perlin * 0.25 + gaussian * 0.10

    return green * 0.35 + perlin * 0.40 + gaussian * 0.25


def _wavelet_noise(shape, latent_image, generator, device):
    if len(shape) < 4:
        return _randn_float32(shape, latent_image, generator, device)

    height = int(shape[-2])
    width = int(shape[-1])
    result = torch.zeros(shape, dtype=torch.float32, layout=latent_image.layout, device=device)
    amplitude = 1.0

    for octave in range(4):
        raw = _randn_float32(shape, latent_image, generator, device)
        factor = 2 ** (octave + 1)
        low_h = max(1, height // factor)
        low_w = max(1, width // factor)
        low = _resize_last_two_dims(raw, low_h, low_w, mode="bilinear")
        blurred = _resize_last_two_dims(low, height, width, mode="bilinear")
        result = result + (raw - blurred) * amplitude
        amplitude *= 0.5

    return result


def _perlin_like_noise(shape, latent_image, generator, device):
    if len(shape) < 4:
        return _randn_float32(shape, latent_image, generator, device)

    base_shape = list(shape)
    height = int(base_shape[-2])
    width = int(base_shape[-1])
    noise = torch.zeros(shape, dtype=torch.float32, layout=latent_image.layout, device=device)
    amplitude = 1.0

    for octave in range(4):
        grid_h = min(height, max(2, 2 ** (octave + 1)))
        grid_w = min(width, max(2, 2 ** (octave + 1)))
        grid_shape = base_shape[:-2] + [grid_h, grid_w]
        grid = _randn_float32(grid_shape, latent_image, generator, device)
        smooth = _resize_last_two_dims(grid, height, width, mode="bicubic")
        noise = noise + smooth * amplitude
        amplitude *= 0.5

    return noise


def _resize_last_two_dims(tensor, height, width, mode):
    original_shape = tensor.shape
    flat = tensor.reshape(-1, 1, original_shape[-2], original_shape[-1])
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        resized = F.interpolate(flat, size=(height, width), mode=mode, align_corners=False)
    else:
        resized = F.interpolate(flat, size=(height, width), mode=mode)
    return resized.reshape(*original_shape[:-2], height, width)


def _randn_float32(shape, latent_image, generator, device):
    return torch.randn(
        shape,
        dtype=torch.float32,
        layout=latent_image.layout,
        generator=generator,
        device=device,
    )


def _apply_noise_strength(noise, noise_strength):
    noise_strength = float(noise_strength)
    if noise_strength == 1.0:
        return noise
    return noise * noise_strength
