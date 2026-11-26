import logging

import torch
import torch.nn.functional as F

from comfy_api.latest import io

from .conditioning_context import CONDITIONING_CONTEXT_KEY, conditioning_fingerprint


def _verified_moss_positions(tensor, context, positions):
    mask = context.get("token_mask") if isinstance(context, dict) else None
    if (
        not isinstance(context, dict)
        or context.get("version") != 1
        or context.get("model_profile") != "moss_image"
        or not isinstance(mask, (tuple, list))
        or len(mask) != tensor.shape[1]
        or context.get("token_count") != tensor.shape[1]
        or any(type(value) is not bool for value in mask)
    ):
        detail = context.get("warning") if isinstance(context, dict) else None
        logging.warning("Oneko Conditioning Token Randomize skipped MOSS conditioning without a valid user-token mask. %s",
                        detail or "Use a Oneko MOSS text encoder before this node.")
        return None

    indices = [index for index in range(*positions.indices(tensor.shape[1])) if mask[index]]
    if not indices:
        return None
    if context.get("source_fingerprint") != conditioning_fingerprint(tensor):
        logging.warning("Oneko Conditioning Token Randomize skipped changed MOSS conditioning. "
                        "Place it after a Oneko encoder or protected adjustment, before mixing prompts.")
        return None
    return torch.tensor(indices, dtype=torch.long, device=tensor.device)


class OnekoConditioningTokenRandomize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OnekoConditioningTokenRandomize",
            display_name="Conditioning Token Randomize (Oneko)",
            category="Oneko/03 Conditioning/Transform",
            description=("Mixes encoded token vectors with seeded random directions of the same length. "
                         "Verified Oneko MOSS context protects template and special-token positions."),
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Float.Input("strength", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="0 uses the original token vectors. 1 uses the random vectors."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffff, control_after_generate=True),
                io.Int.Input("start_token", default=0, min=0,
                             tooltip="Zero-based position in the full encoded sequence, including protected positions."),
                io.Int.Input("token_count", default=0, min=0,
                             tooltip="Number of positions to vary. 0 selects through the end of the sequence."),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, conditioning, strength, seed, start_token, token_count):
        if strength == 0:
            return io.NodeOutput(conditioning)

        stop_token = start_token + token_count if token_count else None
        positions = slice(start_token, stop_token)
        output = []
        for entry in conditioning:
            tensor, metadata = entry
            if not torch.is_tensor(tensor) or tensor.ndim != 3 or not tensor.is_floating_point():
                output.append(entry)
                continue

            entry_positions = positions
            protected = CONDITIONING_CONTEXT_KEY in metadata
            if protected:
                context = metadata[CONDITIONING_CONTEXT_KEY]
                entry_positions = _verified_moss_positions(tensor, context, positions)
                if entry_positions is None:
                    output.append(entry)
                    continue

            selected = tensor[:, entry_positions, :]
            if selected.numel() == 0:
                output.append(entry)
                continue

            dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
            vectors = selected.to(dtype=dtype)
            # Restart the stream so equally shaped scheduled entries share directions.
            generator = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(vectors.shape, generator=generator, dtype=dtype, device="cpu").to(tensor.device)
            lengths = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
            random_vectors = F.normalize(noise, dim=-1) * lengths

            replacement = torch.lerp(vectors, random_vectors, strength).to(dtype=tensor.dtype)
            if protected and torch.equal(selected, replacement):
                output.append(entry)
                continue

            edited = tensor.clone()
            edited[:, entry_positions, :] = replacement
            edited_metadata = metadata.copy()
            if protected:
                edited_metadata[CONDITIONING_CONTEXT_KEY] = {
                    **context, "source_fingerprint": conditioning_fingerprint(edited),
                }
            output.append([edited, edited_metadata])

        return io.NodeOutput(output)


NODE_CLASS_MAPPINGS = {"OnekoConditioningTokenRandomize": OnekoConditioningTokenRandomize}
NODE_DISPLAY_NAME_MAPPINGS = {"OnekoConditioningTokenRandomize": "Conditioning Token Randomize (Oneko)"}
