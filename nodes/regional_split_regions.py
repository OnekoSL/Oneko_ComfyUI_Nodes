
import node_helpers
import torch
import torch.nn.functional as F


class OnekoSplitMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8}),
                "region_count": ("INT", {"default": 2, "min": 2, "max": 3, "step": 1}),
                "orientation": (["horizontal", "vertical"],),
                "split_1": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01}),
                "split_2": ("FLOAT", {"default": 0.67, "min": 0.01, "max": 0.99, "step": 0.01}),
                "overlap": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.25, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK")
    RETURN_NAMES = ("mask_1", "mask_2", "mask_3")
    FUNCTION = "build"
    CATEGORY = "Oneko/04 Regional/Masks"
    DESCRIPTION = "Builds simple 2- or 3-way split masks from width and height."

    def build(self, width, height, region_count, orientation, split_1, split_2, overlap):
        region_count = 3 if region_count >= 3 else 2
        masks = _make_split_masks(region_count, orientation, width, height, split_1, split_2, overlap)
        return (masks[0], masks[1], masks[2])


class OnekoRegionalRectMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8}),
                "region_count": ("INT", {"default": 3, "min": 2, "max": 3, "step": 1}),
                "x_1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "y_1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "w_1": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "h_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "x_2": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "y_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "w_2": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05}),
                "h_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "x_3": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "y_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "w_3": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "h_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "soft_edge": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK")
    RETURN_NAMES = ("mask_1", "mask_2", "mask_3")
    FUNCTION = "build"
    CATEGORY = "Oneko/04 Regional/Masks"
    DESCRIPTION = "Builds 2/3 freely placed rectangular region masks from percentage coordinates."

    def build(
        self,
        width,
        height,
        region_count,
        x_1,
        y_1,
        w_1,
        h_1,
        x_2,
        y_2,
        w_2,
        h_2,
        x_3,
        y_3,
        w_3,
        h_3,
        soft_edge,
    ):
        region_count = 3 if region_count >= 3 else 2
        rects = [(x_1, y_1, w_1, h_1), (x_2, y_2, w_2, h_2), (x_3, y_3, w_3, h_3)]
        masks = _make_rect_masks(region_count, width, height, rects, soft_edge)
        return (masks[0], masks[1], masks[2])


class OnekoNativeRegionalSplitConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
                "width": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8}),
                "region_count": ("INT", {"default": 2, "min": 2, "max": 3, "step": 1}),
                "orientation": (["horizontal", "vertical"],),
                "split_1": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01}),
                "split_2": ("FLOAT", {"default": 0.67, "min": 0.01, "max": 0.99, "step": 0.01}),
                "overlap": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.25, "step": 0.01}),
                "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "strength_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            },
            "optional": {
                "conditioning_3": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("conditioning", "mask_1", "mask_2", "mask_3")
    FUNCTION = "apply"
    CATEGORY = "Oneko/04 Regional/Apply"
    DESCRIPTION = "Builds 2/3 split masks and combines masked conditionings using ComfyUI core metadata."

    def apply(
        self,
        conditioning_1,
        conditioning_2,
        width,
        height,
        region_count,
        orientation,
        split_1,
        split_2,
        overlap,
        strength_1,
        strength_2,
        strength_3,
        set_cond_area,
        conditioning_3=None,
    ):
        region_count = 3 if region_count >= 3 else 2
        conditionings = [conditioning_1, conditioning_2, conditioning_3]
        strengths = [strength_1, strength_2, strength_3]
        if region_count == 3 and conditioning_3 is None:
            raise ValueError("conditioning_3 is required when region_count is 3.")

        masks = _make_split_masks(region_count, orientation, width, height, split_1, split_2, overlap)
        conditioning = _combine_masked_conditionings(conditionings, masks, strengths, region_count, set_cond_area)
        return (conditioning, masks[0], masks[1], masks[2])


class OnekoNativeRegionalRectConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
                "width": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8}),
                "region_count": ("INT", {"default": 3, "min": 2, "max": 3, "step": 1}),
                "x_1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "y_1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "w_1": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "h_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "x_2": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "y_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "w_2": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05}),
                "h_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "x_3": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "y_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "w_3": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "h_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "soft_edge": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "strength_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            },
            "optional": {
                "conditioning_3": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("conditioning", "mask_1", "mask_2", "mask_3")
    FUNCTION = "apply"
    CATEGORY = "Oneko/04 Regional/Apply"
    DESCRIPTION = "Builds 2/3 rectangular masks and combines masked conditionings using ComfyUI core metadata."

    def apply(
        self,
        conditioning_1,
        conditioning_2,
        width,
        height,
        region_count,
        x_1,
        y_1,
        w_1,
        h_1,
        x_2,
        y_2,
        w_2,
        h_2,
        x_3,
        y_3,
        w_3,
        h_3,
        soft_edge,
        strength_1,
        strength_2,
        strength_3,
        set_cond_area,
        conditioning_3=None,
    ):
        region_count = 3 if region_count >= 3 else 2
        conditionings = [conditioning_1, conditioning_2, conditioning_3]
        strengths = [strength_1, strength_2, strength_3]
        if region_count == 3 and conditioning_3 is None:
            raise ValueError("conditioning_3 is required when region_count is 3.")

        rects = [(x_1, y_1, w_1, h_1), (x_2, y_2, w_2, h_2), (x_3, y_3, w_3, h_3)]
        masks = _make_rect_masks(region_count, width, height, rects, soft_edge)
        masks = _ensure_dense_diffusion_coverage(masks, region_count)
        conditioning = _combine_masked_conditionings(conditionings, masks, strengths, region_count, set_cond_area)
        return (conditioning, masks[0], masks[1], masks[2])


def _make_split_masks(region_count, orientation, width, height, split_1, split_2, overlap):
    width = max(8, int(width))
    height = max(8, int(height))
    region_count = 3 if region_count >= 3 else 2
    masks = [torch.zeros((1, height, width), dtype=torch.float32) for _ in range(3)]
    axis_size = width if orientation == "horizontal" else height
    overlap_px = max(0, int(round(axis_size * float(overlap))))

    if region_count == 2:
        boundaries = [0, _to_boundary(split_1, axis_size), axis_size]
    else:
        first = _to_boundary(split_1, axis_size)
        second = _to_boundary(split_2, axis_size)
        if first > second:
            first, second = second, first
        if first == second:
            first = max(1, min(axis_size - 2, first))
            second = first + 1
        boundaries = [0, first, second, axis_size]

    for i in range(region_count):
        start = max(0, boundaries[i] - overlap_px)
        end = min(axis_size, boundaries[i + 1] + overlap_px)
        if orientation == "horizontal":
            masks[i][:, :, start:end] = 1.0
        else:
            masks[i][:, start:end, :] = 1.0

    return masks


def _make_rect_masks(region_count, width, height, rects, soft_edge):
    width = max(8, int(width))
    height = max(8, int(height))
    region_count = 3 if region_count >= 3 else 2
    masks = [torch.zeros((1, height, width), dtype=torch.float32) for _ in range(3)]

    for index in range(region_count):
        masks[index] = _make_rect_mask(width, height, rects[index], soft_edge)

    return masks


def _make_rect_mask(width, height, rect, soft_edge):
    x, y, w, h = (_clamp01(value) for value in rect)
    x0 = min(width, max(0, int(round(x * width))))
    y0 = min(height, max(0, int(round(y * height))))
    x1 = min(width, max(x0, int(round((x + w) * width))))
    y1 = min(height, max(y0, int(round((y + h) * height))))
    mask = torch.zeros((1, height, width), dtype=torch.float32)
    if x1 <= x0 or y1 <= y0:
        return mask

    mask[:, y0:y1, x0:x1] = 1.0
    soft_edge = max(0, int(soft_edge))
    if soft_edge <= 0:
        return mask

    kernel_size = min(soft_edge * 2 + 1, max(3, min(width, height)))
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size < 3:
        return mask

    padded = F.pad(mask.unsqueeze(1), (kernel_size // 2,) * 4, mode="replicate")
    blurred = F.avg_pool2d(padded, kernel_size=kernel_size, stride=1).squeeze(1)
    if blurred.max() > 0:
        blurred = blurred / blurred.max()
    return blurred.clamp(0.0, 1.0)


def _combine_masked_conditionings(conditionings, masks, strengths, region_count, set_cond_area):
    set_area_to_bounds = set_cond_area != "default"
    combined = []
    for index in range(region_count):
        conditioning = conditionings[index]
        if conditioning is None:
            continue
        mask = masks[index]
        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
        combined.extend(
            node_helpers.conditioning_set_values(
                conditioning,
                {
                    "mask": mask,
                    "set_area_to_bounds": set_area_to_bounds,
                    "mask_strength": strengths[index],
                },
            )
        )
    return combined


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _to_boundary(split, axis_size):
    boundary = int(round(axis_size * float(split)))
    return max(1, min(axis_size - 1, boundary))


NODE_CLASS_MAPPINGS = {
    "OnekoSplitMasks": OnekoSplitMasks,
    "OnekoRegionalRectMasks": OnekoRegionalRectMasks,
    "OnekoNativeRegionalSplitConditioning": OnekoNativeRegionalSplitConditioning,
    "OnekoNativeRegionalRectConditioning": OnekoNativeRegionalRectConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoSplitMasks": "Split Masks (Oneko)",
    "OnekoRegionalRectMasks": "Regional Rect Masks (Oneko)",
    "OnekoNativeRegionalSplitConditioning": "Native Regional Split Conditioning (Oneko)",
    "OnekoNativeRegionalRectConditioning": "Native Regional Rect Conditioning (Oneko)"
}
