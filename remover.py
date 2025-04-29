import kornia.filters
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from spandrel import ModelLoader
from torch import Tensor


def mask_unsqueeze(mask: Tensor):
    if len(mask.shape) == 3:  # BHW -> B1HW
        mask = mask.unsqueeze(1)
    elif len(mask.shape) == 2:  # HW -> B1HW
        mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


def to_bchw(image: Tensor, mask: Tensor | None = None):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2)  # BHWC -> BCHW
    if mask is not None:
        mask = mask_unsqueeze(mask)
        if image.shape[2:] != mask.shape[2:]:
            raise ValueError(
                f"Image and mask must be the same size. {image.shape[2:]} != {mask.shape[2:]}"
            )
    return image, mask


def to_bhwc(image: Tensor):
    return image.permute(0, 2, 3, 1)  # BCHW -> BHWC


def mask_floor(mask: Tensor, threshold: float = 0.99):
    return (mask >= threshold).to(mask.dtype)


# torch pad does not support padding greater than image size with "reflect" mode
def pad_reflect_once(x: Tensor, original_padding: tuple[int, int, int, int]):
    _, _, h, w = x.shape
    padding = np.array(original_padding)
    size = np.array([w, w, h, h])

    initial_padding = np.minimum(padding, size - 1)
    additional_padding = padding - initial_padding

    x = F.pad(x, tuple(initial_padding), mode="reflect")
    if np.any(additional_padding > 0):
        x = F.pad(x, tuple(additional_padding), mode="constant")
    return x


def resize_square(image: Tensor, mask: Tensor, size: int):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = 0, 0, w
    if w == size and h == size:
        return image, mask, (pad_w, pad_h, prev_size)

    if w < h:
        pad_w = h - w
        prev_size = h
    elif h < w:
        pad_h = w - h
        prev_size = w
    image = pad_reflect_once(image, (0, pad_w, 0, pad_h))
    mask = pad_reflect_once(mask, (0, pad_w, 0, pad_h))

    if image.shape[-1] != size:
        image = F.interpolate(image, size=size, mode="nearest-exact")
        mask = F.interpolate(mask, size=size, mode="nearest-exact")

    return image, mask, (pad_w, pad_h, prev_size)


def undo_resize_square(image: Tensor, original_size: tuple[int, int, int]):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = original_size
    if prev_size != w or prev_size != h:
        image = F.interpolate(image, size=prev_size, mode="bilinear")
    return image[:, :, 0 : prev_size - pad_h, 0 : prev_size - pad_w]


def gaussian_blur(image: Tensor, radius: int, sigma: float = 0):
    if sigma <= 0:
        sigma = 0.3 * (radius - 1) + 0.8
    return kornia.filters.gaussian_blur2d(image, (radius, radius), (sigma, sigma))


def binary_erosion(mask: Tensor, radius: int):
    kernel = torch.ones(1, 1, radius * 2 + 1, radius * 2 + 1, device=mask.device)
    mask = F.pad(mask, (radius, radius, radius, radius), mode="constant", value=1)
    mask = F.conv2d(mask, kernel, groups=1)
    mask = (mask == kernel.numel()).to(mask.dtype)
    return mask


def binary_dilation(mask: Tensor, radius: int):
    kernel = torch.ones(1, radius * 2 + 1, device=mask.device)
    mask = kornia.filters.filter2d_separable(
        mask, kernel, kernel, border_type="constant"
    )
    mask = (mask > 0).to(mask.dtype)
    return mask


def make_odd(x):
    if x > 0 and x % 2 == 0:
        return x + 1
    return x


def load_model(model_file, device):
    sd = torch.jit.load(model_file, map_location=device).state_dict()
    model = ModelLoader().load_from_state_dict(sd)
    model = model.eval()
    return model


def load_image_mask(image, mask):
    image = torch.from_numpy(image) / 255
    mask = torch.from_numpy(mask[..., 0]) / 255

    # image: BHWC, mask: BHW
    return image.unsqueeze(0), mask.unsqueeze(0)


def pre_process(image: torch.Tensor, mask: torch.Tensor, device):
    # BHWC in, BCHW out
    mask = expand_mask(mask, 5, False)
    image = fill(image, mask, "telea", 0)
    image = blur(image, mask, 255, 0)
    image, mask = to_bchw(image, mask)
    image = image.to(device)
    mask = mask.to(device)
    return image, mask


def expand_mask(mask: torch.Tensor, expand: int, tapered_corners: bool):
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c], [1, 1, 1], [c, 1, c]])
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in mask:
        output = m.numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)
    return torch.stack(out, dim=0)


def fill(image: torch.Tensor, mask: torch.Tensor, fill: str, falloff: int):
    image = image.detach().clone()
    alpha = mask_unsqueeze(mask_floor(mask))
    assert alpha.shape[0] == image.shape[0], "Image and mask batch size does not match"

    falloff = make_odd(falloff)
    if falloff > 0:
        erosion = binary_erosion(alpha, falloff)
        alpha = alpha * gaussian_blur(erosion, falloff)

    if fill == "neutral":
        m = (1.0 - alpha).squeeze(1)
        for i in range(3):
            image[:, :, :, i] -= 0.5
            image[:, :, :, i] *= m
            image[:, :, :, i] += 0.5
    else:
        import cv2

        method = cv2.INPAINT_TELEA if fill == "telea" else cv2.INPAINT_NS
        for slice, alpha_slice in zip(image, alpha):
            alpha_np = alpha_slice.squeeze().cpu().numpy()
            alpha_bc = alpha_np.reshape(*alpha_np.shape, 1)
            image_np = slice.cpu().numpy()
            filled_np = cv2.inpaint(
                (255.0 * image_np).astype(np.uint8),
                (255.0 * alpha_np).astype(np.uint8),
                3,
                method,
            )
            filled_np = filled_np.astype(np.float32) / 255.0
            filled_np = image_np * (1.0 - alpha_bc) + filled_np * alpha_bc
            slice.copy_(torch.from_numpy(filled_np))

    return image


def blur(image: torch.Tensor, mask: torch.Tensor, blur: int, falloff: int):
    blur = make_odd(blur)
    falloff = min(make_odd(falloff), blur - 2)
    image, mask = to_bchw(image, mask)  # BCHW

    original = image.clone()
    alpha = mask_floor(mask)
    if falloff > 0:
        erosion = binary_erosion(alpha, falloff)
        alpha = alpha * gaussian_blur(erosion, falloff)
    alpha = alpha.expand(-1, 3, -1, -1)

    image = gaussian_blur(image, blur)
    image = original + (image - original) * alpha
    return to_bhwc(image)  # BHWC


def inpaint(inpaint_model, image: torch.Tensor, mask: torch.Tensor, seed: int):
    required_size = 256

    batch_size = image.shape[0]
    if mask.shape[0] != batch_size:
        mask = mask[0].unsqueeze(0).repeat(batch_size, 1, 1, 1)

    device = image.device
    inpaint_model = inpaint_model.to(device)
    batch_image = []

    for i in range(batch_size):
        work_image, work_mask = image[i].unsqueeze(0), mask[i].unsqueeze(0)
        work_image, work_mask, original_size = resize_square(
            work_image, work_mask, required_size
        )
        work_mask = mask_floor(work_mask)

        torch.manual_seed(seed)
        work_image = inpaint_model(
            work_image.to(device).float(), work_mask.to(device).float()
        )

        work_image.to(device)
        work_image = undo_resize_square(work_image.to(device), original_size)
        work_image = image[i] + (work_image - image[i]) * mask_floor(mask[i])

        batch_image.append(work_image)

    # inpaint_model.cpu()
    result = torch.cat(batch_image, dim=0)
    return result
