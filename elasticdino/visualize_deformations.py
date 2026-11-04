"""Draw ElasticDino's inference deformations on the input image.

Each upscaling stage runs one or more deformer blocks. A block predicts a
sampling field and `grid_sample`s the feature map through it. Between stages
the features are nearest-neighbor upsampled. This module replays those exact
fields on the RGB image, so the picture moves the same way the features do.

`model(images, return_displacements=True)` returns:

- `displacements`: per stage, per block, tensor `(B, 2, H, W)` added to the
  normalized base grid. Channel 0 is x (width), channel 1 is y (height).
- `sampling_fields`: the grids actually passed to `grid_sample`,
  `(B, H, W, 2)`, in `align_corners=False` coordinates.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from elasticdino.model.elasticdino import ElasticDino, make_base_locations
from elasticdino.training.util import reduce_dimension


def sampling_field(displacements):
  """Rebuild the `grid_sample` field from a `(B, 2, H, W)` displacement."""
  batch, channels, height, width = displacements.shape
  if channels != 2 or height != width:
    raise ValueError(f"Expected displacements (B, 2, H, H), got {tuple(displacements.shape)}")
  base = make_base_locations(batch, height, displacements.dtype, displacements.device)
  base = base.permute(0, 3, 1, 2)
  return (base + displacements).permute(0, 2, 3, 1)


def _middlebury_colorwheel():
  """Baker et al. color wheel: 55 hues, red at the right, white at zero flow."""
  def ramp(n):
    return torch.floor(255 * torch.arange(n) / n)

  ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
  wheel = torch.zeros(ry + yg + gc + cb + bm + mr, 3)
  col = 0
  wheel[col:col + ry, 0] = 255
  wheel[col:col + ry, 1] = ramp(ry)
  col += ry
  wheel[col:col + yg, 0] = 255 - ramp(yg)
  wheel[col:col + yg, 1] = 255
  col += yg
  wheel[col:col + gc, 1] = 255
  wheel[col:col + gc, 2] = ramp(gc)
  col += gc
  wheel[col:col + cb, 1] = 255 - ramp(cb)
  wheel[col:col + cb, 2] = 255
  col += cb
  wheel[col:col + bm, 2] = 255
  wheel[col:col + bm, 0] = ramp(bm)
  col += bm
  wheel[col:col + mr, 2] = 255 - ramp(mr)
  wheel[col:col + mr, 0] = 255
  return wheel / 255.0


_MIDDLEBURY_WHEEL = _middlebury_colorwheel()


def _middlebury_flow_rgb(u, v):
  """Map flow in units of `magnitude_scale` to RGB. Zero flow is white."""
  wheel = _MIDDLEBURY_WHEEL.to(device=u.device, dtype=u.dtype)
  ncols = wheel.shape[0]
  radius = torch.sqrt(u * u + v * v)
  angle = torch.atan2(-v, -u) / math.pi
  fk = (angle + 1) / 2 * (ncols - 1)
  k0 = torch.floor(fk).long().clamp(0, ncols - 1)
  k1 = (k0 + 1) % ncols
  fraction = (fk - k0.to(fk.dtype)).unsqueeze(0)
  color = (1 - fraction) * wheel[k0].permute(2, 0, 1) + fraction * wheel[k1].permute(2, 0, 1)
  radius = radius.unsqueeze(0)
  inside = radius <= 1
  color = torch.where(inside, 1 - radius.clamp(max=1) * (1 - color), color * 0.75)
  return color.clamp(0, 1)


def displacement_to_rgb(displacements, magnitude_scale):
  """Middlebury optical flow colors. Hue is direction, white is no motion."""
  dx = displacements[0].float()
  dy = displacements[1].float()
  scale = max(float(magnitude_scale), 1e-8)
  return _middlebury_flow_rgb(dx / scale, dy / scale)


def _lattice(image):
  """Red horizontal and cyan vertical lines on a dimmed copy of `image` (3, H, W)."""
  vis = image * 0.55
  step = max(2, image.shape[-2] // 8)
  red = torch.tensor([1.0, 0.15, 0.15], device=image.device, dtype=image.dtype).view(3, 1, 1)
  cyan = torch.tensor([0.15, 0.9, 1.0], device=image.device, dtype=image.dtype).view(3, 1, 1)
  vis[:, ::step, :] = red
  vis[:, :, ::step] = cyan
  return vis


def _guidance_image(images, target_size, feature_size):
  """Match ElasticDino: bilinear to `target_size`, then bilinear to the feature grid."""
  target = F.interpolate(images, target_size, mode="bilinear", align_corners=False, antialias=True)
  if feature_size == target_size:
    return target
  return F.interpolate(target, feature_size, mode="bilinear", align_corners=False)


def deformation_frames(images, sampling_fields, target_size, image_index=0):
  """Replay feature warps on one image.

  `sampling_fields` is the structure returned by `ElasticDino.forward`:
  a list of stages, each a list of `(B, H, W, 2)` grids.

  Returns a list of frame dicts in inference order. `kind` is `"start"`,
  `"deform"`, or `"upsample"`.
  """
  if images.ndim != 4 or images.shape[1] != 3:
    raise ValueError(f"Expected images (B, 3, H, W), got {tuple(images.shape)}")
  if not sampling_fields or any(len(stage) == 0 for stage in sampling_fields):
    raise ValueError("sampling_fields is empty; run the model with return_displacements=True")

  index = image_index
  image = images[index:index + 1].detach().float().clamp(0, 1)
  first_size = sampling_fields[0][0].shape[1]
  canvas = _guidance_image(image, target_size, first_size)
  lattice = _lattice(canvas[0]).unsqueeze(0)
  frames = [_frame("start", 0, None, first_size, canvas, lattice, canvas, None)]

  for stage_index, stage_fields in enumerate(sampling_fields):
    size = stage_fields[0].shape[1]
    if canvas.shape[-1] != size:
      canvas = F.interpolate(canvas, size, mode="nearest")
      lattice = F.interpolate(lattice, size, mode="nearest")
    guidance = _guidance_image(image, target_size, size)
    for block_index, field in enumerate(stage_fields):
      grid = field[index:index + 1].detach().float()
      if grid.shape[1] != size or grid.shape[2] != size:
        raise ValueError(f"Inconsistent sampling field shape {tuple(field.shape)} at stage {stage_index}")
      canvas = F.grid_sample(canvas, grid, padding_mode="border", align_corners=False)
      lattice = F.grid_sample(lattice, grid, padding_mode="border", align_corners=False)
      displacement = _displacement_from_field(grid)
      frames.append(_frame(
          "deform", stage_index, block_index, size, canvas, lattice, guidance, displacement,
      ))
    is_last = stage_index == len(sampling_fields) - 1
    if not is_last:
      next_size = sampling_fields[stage_index + 1][0].shape[1]
      canvas = F.interpolate(canvas, next_size, mode="nearest")
      lattice = F.interpolate(lattice, next_size, mode="nearest")
      next_guidance = _guidance_image(image, target_size, next_size)
      frames.append(_frame(
          "upsample", stage_index + 1, None, next_size, canvas, lattice, next_guidance, None,
      ))
  return frames


def _pixel_centers(size, dtype, device):
  """Coordinate where `grid_sample(..., align_corners=False)` reads pixel centers.

  The model adds displacements to `i * 2/size - 1`. The center of pixel i is
  one half-pixel further, at that value plus `1/size`.
  """
  return make_base_locations(1, size, dtype, device) + (1.0 / size)


def _displacement_from_field(field):
  """field is (1, H, W, 2). Offset from each pixel sampling itself."""
  height = field.shape[1]
  centers = _pixel_centers(height, field.dtype, field.device)
  return (field - centers).permute(0, 3, 1, 2)[0]


def _apply_sampling_field(composed, grid):
  """Compose one block grid onto the source map built so far.

  `grid` is the tensor passed to `grid_sample` on the features, so indexing
  the source map with it follows the same pixels. A field of pixel centers
  reads the source stored at those pixels.
  """
  if composed is None:
    return grid
  return F.grid_sample(
      composed.permute(0, 3, 1, 2),
      grid,
      mode="bilinear",
      padding_mode="border",
      align_corners=False,
  ).permute(0, 2, 3, 1)


def iter_composed_fields(sampling_fields, image_index=0):
  """Yield the composed source map at every feature-map step.

  Each item is `(label, composed)` with `composed` shaped `(1, H, W, 2)`.
  The first item is the identity on the backbone grid. Later items are the
  map after each deformer block, in the same order as `iter_feature_maps`.
  Nearest upsamples are applied but not yielded.
  """
  if not sampling_fields or any(len(stage) == 0 for stage in sampling_fields):
    raise ValueError("sampling_fields is empty")
  first = sampling_fields[0][0]
  size = first.shape[1]
  composed = _pixel_centers(size, torch.float32, first.device)
  yield f"{size} input", composed
  for stage in sampling_fields:
    size = stage[0].shape[1]
    n_blocks = len(stage)
    if composed.shape[1] != size:
      composed = F.interpolate(composed.permute(0, 3, 1, 2), (size, size), mode="nearest")
      composed = composed.permute(0, 2, 3, 1)
    for block_index, field in enumerate(stage):
      grid = field[image_index:image_index + 1].detach().float()
      composed = _apply_sampling_field(composed, grid)
      yield f"{size} step {block_index + 1}/{n_blocks}", composed


def compose_sampling_fields(sampling_fields, image_index=0):
  """Fold every block grid into one map from the final pixels back to the input.

  Returns `(1, H, W, 2)` source coordinates in the same normalized frame as
  `grid_sample` (`align_corners=False`). A final pixel samples the input at
  the coordinate stored in that pixel.
  """
  composed = None
  for _, composed in iter_composed_fields(sampling_fields, image_index=image_index):
    pass
  return composed


def _displacement_of(composed):
  """Offset of a composed source map from each pixel sampling itself."""
  return _displacement_from_field(composed)


def iter_composed_displacements(sampling_fields, image_index=0):
  """Yield `(label, displacement)` at every feature-map step. Displacement is `(2, H, W)`."""
  for label, composed in iter_composed_fields(sampling_fields, image_index=image_index):
    yield label, _displacement_of(composed)


def displacement_components_rgb(displacement, scale=None):
  """Pack a `(2, H, W)` displacement into RGB. R is x, G is y, 0.5 is zero."""
  dx = displacement[0].float()
  dy = displacement[1].float()
  if scale is None:
    scale = torch.maximum(dx.abs().max(), dy.abs().max()).clamp(min=1e-8)
  else:
    scale = max(float(scale), 1e-8)
  rgb = torch.empty(3, dx.shape[0], dx.shape[1], dtype=torch.float32, device=dx.device)
  rgb[0] = (0.5 + 0.5 * dx / scale).clamp(0, 1)
  rgb[1] = (0.5 + 0.5 * dy / scale).clamp(0, 1)
  rgb[2] = 0.5
  return rgb, float(scale)


def composed_displacement(sampling_fields, image_index=0):
  """`(2, H, W)` displacement of the composed field at the final grid."""
  displacement = None
  for _, displacement in iter_composed_displacements(sampling_fields, image_index=image_index):
    pass
  return displacement


def render_composed_components(sampling_fields, image_index=0, display_size=512, displacement=None):
  """RGB image of the composed displacement, plus the `(2, H, W)` field."""
  if displacement is None:
    displacement = composed_displacement(sampling_fields, image_index=image_index)
  height = displacement.shape[-1]
  rgb, scale = displacement_components_rgb(displacement)
  image = _to_pil(rgb, max(display_size, height), mode="nearest")
  scale_px = scale * height / 2
  percent = 100.0 * scale / 2
  caption = (
      f"Composed field {height}x{height}. R = x (right positive), G = y (down positive), gray = 0. "
      f"Full red or green = {scale_px:.2f}px ({percent:.2f}% of width)."
  )
  return _captioned(image, caption), displacement.detach(), scale_px


def _labeled_color_wheel(size):
  """Wheel in image coordinates: +x is right, +y is down."""
  wheel = _color_wheel(size)
  margin = 40
  canvas = Image.new("RGB", (size + margin * 2, size + margin * 2), (255, 255, 255))
  canvas.paste(wheel, (margin, margin))
  draw = ImageDraw.Draw(canvas)
  font = _load_font(12)
  draw.text((margin + size / 2 - 8, 4), "up", fill=(20, 20, 20), font=font)
  draw.text((margin + size / 2 - 14, margin + size + 8), "down", fill=(20, 20, 20), font=font)
  draw.text((2, margin + size / 2 - 6), "left", fill=(20, 20, 20), font=font)
  draw.text((margin + size + 4, margin + size / 2 - 6), "right", fill=(20, 20, 20), font=font)
  return canvas


def render_composed_flow(displacement, display_size=512):
  """Color-wheel image of a composed `(2, H, W)` displacement."""
  height = displacement.shape[-1]
  magnitude = displacement.float().norm(dim=0)
  scale = float(magnitude.max().clamp(min=1e-8))
  rgb = displacement_to_rgb(displacement, scale)
  field = _to_pil(rgb, max(display_size, height), mode="nearest")
  wheel = _labeled_color_wheel(min(160, field.height))
  gap = 16
  body_h = max(field.height, wheel.height)
  body = Image.new("RGB", (field.width + gap + wheel.width, body_h), (255, 255, 255))
  body.paste(field, (0, (body_h - field.height) // 2))
  body.paste(wheel, (field.width + gap, (body_h - wheel.height) // 2))
  scale_px = scale * height / 2
  percent = 100.0 * scale / 2
  caption = (
      f"Composed field {height}x{height}, Middlebury flow colors. Hue = direction, white = 0. "
      f"Full saturation = {scale_px:.2f}px ({percent:.2f}% of width)."
  )
  return _captioned(body, caption), scale_px


def _labeled_cell(label, image, panel):
  label_h = 22
  cell = Image.new("RGB", (panel, panel + label_h), (255, 255, 255))
  cell.paste(image.resize((panel, panel)), (0, label_h))
  ImageDraw.Draw(cell).text((4, 3), label, fill=(0, 0, 0), font=_load_font(12))
  return cell


def _hstack(images, gap=6):
  height = max(image.height for image in images)
  width = sum(image.width for image in images) + gap * (len(images) - 1)
  canvas = Image.new("RGB", (width, height), (255, 255, 255))
  x = 0
  for image in images:
    canvas.paste(image, (x, 0))
    x += image.width + gap
  return canvas


def render_composition_progression(sampling_fields, image_index=0, panel=160):
  """One column per feature-map step, showing the composed field so far.

  Both strips use a single scale taken from the strongest step, so a dim
  early column is a smaller displacement than a saturated later column.
  """
  steps = list(iter_composed_displacements(sampling_fields, image_index=image_index))
  component_scale = 1e-8
  flow_scale = 1e-8
  for _, displacement in steps:
    displacement = displacement.float()
    component_scale = max(component_scale, float(torch.maximum(displacement[0].abs().max(), displacement[1].abs().max())))
    flow_scale = max(flow_scale, float(displacement.norm(dim=0).max()))

  component_cells = []
  flow_cells = []
  for label, displacement in steps:
    rgb, _ = displacement_components_rgb(displacement, scale=component_scale)
    component_cells.append(_labeled_cell(label, _to_pil(rgb, panel, mode="nearest"), panel))
    flow_cells.append(_labeled_cell(label, _to_pil(displacement_to_rgb(displacement, flow_scale), panel, mode="nearest"), panel))

  components = _captioned(
      _hstack(component_cells),
      "Composed field after the input and after each deformer block. "
      "R = x (right positive), G = y (down positive), gray = 0. "
      f"Shared scale: full red or green = {100.0 * component_scale / 2:.2f}% of width.",
  )
  flow = _captioned(
      _hstack(flow_cells + [_labeled_color_wheel(min(140, panel))]),
      "Composed field after the input and after each deformer block, color wheel. "
      "Hue = direction, white = 0. "
      f"Shared scale: full saturation = {100.0 * flow_scale / 2:.2f}% of width.",
  )
  return components, flow, steps


def iter_step_displacements(sampling_fields, image_index=0):
  """Yield `(label, displacement)` for each feature-map step, one block at a time.

  The input column is zero: it does not predict a sampling field. Nearest
  upsamples are skipped. Labels match `iter_feature_maps`.
  """
  if not sampling_fields or any(len(stage) == 0 for stage in sampling_fields):
    raise ValueError("sampling_fields is empty")
  first = sampling_fields[0][0]
  size = first.shape[1]
  yield f"{size} input", torch.zeros(2, size, size, dtype=torch.float32, device=first.device)
  for stage in sampling_fields:
    stage_size = stage[0].shape[1]
    n_blocks = len(stage)
    if stage_size != size:
      size = stage_size
    for block_index, field in enumerate(stage):
      grid = field[image_index:image_index + 1].detach().float()
      yield f"{size} step {block_index + 1}/{n_blocks}", _displacement_of(grid)


def render_step_flow(sampling_fields, image_index=0, panel=160):
  """Color-wheel strip of each block's own displacement, without earlier blocks."""
  steps = list(iter_step_displacements(sampling_fields, image_index=image_index))
  flow_scale = 1e-8
  for _, displacement in steps:
    flow_scale = max(flow_scale, float(displacement.float().norm(dim=0).max()))
  cells = [
      _labeled_cell(label, _to_pil(displacement_to_rgb(displacement, flow_scale), panel, mode="nearest"), panel)
      for label, displacement in steps
  ]
  image = _captioned(
      _hstack(cells + [_labeled_color_wheel(min(140, panel))]),
      "This step only, color wheel, after the input and after each deformer block. "
      "Hue = direction, white = 0. "
      f"Shared scale: full saturation = {100.0 * flow_scale / 2:.2f}% of width.",
  )
  return image, steps


def iter_feature_maps(original_features, sampling_fields, image_index=0):
  """Replay the recorded grids on the backbone feature map.

  Yields `(label, features)` with features shaped `(1, C, H, W)`, starting
  from the low-resolution backbone map and ending at the final grid.
  Nearest upsamples are applied but not yielded.
  """
  features = original_features[image_index:image_index + 1].detach().float()
  yield f"{features.shape[-1]} input", features
  for stage in sampling_fields:
    size = stage[0].shape[1]
    n_blocks = len(stage)
    if features.shape[-1] != size:
      features = F.interpolate(features, (size, size), mode="nearest")
    for block_index, field in enumerate(stage):
      grid = field[image_index:image_index + 1].detach().float()
      features = F.grid_sample(features, grid, padding_mode="border", align_corners=False)
      yield f"{size} step {block_index + 1}/{n_blocks}", features


def render_feature_strip(original_features, sampling_fields, image_index=0, panel=160):
  """PCA strip of the feature map after every deformer block.

  The three colors are fit on the input grid and reused for every later map,
  so a color keeps the same meaning from the low-resolution map to the last one.
  """
  steps = list(iter_feature_maps(original_features, sampling_fields, image_index=image_index))
  first = steps[0][1]
  rest = [features for _, features in steps[1:]]
  colored, others = reduce_dimension(first, rest, 3)
  rgb_maps = [colored] + others
  cells = []
  for (label, _), rgb in zip(steps, rgb_maps):
    rgb = torch.nan_to_num(rgb[0], nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    cells.append(_labeled_cell(label, _to_pil(rgb, panel, mode="nearest"), panel))
  image = _captioned(
      _hstack(cells),
      "Feature maps from the backbone grid to the final grid. "
      "PCA is fit on the input map and applied to every later step.",
  )
  return image, [label for label, _ in steps]


def progression_panels(original_features, sampling_fields, panel=160, image_index=0):
  """Bare panels for the feature strip, cumulative flow, and per-step flow.

  Returns three lists of `panel`-sized images in the same step order. There
  are no labels, gaps, captions, or color wheels, so the lists can be stacked
  into one grid.
  """
  flow_steps = list(iter_composed_displacements(sampling_fields, image_index=image_index))
  feature_steps = list(iter_feature_maps(original_features, sampling_fields, image_index=image_index))
  per_steps = list(iter_step_displacements(sampling_fields, image_index=image_index))
  flow_labels = [label for label, _ in flow_steps]
  feature_labels = [label for label, _ in feature_steps]
  per_labels = [label for label, _ in per_steps]
  if not (flow_labels == feature_labels == per_labels):
    raise ValueError(f"Strip steps differ: {flow_labels} vs {feature_labels} vs {per_labels}")

  flow_scale = 1e-8
  for _, displacement in flow_steps:
    flow_scale = max(flow_scale, float(displacement.float().norm(dim=0).max()))
  per_scale = 1e-8
  for _, displacement in per_steps:
    per_scale = max(per_scale, float(displacement.float().norm(dim=0).max()))

  first = feature_steps[0][1]
  rest = [features for _, features in feature_steps[1:]]
  colored, others = reduce_dimension(first, rest, 3)
  feature_panels = []
  for rgb in [colored] + others:
    rgb = torch.nan_to_num(rgb[0], nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    feature_panels.append(_to_pil(rgb, panel, mode="nearest"))
  flow_panels = [
      _to_pil(displacement_to_rgb(displacement, flow_scale), panel, mode="nearest")
      for _, displacement in flow_steps
  ]
  per_panels = [
      _to_pil(displacement_to_rgb(displacement, per_scale), panel, mode="nearest")
      for _, displacement in per_steps
  ]
  return feature_panels, flow_panels, per_panels


def _hstack_flush(images):
  height = images[0].height
  width = sum(image.width for image in images)
  canvas = Image.new("RGB", (width, height))
  x = 0
  for image in images:
    canvas.paste(image, (x, 0))
    x += image.width
  return canvas


def _vstack_flush(images):
  width = images[0].width
  if any(image.width != width for image in images):
    raise ValueError("Strip rows have different widths")
  height = sum(image.height for image in images)
  canvas = Image.new("RGB", (width, height))
  y = 0
  for image in images:
    canvas.paste(image, (0, y))
    y += image.height
  return canvas


def render_strip_figure(original_image, rows):
  """Stack panel rows with no gaps and place the original image once on the left.

  The photo is one panel, the same size as each strip cell, centered in the
  left column. With three rows it lines up with the middle row. With two rows
  it sits halfway between them. Strip columns stay aligned, and the rest of
  the left column is transparent.
  """
  if not rows or any(len(row) == 0 for row in rows):
    raise ValueError("rows must be non-empty lists of panels")
  panel_w, panel_h = rows[0][0].size
  if any(image.size != (panel_w, panel_h) for row in rows for image in row):
    raise ValueError("Every strip cell must be the same size")
  body_rows = [_hstack_flush(row) for row in rows]
  resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
  photo = original_image.convert("RGB").resize((panel_w, panel_h), resample)
  height = panel_h * len(body_rows)
  canvas = Image.new("RGBA", (panel_w + body_rows[0].width, height), (0, 0, 0, 0))
  canvas.paste(photo, (0, (height - panel_h) // 2))
  y = 0
  for row in body_rows:
    canvas.paste(row, (panel_w, y))
    y += panel_h
  return canvas


def render_flow_feature_gif(original_features, sampling_fields, image_index=0, panel=220, duration_ms=500):
  """GIF with cumulative optical flow beside the feature map at the same step.

  Flow brightness uses one scale for the whole sequence, and feature colors use
  the same PCA as the feature strip.
  """
  flow_steps = list(iter_composed_displacements(sampling_fields, image_index=image_index))
  feature_steps = list(iter_feature_maps(original_features, sampling_fields, image_index=image_index))
  flow_labels = [label for label, _ in flow_steps]
  feature_labels = [label for label, _ in feature_steps]
  if flow_labels != feature_labels:
    raise ValueError(f"Flow and feature steps differ: {flow_labels} vs {feature_labels}")

  flow_scale = 1e-8
  for _, displacement in flow_steps:
    flow_scale = max(flow_scale, float(displacement.float().norm(dim=0).max()))
  first = feature_steps[0][1]
  rest = [features for _, features in feature_steps[1:]]
  colored, others = reduce_dimension(first, rest, 3)
  feature_rgbs = [colored] + others

  frames = []
  for (_, displacement), rgb in zip(flow_steps, feature_rgbs):
    flow_image = _to_pil(displacement_to_rgb(displacement, flow_scale), panel, mode="nearest")
    feature_rgb = torch.nan_to_num(rgb[0], nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    feature_image = _to_pil(feature_rgb, panel, mode="nearest")
    frames.append(_flow_feature_frame(flow_image, feature_image))
  return frames, duration_ms


def _flow_feature_frame(flow_image, feature_image):
  height = max(flow_image.height, feature_image.height)
  width = flow_image.width + feature_image.width
  canvas = Image.new("RGB", (width, height), (255, 255, 255))
  canvas.paste(flow_image, (0, 0))
  canvas.paste(feature_image, (flow_image.width, 0))
  return canvas


def save_gif(frames, path, duration_ms):
  frames[0].save(
      path,
      save_all=True,
      append_images=frames[1:],
      duration=duration_ms,
      loop=0,
      disposal=2,
  )


def _captioned(image, text):
  small = _load_font(12)
  probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
  try:
    text_width = int(probe.textlength(text, font=small))
  except Exception:
    text_width = len(text) * 7
  width = max(image.width, text_width + 16)
  canvas = Image.new("RGB", (width, image.height + 36), (255, 255, 255))
  canvas.paste(image, (0, 28))
  ImageDraw.Draw(canvas).text((8, 6), text, fill=(20, 20, 20), font=small)
  return canvas


def _pad_to_width(image, width):
  """Extend a strip on the right so several strips share one column grid."""
  if image.width == width:
    return image
  canvas = Image.new("RGB", (width, image.height), (255, 255, 255))
  canvas.paste(image, (0, 0))
  return canvas


def _align_progression_strips(strips):
  width = max(image.width for image in strips)
  return [_pad_to_width(image, width) for image in strips]


def _frame(kind, stage, block, size, canvas, lattice, guidance, displacement):
  return dict(
      kind=kind,
      stage=stage,
      block=block,
      size=size,
      warped=canvas[0].detach(),
      lattice=lattice[0].detach(),
      guidance=guidance[0].detach(),
      displacement=None if displacement is None else displacement.detach(),
  )


def _magnitude_scale(frames):
  values = [frame["displacement"].float().norm(dim=0).reshape(-1) for frame in frames if frame["displacement"] is not None]
  if not values:
    return 0.0
  stacked = torch.cat(values)
  if stacked.numel() == 0:
    return 0.0
  return float(torch.quantile(stacked, 0.95))


def _load_font(size):
  try:
    return ImageFont.load_default(size=size)
  except TypeError:
    return ImageFont.load_default()


def _resize_chw(chw, size, mode):
  x = chw.detach().float().clamp(0, 1)
  if x.shape[-2] == size and x.shape[-1] == size:
    return x
  kwargs = {"align_corners": False} if mode == "bilinear" else {}
  return F.interpolate(x.unsqueeze(0), (size, size), mode=mode, **kwargs)[0]


def _to_pil(chw, size, mode="bilinear"):
  array = (_resize_chw(chw, size, mode).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
  return Image.fromarray(array)


def _color_wheel(size):
  coords = torch.linspace(-1, 1, size)
  gy, gx = torch.meshgrid(coords, coords, indexing="ij")
  radius = torch.sqrt(gx * gx + gy * gy)
  rgb = _middlebury_flow_rgb(gx, gy)
  rgb = torch.where((radius <= 1).unsqueeze(0), rgb, torch.ones_like(rgb))
  array = (rgb.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
  return Image.fromarray(array)


def _draw_arrows(image, displacement, panel, arrow_scale):
  if displacement is None:
    return image
  draw = ImageDraw.Draw(image)
  disp = displacement.detach().float().cpu()
  height, width = disp.shape[-2:]
  # One arrow per sampled pixel. 12 across is denser than the old 7x7 grid
  # and still leaves a gap between arrow heads on a ~176px panel.
  count = 12 if min(height, width) >= 12 else min(height, width)
  ys = np.linspace(0, height - 1, count).astype(int)
  xs = np.linspace(0, width - 1, count).astype(int)
  for y in ys:
    for x in xs:
      # Normalized displacement of 1 is half the image, so panel / 2 pixels.
      dx = float(disp[0, y, x]) * panel / 2 * arrow_scale
      dy = float(disp[1, y, x]) * panel / 2 * arrow_scale
      if math.hypot(dx, dy) < 1.5:
        continue
      x0 = (x + 0.5) / width * panel
      y0 = (y + 0.5) / height * panel
      _arrow(draw, x0, y0, x0 + dx, y0 + dy)
  return image


def _arrow(draw, x0, y0, x1, y1):
  for fill, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
    draw.line((x0, y0, x1, y1), fill=fill, width=width)
    angle = math.atan2(y1 - y0, x1 - x0)
    length = math.hypot(x1 - x0, y1 - y0)
    head = min(5.0, length * 0.35)
    for turn in (math.radians(150), -math.radians(150)):
      tip = angle + turn
      draw.line((x1, y1, x1 + head * math.cos(tip), y1 + head * math.sin(tip)), fill=fill, width=width)


def _frame_label(frame, n_blocks):
  size = frame["size"]
  if frame["kind"] == "start":
    return f"{size} start"
  if frame["kind"] == "upsample":
    return f"{size} upsample"
  block = frame["block"] + 1
  total = n_blocks[frame["stage"]]
  shift = frame["displacement"].float().norm(dim=0).max() * size / 2
  return f"{size} step {block}/{total}  {shift:.2f}px"


def render_deformation_montage(frames, panel=176):
  """Stack the replay into one RGB image.

  Each column is one moment in inference. Rows are the warped photo, the
  warped lattice, and that step's displacement drawn on the guidance image.
  """
  if not frames:
    raise ValueError("No frames to render")
  n_blocks = {}
  for frame in frames:
    if frame["kind"] == "deform":
      n_blocks[frame["stage"]] = max(n_blocks.get(frame["stage"], 0), frame["block"] + 1)

  magnitude_scale = _magnitude_scale(frames)
  # A normalized displacement of 1 is half the panel. Stretch tiny fields so
  # the 95th percentile arrow is about panel/12 pixels long.
  natural = magnitude_scale * panel / 2
  target_arrow = panel / 12
  arrow_scale = 1.0 if natural >= 6 else (target_arrow / natural if natural > 1e-8 else 1.0)

  label_h = 22
  gap = 6
  margin = 10
  title_h = 86
  font = _load_font(14)
  small = _load_font(12)
  wheel = _color_wheel(title_h - 16)
  percent = 100.0 * magnitude_scale / 2
  scale_note = "true length" if arrow_scale == 1 else f"arrows x{arrow_scale:.1f} so small shifts stay visible"
  notes = [
      "Top: input image warped by the same fields that resample the features.",
      "Middle: lattice warped the same way. Red lines started horizontal, cyan vertical.",
      "Bottom: that step's displacement on the guidance image. Hue is direction, white is no shift.",
      f"Full saturation ~= {percent:.2f}% of the image width. {scale_note}.",
  ]
  text_x = margin + wheel.width + 12
  probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
  try:
    note_width = max(int(probe.textlength(line, font=small)) for line in notes)
  except Exception:
    note_width = max(len(line) for line in notes) * 7
  width = margin * 2 + len(frames) * panel + (len(frames) - 1) * gap
  width = max(width, text_x + note_width + margin)
  height = margin + title_h + 3 * (panel + label_h) + 2 * gap + margin
  montage = Image.new("RGB", (width, height), (255, 255, 255))
  draw = ImageDraw.Draw(montage)
  montage.paste(wheel, (margin, margin))
  for line_index, line in enumerate(notes):
    draw.text((text_x, margin + 4 + line_index * 18), line, fill=(20, 20, 20), font=font if line_index == 0 else small)

  columns = []
  for frame in frames:
    photo = _to_pil(frame["warped"], panel, "bilinear")
    lattice = _to_pil(frame["lattice"], panel, "nearest")
    if frame["displacement"] is None:
      overlay = _to_pil(frame["guidance"], panel, "bilinear")
    else:
      color = displacement_to_rgb(frame["displacement"], magnitude_scale)
      strength = (frame["displacement"].float().norm(dim=0) / max(magnitude_scale, 1e-8)).clamp(0, 1)
      alpha = strength.unsqueeze(0)
      blended = frame["guidance"].float() * (1 - alpha) + color * alpha
      overlay = _to_pil(blended, panel, "bilinear")
      _draw_arrows(overlay, frame["displacement"], panel, arrow_scale)
    labeled = []
    for image in (photo, lattice, overlay):
      cell = Image.new("RGB", (panel, panel + label_h), (255, 255, 255))
      cell.paste(image, (0, label_h))
      ImageDraw.Draw(cell).text((4, 3), _frame_label(frame, n_blocks), fill=(0, 0, 0), font=small)
      labeled.append(cell)
    columns.append(labeled)

  y0 = margin + title_h
  for row in range(3):
    x = margin
    y = y0 + row * (panel + label_h + gap)
    for column in columns:
      montage.paste(column[row], (x, y))
      x += panel + gap
  return montage


@torch.no_grad()
def visualize_deformations(model, images, output_path=None, image_index=0, panel=176):
  """Run inference and draw every deformation on `images`.

  `images` is the same `(B, 3, H, W)` float tensor in `[0, 1]` that
  `ElasticDino.forward` expects. Returns the montage and the forward dict.
  """
  was_training = model.stages.training
  model.eval()
  try:
    outputs = model(images, return_displacements=True, return_original_features=True)
  finally:
    model.stages.train(was_training)
  frames = deformation_frames(
      images,
      outputs["sampling_fields"],
      model.config["target_size"],
      image_index=image_index,
  )
  montage = render_deformation_montage(frames, panel=panel)
  components, composed_displacement, composed_scale_px = render_composed_components(
      outputs["sampling_fields"],
      image_index=image_index,
  )
  flow, flow_scale_px = render_composed_flow(composed_displacement)
  component_steps, flow_steps, composition_steps = render_composition_progression(
      outputs["sampling_fields"],
      image_index=image_index,
  )
  flow_per_step, _ = render_step_flow(
      outputs["sampling_fields"],
      image_index=image_index,
  )
  feature_strip, feature_labels = render_feature_strip(
      outputs["original_features"],
      outputs["sampling_fields"],
      image_index=image_index,
  )
  component_steps, flow_steps, flow_per_step, feature_strip = _align_progression_strips([
      component_steps, flow_steps, flow_per_step, feature_strip,
  ])
  gif_frames, gif_duration_ms = render_flow_feature_gif(
      outputs["original_features"],
      outputs["sampling_fields"],
      image_index=image_index,
  )
  outputs["composed_displacement"] = composed_displacement
  outputs["composed_scale_px"] = composed_scale_px
  outputs["composed_rgb"] = components
  outputs["composed_flow"] = flow
  outputs["composed_flow_scale_px"] = flow_scale_px
  outputs["composed_components_steps"] = component_steps
  outputs["composed_flow_steps"] = flow_steps
  outputs["flow_per_step"] = flow_per_step
  outputs["feature_strip"] = feature_strip
  outputs["feature_labels"] = feature_labels
  outputs["flow_feature_frames"] = gif_frames
  outputs["flow_feature_duration_ms"] = gif_duration_ms
  outputs["composition_steps"] = [label for label, _ in composition_steps]
  if output_path is not None:
    montage.save(output_path)
    stem = Path(output_path)
    components_path = stem.with_name(stem.stem + "_components" + stem.suffix)
    flow_path = stem.with_name(stem.stem + "_flow" + stem.suffix)
    component_steps_path = stem.with_name(stem.stem + "_components_steps" + stem.suffix)
    flow_steps_path = stem.with_name(stem.stem + "_flow_steps" + stem.suffix)
    flow_per_step_path = stem.with_name(stem.stem + "_flow_per_step" + stem.suffix)
    feature_strip_path = stem.with_name(stem.stem + "_features" + stem.suffix)
    gif_path = stem.with_name(stem.stem + "_flow_features.gif")
    components.save(components_path)
    flow.save(flow_path)
    component_steps.save(component_steps_path)
    flow_steps.save(flow_steps_path)
    flow_per_step.save(flow_per_step_path)
    feature_strip.save(feature_strip_path)
    save_gif(gif_frames, gif_path, gif_duration_ms)
    outputs["composed_rgb_path"] = str(components_path)
    outputs["composed_flow_path"] = str(flow_path)
    outputs["composed_components_steps_path"] = str(component_steps_path)
    outputs["composed_flow_steps_path"] = str(flow_steps_path)
    outputs["flow_per_step_path"] = str(flow_per_step_path)
    outputs["feature_strip_path"] = str(feature_strip_path)
    outputs["flow_feature_gif_path"] = str(gif_path)
  return montage, outputs


def _load_image(path, device):
  image = Image.open(path).convert("RGB")
  tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
  return tensor.unsqueeze(0).to(device)


def main():
  parser = argparse.ArgumentParser(description="Visualize ElasticDino deformations on an input image.")
  parser.add_argument("--image", required=True, help="RGB image to run through ElasticDino")
  parser.add_argument("--model", default="elasticdino-32-S-no-reg", help="Key in ElasticDino CONFIGS")
  parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to the Hugging Face download.")
  parser.add_argument("--dino-repo", default="facebookresearch/dinov2")
  parser.add_argument("--output", default="deformation_visualization.png")
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--panel", type=int, default=176)
  args = parser.parse_args()

  device = torch.device(args.device)
  model = ElasticDino.from_pretrained(args.model, checkpoint_path=args.checkpoint, dino_repo=args.dino_repo)
  model.to(device)
  images = _load_image(args.image, device)
  montage, outputs = visualize_deformations(model, images, output_path=args.output, panel=args.panel)
  print(f"Wrote {args.output} ({montage.width}x{montage.height})")
  print(f"Wrote {outputs['composed_rgb_path']}")
  print(f"Wrote {outputs['composed_flow_path']}")
  print(f"Wrote {outputs['composed_components_steps_path']}")
  print(f"Wrote {outputs['composed_flow_steps_path']}")
  print(f"Wrote {outputs['flow_per_step_path']}")
  print(f"Wrote {outputs['feature_strip_path']}")
  print(f"Wrote {outputs['flow_feature_gif_path']}")


if __name__ == "__main__":
  main()
