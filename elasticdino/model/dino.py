
import torchvision.transforms as transforms
import torch
import math
import os
import logging

import torch.nn as nn

# logger = logging.getLogger("ElasticDino")

MINIMAL_IMAGE_SIZE = 224
MINIMAL_GRID_SIZE = MINIMAL_IMAGE_SIZE // 14


FEATURE_SIZES = {
    's': 384,
    # 'b': 768,
    # 'l': 1024,
    # 'g': 1536
}

def resize_for_dino(images, starting_size):
    factor = starting_size // MINIMAL_GRID_SIZE
    images = torch.nn.functional.interpolate(images, factor * MINIMAL_IMAGE_SIZE, mode="bilinear", align_corners=False, antialias=True)
    return images  


DINO_CACHE = {}

class DinoV2(nn.Module):
  def __init__(self, dino_repo, dino_model, with_reg=True):
    super().__init__()
    # logger.info("Loading DinoV2 backbone")
    if dino_model in DINO_CACHE:
      dino_backbone = DINO_CACHE[dino_model]
    else:
      source = "github" if dino_repo == 'facebookresearch/dinov2' else "local"
      # !!!!! swtich to non reg version
      if with_reg:
        dino_backbone = torch.hub.load(dino_repo, f'dinov2_vit{dino_model}14_reg', source=source)
        print("!!!!! Loaded DinoV2 backbone with reg")
      else:
        dino_backbone = torch.hub.load(dino_repo, f'dinov2_vit{dino_model}14', source=source)
        print("!!!!! Loaded DinoV2 backbone with no reg")
      DINO_CACHE[dino_model] = dino_backbone

    # logger.info("DinoV2 backbone loaded")
    dino_backbone.requires_grad_(False)
    self.dino_backbone = dino_backbone.eval()
    self.feature_size = FEATURE_SIZES[dino_model]

  def prepare_images(self, images, device="cuda"):
    tensors = [transforms.functional.pil_to_tensor(i) for i in images]
    tensors = torch.stack(tensors)
    tensors = tensors.to(dtype=torch.float32, device=device) / 255.0
    return tensors

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # Doc: functions after addition of normalization
  def get_features_for_tensor(self, images):
    return self.get_intermediate_features_for_tensor(images, 1)[0]

  def get_intermediate_features_for_tensor(self, images, n):
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    with torch.no_grad():
      images = transform(images)
      res = self.dino_backbone.get_intermediate_layers(images, n=n)
      grid_size = int(math.sqrt(res[0].shape[1]))
      res = [r.reshape((r.shape[0], grid_size, grid_size, self.feature_size)).permute((0, 3, 1, 2)) for r in res]
      del images
      return res
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!

  # Doc: Original functions (without normalization)
  # def get_features_for_tensor(self, images):
  #   return self.get_intermediate_features_for_tensor(images, 1)[0]

  # def get_intermediate_features_for_tensor(self, images, n):
  #   with torch.no_grad():
  #     res = self.dino_backbone.get_intermediate_layers(images, n=n)
  #     grid_size = int(math.sqrt(res[0].shape[1]))
  #     res = [r.reshape((r.shape[0], grid_size, grid_size, self.feature_size)).permute((0, 3, 1, 2)) for r in res]
  #     del images
  #     return res
  
  def get_features(self, images):
    images = self.prepare_images(images)
    return self.get_features_for_tensor(images)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DinoV3 !!!!!!!!!!!!!!!!!!!!!!!


MINIMAL_IMAGE_SIZE_V3 = 256
MINIMAL_GRID_SIZE_V3 = MINIMAL_IMAGE_SIZE_V3 // 16


FEATURE_SIZES_V3 = {
    's': 384,
    # 'b': 768,
    # 'l': 1024,
    # 'g': 1536
}

def resize_for_dino_V3(images, starting_size):
    factor = starting_size // MINIMAL_GRID_SIZE_V3
    images = torch.nn.functional.interpolate(images, factor * MINIMAL_IMAGE_SIZE_V3, mode="bilinear", align_corners=False, antialias=True)
    return images  

DINO_CACHE_V3 = {}

class DinoV3(nn.Module):
  def __init__(self, dino_repo, dino_model):
    super().__init__()
    # logger.info("Loading DinoV3 backbone")
    if dino_model in DINO_CACHE_V3:
      dino_backbone = DINO_CACHE_V3[dino_model]
    else:
      dino_backbone = torch.hub.load("/home/dcor/seanzaretzky/elasticdino/DinoV3/dinov3", 'dinov3_vits16plus', source='local', weights="/home/dcor/seanzaretzky/elasticdino/DinoV3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth").to("cuda")
      # source = "github" if dino_repo == 'facebookresearch/dinov3' else "local"
      # dino_backbone = torch.hub.load(dino_repo, f'dinov3_vit{dino_model}16plus', source=source)
      DINO_CACHE_V3[dino_model] = dino_backbone

    # logger.info("DinoV3 backbone loaded")
    dino_backbone.requires_grad_(False)
    self.dino_backbone = dino_backbone.eval()
    self.feature_size = FEATURE_SIZES_V3[dino_model]

  def prepare_images(self, images, device="cuda"):
    tensors = [transforms.functional.pil_to_tensor(i) for i in images]
    tensors = torch.stack(tensors)
    tensors = tensors.to(dtype=torch.float32, device=device) / 255.0
    return tensors

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # Doc: functions after addition of normalization
  def get_features_for_tensor(self, images):
    return self.get_intermediate_features_for_tensor(images, 1)[0]

  def get_intermediate_features_for_tensor(self, images, n):
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    with torch.no_grad():
      images = transform(images)
      res = self.dino_backbone.get_intermediate_layers(images, n=n)
      grid_size = int(math.sqrt(res[0].shape[1]))
      res = [r.reshape((r.shape[0], grid_size, grid_size, self.feature_size)).permute((0, 3, 1, 2)) for r in res]
      del images
      return res
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!

  # Doc: Original functions (without normalization)
  # def get_features_for_tensor(self, images):
  #   return self.get_intermediate_features_for_tensor(images, 1)[0]

  # def get_intermediate_features_for_tensor(self, images, n):
  #   with torch.no_grad():
  #     res = self.dino_backbone.get_intermediate_layers(images, n=n)
  #     grid_size = int(math.sqrt(res[0].shape[1]))
  #     res = [r.reshape((r.shape[0], grid_size, grid_size, self.feature_size)).permute((0, 3, 1, 2)) for r in res]
  #     del images
  #     return res
  
  def get_features(self, images):
    images = self.prepare_images(images)
    return self.get_features_for_tensor(images)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CLIP !!!!!!!!!!!!!!!!!!!!!!!


MINIMAL_IMAGE_SIZE_CLIP = 256
MINIMAL_GRID_SIZE_CLIP = MINIMAL_IMAGE_SIZE_CLIP // 16


# FEATURE_SIZES_V3 = {
#     's': 384,
#     # 'b': 768,
#     # 'l': 1024,
#     # 'g': 1536
# }

def resize_for_clip(images, starting_size):
    factor = starting_size // MINIMAL_GRID_SIZE_CLIP
    images = torch.nn.functional.interpolate(images, factor * MINIMAL_IMAGE_SIZE_CLIP, mode="bilinear", align_corners=False, antialias=True)
    return images  

import timm

# !!! seems the model size (s,b,l, ....) is hard coded will probably need to change that.
class ClipWrapper:
  def __init__(self, clip_model):

    model = timm.create_model(
                "vit_base_patch16_clip_384",
                pretrained=True,
                num_classes=0,
                dynamic_img_size=True,
                dynamic_img_pad=False,
                # **kwargs,
            )
    self.model = model.eval().cuda()

  def prepare_images(self, images, image_check=True):
    pass

  @torch.no_grad()
  def get_features_for_tensor(self, images, image_check=True):
    feats, _ = self.model.forward_intermediates(
                images,
                1,
                return_prefix_tokens=True,
                norm=True,
                output_fmt="NCHW",
                intermediates_only=True,
            )[0]
    return feats