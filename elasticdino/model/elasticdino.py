import torch
import torch.nn as nn
import math
from elasticdino.model.dino import DinoV2, resize_for_dino
from elasticdino.model.dino import DinoV3, resize_for_dino_V3
from elasticdino.model.dino import ClipWrapper, resize_for_clip
from elasticdino.model.layers import ProjectionLayer, ResidualBlock, Activation, FCLayer
import logging
from huggingface_hub import hf_hub_download


# logger = logging.getLogger("ElasticDino")

# Doc ? seems to not be used !!
class TaskHeads(nn.Module):
  def __init__(self, n_features_in, n_segmentation_classes):
    super().__init__()
    
    def make_head(dim):
      return nn.Sequential(
        nn.Conv2d(n_features_in, n_features_in // 2, 1),
        nn.ReLU(),
        nn.Conv2d(n_features_in // 2, n_features_in // 4, 1),
        nn.ReLU(),
        nn.Conv2d(n_features_in // 4, n_features_in // 8, 1),
        nn.ReLU(),
        nn.Conv2d(n_features_in // 8, dim, 1),
    )

    self.reproduction_head = make_head(3)

    self.depth_head = make_head(1)

    self.segmentation_head = nn.Sequential(
      nn.Conv2d(n_features_in, n_features_in, 1),
        nn.ReLU(),
        nn.Conv2d(n_features_in, n_features_in, 1),
        nn.ReLU(),
        nn.Conv2d(n_features_in, n_segmentation_classes, 1),
    )

    self.diffuse_illumination_head = make_head(3)
    self.diffuse_reflectance_head = make_head(3)
    self.residual_head = make_head(3)
    self.normal_bump_cam_head = make_head(3)

  def forward(self, x):
    return dict(
                depth=self.depth_head(x),
                segmentation=self.segmentation_head(x),
                reproduced=self.reproduction_head(x),
                diffuse_illuminations=self.diffuse_illumination_head(x),
                diffuse_reflectances=self.diffuse_reflectance_head(x),
                residuals=self.residual_head(x),
                normal_bump_cams=self.normal_bump_cam_head(x),
              )
  
def make_base_locations(batch_size, size, dtype, device):
    # Doc torch.arange is a PyTorch function that returns a 1D tensor with evenly spaced values within a specified range
    # These lines create range in [-1,1) (normalized coordinates)
    x = torch.arange(size, device=device, dtype=dtype) * (2 / size) - 1
    y = torch.arange(size, device=device, dtype=dtype) * (2 / size) - 1
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    res = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return res


class DeformerBlock(nn.Module):
  # Doc n_features_in is number of features in the input feature map and n_image_features is number of features in the input image
  def __init__(self, n_layers, n_features, n_features_in, n_image_features):
    super().__init__()
    self.image_encoder = ProjectionLayer(n_image_features, n_features)
    self.feature_encoder = ProjectionLayer(n_features_in, n_features)

    # Doc n_features * 2 because use it on concatenation of image and feature encoders
    self.convs = nn.Sequential(
        ProjectionLayer(n_features * 2, n_features),
        *[ResidualBlock(n_features) for _ in range(n_layers)]
    )

    last_layer = nn.Conv2d(n_features // 8, 2, 1)
    # initialize last layer to small values and no bias to have an initial field that is close to the identity
    torch.nn.init.normal_(last_layer.weight, mean=0.0, std=0.003, generator=None)
    nn.init.zeros_(last_layer.bias)

    self.deformer = nn.Sequential(
        nn.Conv2d(n_features, n_features // 2, 1),
        Activation(),
        nn.Conv2d(n_features // 2, n_features // 4, 1),
        Activation(),
        nn.Conv2d(n_features // 4, n_features // 8, 1),
        Activation(),
        last_layer,
    )


  def forward(self, features, image, return_displacements=False, additional_inputs=[]):
    image = self.image_encoder(image)
    f = self.feature_encoder(features)
    f = self.convs(torch.cat([f, image], dim=1))
    # Doc Permute operations: Convert between (B,H,W,2) ↔ (B,2,H,W) tensor layouts
    base_locations = make_base_locations(image.shape[0], image.shape[-1], image.dtype, image.device).permute((0, 3, 1, 2))
    # Doc Neural network predicts 2D displacement vectors (dx, dy) for each pixel
    displacements = self.deformer(f)
    field = base_locations + displacements
    field = field.permute((0, 2, 3, 1))
    # Doc grid_sample: Samples input tensors at deformed coordinates
    # Doc padding_mode="border": Uses edge values for out-of-bounds sampling
    results = dict(features=torch.nn.functional.grid_sample(features, field, padding_mode="border", align_corners=False))
    add = []
    for x in additional_inputs:
      add.append(torch.nn.functional.grid_sample(x, field, padding_mode="border", align_corners=False))
    results["additional_inputs"] = add
    if return_displacements:
      results["displacements"] = displacements
    return results


class ElasticDinoStage(nn.Module):
  def __init__(self, layer_config, n_features_in, n_image_features):
    super().__init__()
    self.blocks = nn.ModuleList([
        DeformerBlock(layer_config["layers_per_block"], layer_config["hidden_features"], n_features_in, n_image_features)
        for _ in range(layer_config["n_blocks"])
    ])

  def forward(self, features, images, return_displacements=False, additional_inputs=[]):
    displacements = []
    images = torch.nn.functional.interpolate(images, features.shape[-1], mode="bilinear")
    for block in self.blocks:
        block_results = block(features, images, return_displacements, additional_inputs)
        features = block_results["features"]
        additional_inputs = block_results["additional_inputs"]
        if return_displacements:
          displacements.append(block_results["displacements"])
    results = dict(features=features)
    if return_displacements:
      results["displacements"] = displacements
    if additional_inputs is not None:
      results["additional_inputs"] = additional_inputs
    return results


CONFIGS = {
  # "elasticdino-64-L":  dict(
  #       dino_model="l",
  #       n_features_in=1024,
  #       layers={
  #           64: dict(hidden_features=256, n_blocks=4, layers_per_block=8),
  #           128: dict(hidden_features=256, n_blocks=3, layers_per_block=8),
  #       },
  #       start_size=64,
  #       target_size=128,
  #   ),
  # # # !!!!!!!!!!!! newly commented out
    # "elasticdino-32-L": dict(
    #     dino_model="l",
    #     n_features_in=1024,
    #     layers={
    #         32: dict(hidden_features=512, n_blocks=5, layers_per_block=8),
    #         64: dict(hidden_features=256, n_blocks=4, layers_per_block=8),
    #         128: dict(hidden_features=256, n_blocks=3, layers_per_block=8),
    #     },
    #     start_size=32,
    #     target_size=128,
    # ),
    # "elasticdino-64-L":  dict(
    #     dino_model="l",
    #     n_features_in=1024,
    #     layers={
    #         64: dict(hidden_features=512, n_blocks=3, layers_per_block=6),
    #         128: dict(hidden_features=256, n_blocks=2, layers_per_block=6),
    #         256: dict(hidden_features=128, n_blocks=1, layers_per_block=4),
    #     },
    #     start_size=64,
    #     target_size=256,
    # ),
    # "elasticdino-64-S":  dict(
    #     dino_model="s",
    #     n_features_in=384,
    #     layers={
    #         64: dict(hidden_features=512, n_blocks=3, layers_per_block=6),
    #         128: dict(hidden_features=256, n_blocks=2, layers_per_block=6),
    #         256: dict(hidden_features=128, n_blocks=1, layers_per_block=4),
    #     },
    #     start_size=64,
    #     target_size=256,
    # ),
    # "elasticdino-64-B":  dict(
    #     dino_model="b",
    #     n_features_in=768,
    #     layers={
    #         64: dict(hidden_features=512, n_blocks=3, layers_per_block=6),
    #         128: dict(hidden_features=256, n_blocks=2, layers_per_block=6),
    #         256: dict(hidden_features=128, n_blocks=1, layers_per_block=4),
    #     },
    #     start_size=64,
    #     target_size=256,
    # ),
    # "elasticdino-64-G":  dict(
    #     dino_model="g",
    #     n_features_in=1536,
    #     layers={
    #         64: dict(hidden_features=512, n_blocks=3, layers_per_block=6),
    #         128: dict(hidden_features=256, n_blocks=2, layers_per_block=6),
    #         256: dict(hidden_features=128, n_blocks=1, layers_per_block=4),
    #     },
    #     start_size=64,
    #     target_size=256,
    # ),
    # !!!!!!!!!!!!!! used this until noticed with reg
    "elasticdino-new-32-S": dict(
        dino_model="s",
        n_features_in=384,
        layers={
            32: dict(hidden_features=512, n_blocks=3, layers_per_block=6),
            64: dict(hidden_features=512, n_blocks=3, layers_per_block=6),
            128: dict(hidden_features=256, n_blocks=2, layers_per_block=6),
            256: dict(hidden_features=128, n_blocks=1, layers_per_block=4),
        },
        start_size=32,
        target_size=256,
        with_reg=True,
    ),
    "elasticdino-32-S-no-reg": dict(
        dino_model="s",
        n_features_in=384,
        layers={
            32: dict(hidden_features=512, n_blocks=3, layers_per_block=6),
            64: dict(hidden_features=512, n_blocks=3, layers_per_block=6),
            128: dict(hidden_features=256, n_blocks=2, layers_per_block=6),
            256: dict(hidden_features=128, n_blocks=1, layers_per_block=4),
        },
        start_size=32,
        target_size=256,
        with_reg=False,
    ),
    "elasticdino-3-32-S": dict(
        dino_model="s",
        n_features_in=384,
        layers={
            32: dict(hidden_features=512, n_blocks=3, layers_per_block=6),
            64: dict(hidden_features=512, n_blocks=3, layers_per_block=6),
            128: dict(hidden_features=256, n_blocks=2, layers_per_block=6),
            256: dict(hidden_features=128, n_blocks=1, layers_per_block=4),
        },
        start_size=32,
        target_size=256,
    ),
    "elasticdino-clip-32-b": dict(
        clip_model="b",
        # n_features_in=384, # !!!! maybe needs to be 768
        n_features_in=768, # !!!! maybe needs to be 768
        layers={
            32: dict(hidden_features=512, n_blocks=3, layers_per_block=6),
            64: dict(hidden_features=512, n_blocks=3, layers_per_block=6),
            128: dict(hidden_features=256, n_blocks=2, layers_per_block=6),
            256: dict(hidden_features=128, n_blocks=1, layers_per_block=4),
        },
        start_size=32,
        target_size=256,
    ),
}

def repair_checkpoint(path):
    ckpt = torch.load(path, weights_only=True, map_location=torch.device("cpu"))
    in_state_dict = ckpt
    pairings = [
        (src_key, "".join(src_key.split("_orig_mod.")))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        return  # Do not write checkpoint if no need to repair!
    out_state_dict = {}
    for src_key, dest_key in pairings:
        out_state_dict[dest_key] = in_state_dict[src_key]
    ckpt = out_state_dict
    torch.save(ckpt, path)


class ElasticDino(nn.Module):
  def __init__(self, config, dino_repo, DinoV3_backbone=False, Clip_backbone=False):
    super().__init__()
    self.config = config

    n_features_in = config["n_features_in"]
    layer_configs = config["layers"]

    # Doc probably 3 for RGB
    n_image_features = 3
    n_upscales = int(math.log2(config["target_size"] // config["start_size"])) + 1
    assert n_upscales == len(layer_configs), "Incompatible resolutions and feature config"

    self.stages = nn.ModuleList([
        ElasticDinoStage(layer_configs[res], n_features_in, n_image_features) for res in layer_configs
    ])

    self.use_clip = Clip_backbone
    print(f"in elasticdino init {self.use_clip=}")
    self.use_dinov3 = DinoV3_backbone
    if self.use_clip:
      self.dino = ClipWrapper(config["clip_model"]) # !!!! change it to be backbone instead of dino / clip
      print("Called clip wrapper in elasticdino")
    else:
      if DinoV3_backbone:
        self.dino = DinoV3(dino_repo, config["dino_model"])
      else:
        self.dino = DinoV2(dino_repo, config["dino_model"], config["with_reg"])

  def forward(self, images, return_all_scales=False, return_original_features=False, return_displacements=False, n_hidden_layers=None, external_features=None):
    if external_features != None:
      assert return_original_features == False and n_hidden_layers == None

    if self.use_clip:
      resize_for_back = resize_for_clip
    else:
      if self.use_dinov3:
        resize_for_back = resize_for_dino_V3
      else:
        resize_for_back = resize_for_dino

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
      if external_features == None:
        if n_hidden_layers is None:
          additional_inputs = []
          features_in = self.dino.get_features_for_tensor(resize_for_back(images, self.config["start_size"]))
        else:
          additional_inputs = self.dino.get_intermediate_features_for_tensor(resize_for_back(images, self.config["start_size"]), n_hidden_layers)
          features_in = additional_inputs.pop(-1)
      else: 
        additional_inputs = []
        features_in = external_features
      features = features_in
      images = nn.functional.interpolate(images, self.config["target_size"], mode="bilinear", antialias=True)
      results = []
      all_displacements = []
      n = len(self.stages)
      current_size = features.shape[-1]
      for i in range(n):
        stage_outputs = self.stages[i](features, images, return_displacements, additional_inputs)
        features = stage_outputs["features"]

        if return_all_scales:
          results.append(features)
        additional_inputs = stage_outputs["additional_inputs"]

        del stage_outputs

        if return_displacements:
          all_displacements.append(features["displacements"])
        if i < n - 1:
          current_size *= 2
          features = torch.nn.functional.interpolate(features, current_size, mode="nearest")
      
      if (not return_all_scales) and (not return_original_features) and (not return_displacements) and n_hidden_layers is None:
        return features

      out = dict(deformed_features=features)
      if return_all_scales:
        out["all_scales"] = results
      if return_original_features:
        out["original_features"] = features_in
      if return_displacements:
        out["displacements"] = all_displacements
      if n_hidden_layers is not None:
        additional_inputs.append(features)
        out["hidden_layers"] = additional_inputs
      
      del additional_inputs
      del all_displacements
      del features_in
      del results
      del features
      del images
      return out
    

  def parameters(self):
    return self.stages.parameters()
  
  def train(self, value=True):
    self.stages.train(value)
  
  # !!!! for clean code would need to extract to util function and have 2 from_pretrained (dinov2 and dinov3)
  def from_pretrained(model_name, checkpoint_path=None, dino_repo='facebookresearch/dinov2'):
    config = CONFIGS[model_name]
    if config.get("clip_model") and config["clip_model"] is not None:
      is_clip_backbone = True
    else:
      is_clip_backbone = False

    if checkpoint_path is None:
      # checkpoint_path = hf_hub_download(repo_id=f"ulyssemizrahi/{model_name}", filename=f"{model_name}.pth")
      checkpoint_path = hf_hub_download(repo_id=f"articuno7/{model_name}", filename=f"{model_name}.pth")
    repair_checkpoint(checkpoint_path) # Doc ???? Why would need to repair?
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    # !!!!!!!!!!!!!
    if is_clip_backbone:
      print(f"!!!!!!!!!!!! using Clip, {model_name}")
      model = ElasticDino(config, None, Clip_backbone=True)
    else:
      if dino_repo == 'dinov3':
        print(f"!!!!!!!!!!!! using DinoV3, {model_name} {dino_repo=}")
        model = ElasticDino(config, dino_repo, DinoV3_backbone=True)
      else:
        print(f"!!!!!!!!!!!! using DinoV2, {model_name} {dino_repo=}")
        model = ElasticDino(config, dino_repo)
    # don't load parameters in the pretrained dino
    tmp_dino = model.dino
    model.dino = None
    model.load_state_dict(checkpoint)
    model.dino = tmp_dino
    return model
