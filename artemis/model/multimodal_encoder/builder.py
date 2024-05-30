import os
from .clip_encoder import CLIPVisionTower
from ..layers import MLVLROIQueryModule
from torch_kmeans import KMeans
# ============================================================================================================

def build_image_tower(image_tower_cfg, **kwargs):
    image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
    is_absolute_path_exists = os.path.exists(image_tower)
    if is_absolute_path_exists or image_tower.startswith("openai") or image_tower.startswith("laion"):
        return CLIPVisionTower(image_tower, args=image_tower_cfg, **kwargs)

    raise ValueError(f'Unknown image tower: {image_tower}')

def build_spi_model(spi_model_config):
    mm_hidden_size = getattr(spi_model_config, 'mm_hidden_size', 1024)
    hidden_size = getattr(spi_model_config, 'hidden_size', 4096)
    num_levels = getattr(spi_model_config, 'num_levels', 4)
    return MLVLROIQueryModule(mm_hidden_size, hidden_size, num_levels)

def build_kmeans(kmeans_config):
    n_clusters = getattr(kmeans_config, 'kmeans_n_clusters', 4)
    max_iter = getattr(kmeans_config, 'kmeans_max_iter', 3)
    return KMeans(n_clusters=n_clusters, max_iter=max_iter, verbose=False)

# ============================================================================================================
