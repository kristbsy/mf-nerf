from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.configs.base_config import ViewerConfig

from .mfnerf import MfNerfModelConfig

mfnerf = MethodSpecification(
      config = TrainerConfig(
            method_name="mfnerf",
            steps_per_eval_batch=500,
            steps_per_save=2000,
            max_num_iterations=30000,
            mixed_precision=True,
            pipeline=VanillaPipelineConfig(
                datamanager=ParallelDataManagerConfig(
                    dataparser=NerfstudioDataParserConfig(),
                    train_num_rays_per_batch=4096,
                    eval_num_rays_per_batch=4096,
                ),
                model=MfNerfModelConfig(
                    eval_num_rays_per_chunk=1 << 15,
                    average_init_density=0.01,
                    camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                ),
            ),
            optimizers={
                "proposal_networks": {
                    "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                    "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
                },
                "fields": {
                    "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                    "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
                },
                "camera_opt": {
                    "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                    "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
                },
            },
            viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
            vis="viewer",
    ),
    description="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)
