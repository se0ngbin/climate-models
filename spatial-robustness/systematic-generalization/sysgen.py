import torch
import sys
sys.path.insert(0, '/home/snandy/climate-learn-sys-gen/src/')

from climate_learn.utils.datetime import Year, Days, Hours
from climate_learn.data import DataModule


dataset_path = "/data0/datasets/weatherbench/data/weatherbench/era5/5.625deg/"


data_module = DataModule(
    dataset = "ERA5",
    task = "forecasting",
    root_dir = dataset_path,
    in_vars = ["temperature", "geopotential", "2m_temperature"],
    out_vars = ["temperature_850", "geopotential_500", "2m_temperature"],
    train_start_year = Year(1979),
    val_start_year = Year(2016),
    test_start_year = Year(2017),
    end_year = Year(2018),
    pred_range = Days(3),
    subsample = Hours(6),
    batch_size = 32,
    num_workers = 8,
)

from climate_learn.models import load_model

# model_kwargs = {
#     "img_size": [32, 64],
#     "patch_size": 2,
#     "drop_path": 0.1,
#     "drop_rate": 0.1,
#     "learn_pos_emb": True,
#     "in_vars": data_module.hparams.in_vars,
#     "out_vars": data_module.hparams.out_vars,
#     "embed_dim": 128,
#     "depth": 8,
#     "decoder_depth": 0,
#     "num_heads": 4,
#     "mlp_ratio": 4,
# }
model_kwargs = {
    "in_channels": len(data_module.hparams.in_vars),
    "out_channels": len(data_module.hparams.out_vars),
    "n_blocks": 4
}

optim_kwargs = {
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "warmup_epochs": 1,
    "max_epochs": 64,
}

# model_module = load_model(name = "vit", task = "forecasting", model_kwargs = model_kwargs, optim_kwargs = optim_kwargs)
resnet_model_module = load_model(name = "resnet", task = "forecasting", model_kwargs = model_kwargs, optim_kwargs = optim_kwargs)
#unet_model_module = load_model(name = "unet", task = "forecasting", model_kwargs = model_kwargs, optim_kwargs = optim_kwargs)

# add_description
from climate_learn.models import set_climatology
set_climatology(resnet_model_module, data_module)
#set_climatology(unet_model_module, data_module)

from climate_learn.training import Trainer, WandbLogger

resnet_trainer = Trainer(
    seed = 0,
    accelerator = "gpu",
    precision = 16,
    max_epochs = 64,
    # logger = WandbLogger(project = "climate_learn", name = "forecast-vit")
)

# unet_trainer = Trainer(
#     seed = 0,
#     accelerator = "gpu",
#     precision = 16,
#     max_epochs = 5,
#     # logger = WandbLogger(project = "climate_learn", name = "forecast-vit")
# )

resnet_trainer.fit(resnet_model_module, data_module)

resnet_trainer.test(resnet_model_module, data_module)