from pathlib import Path
import torch
import xarray as xr
import sys

cmip_path = Path("/data0/datasets/weatherbench/data/esgf/cmip6/5.625deg")   # replace with path
era_path = Path("/data0/datasets/weatherbench/data/weatherbench/era5/5.625deg")
sys.path.insert(0, '/home/seongbin/climate-learn/src')

from climate_learn.utils.datetime import Year, Days, Hours
from climate_learn.data import DataModule
from climate_learn.models import load_model
from climate_learn.models import set_climatology
from climate_learn.models import fit_lin_reg_baseline
from climate_learn.training import Trainer, WandbLogger

era5_data_module = DataModule(
    dataset = "ERA5",
    task = "forecasting",
    root_dir = era_path,
    in_vars = ["temperature", "geopotential", "2m_temperature"],
    out_vars = ["temperature_850", "geopotential_500", "2m_temperature"],
    train_start_year = Year(1979),
    val_start_year = Year(2012),
    test_start_year = Year(2013),
    end_year = Year(2014),
    pred_range = Days(3),
    subsample = Hours(6),
    batch_size = 32,
    num_workers = 64
) 

cmip6_data_module = DataModule(
    dataset = "CMIP6",
    task = "forecasting",
    root_dir = cmip_path,
    in_vars = ["temperature", "geopotential", "air_temperature"],
    out_vars = ["temperature_850", "geopotential_500", "air_temperature"],
    train_start_year = Year(1979),
    val_start_year = Year(2012),
    test_start_year = Year(2013),
    end_year = Year(2014),
    pred_range = Days(3),
    batch_size = 32,
    num_workers = 64
) 

era5_nb = len(era5_data_module.train_dataloader())
era5_wepochs = 1000 // era5_nb + 1
cmip6_nb = len(cmip6_data_module.train_dataloader())
cmip6_wepochs = 1000 // cmip6_nb + 1




cmip_model_kwargs = {
    "in_channels": len(cmip6_data_module.hparams.in_vars),
    "out_channels": len(cmip6_data_module.hparams.out_vars),
    "n_blocks": 19
}

optim_kwargs = {
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "warmup_epochs": cmip6_wepochs,
    "max_epochs": 100,
}

cmip_model_module = load_model(name = "resnet", task = "forecasting", model_kwargs = cmip_model_kwargs, optim_kwargs = optim_kwargs)

set_climatology(cmip_model_module, cmip6_data_module)

cmip_trainer = Trainer(
    seed = 0,
    accelerator = "gpu",
    precision = 16,
    max_epochs = 100,
    logger = WandbLogger(project = "climate_tutorial", name = "cmip-era-3day")
)

cmip_trainer.fit(cmip_model_module, cmip6_data_module)

fit_lin_reg_baseline(cmip_model_module, cmip6_data_module, reg_hparam=0.0)

era5_model_kwargs = {
    "in_channels": len(era5_data_module.hparams.in_vars),
    "out_channels": len(era5_data_module.hparams.out_vars),
    "n_blocks": 19
}

optim_kwargs = {
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "warmup_epochs": era5_wepochs,
    "max_epochs": 100,
}

era5_model_module = load_model(name = "resnet", task = "forecasting", model_kwargs = era5_model_kwargs, optim_kwargs = optim_kwargs)
set_climatology(era5_model_module, era5_data_module)

era5_trainer = Trainer(
    seed = 0,
    accelerator = "gpu",
    precision = 16,
    max_epochs = 100,
    logger = WandbLogger(project = "climate_tutorial", name = "era-cmip-3day")
)

era5_trainer.fit(era5_model_module, era5_data_module)

fit_lin_reg_baseline(era5_model_module, era5_data_module, reg_hparam=0.0)




# cmip -> era
cmip_trainer.test(cmip_model_module, era5_data_module)

# era -> era
era5_trainer.test(era5_model_module, era5_data_module)

# cmip -> cmip
cmip_trainer.test(cmip_model_module, cmip6_data_module)

# era -> cmip
era5_trainer.test(era5_model_module, cmip6_data_module)