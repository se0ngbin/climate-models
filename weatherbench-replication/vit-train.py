from pathlib import Path
import sys
sys.path.insert(0, '/u/home/s/se0ngbin/climate_tutorial')

path = Path("/u/scratch/s/se0ngbin/weatherbench")
path = path/"data/weatherbench/era5/5.625"

from climate_tutorial.utils.data import load_dataset, view
from climate_tutorial.utils.datetime import Year, Days, Hours
from climate_tutorial.data import DataModule

data_module = DataModule(
    dataset = "ERA5",
    task = "forecasting",
    root_dir = path,
    in_vars = ["geopotential_500", "temperature_850"],
    out_vars = ["geopotential_500", "temperature_850"],
    train_start_year = Year(1979),
    val_start_year = Year(2015),
    test_start_year = Year(2017),
    end_year = Year(2018),
    pred_range = Days(3),
    subsample = Hours(6),
    batch_size = 128,
    num_workers = 24
)

from climate_tutorial.models import load_model

model_kwargs = {
    "img_size": [32, 64],
    "patch_size": 2,
    "drop_path": 0.1,
    "drop_rate": 0.1,
    "learn_pos_emb": True,
    "in_vars": data_module.hparams.in_vars,
    "out_vars": data_module.hparams.out_vars,
    "embed_dim": 128,
    "depth": 8,
    "decoder_depth": 0,
    "num_heads": 4,
    "mlp_ratio": 4,
}

optim_kwargs = {
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "warmup_epochs": 1,
    "max_epochs": 5,
}

model_module = load_model(name = "vit", task = "forecasting", model_kwargs = model_kwargs, optim_kwargs = optim_kwargs)

# sets pred_range and other vars for ForecastLitModule
from climate_tutorial.models import set_climatology
set_climatology(model_module, data_module) 

from climate_tutorial.training import Trainer, WandbLogger

trainer = Trainer(
    seed = 0,
    accelerator = "gpu",
    precision = 16,
    max_epochs = 5,
    # logger = WandbLogger(project = "climate_tutorial", name = "forecast-vit")
)

trainer.fit(model_module, data_module)
