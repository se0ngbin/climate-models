import torch
import sys
sys.path.insert(0, '/home/snandy/climate-learn-sys-gen/src/')

from climate_learn.utils.datetime import Year, Days, Hours
from climate_learn.data import DataModule


dataset_path = "/data0/datasets/weatherbench/data/weatherbench/era5/5.625deg/"

print(torch.cuda.is_available())


data_module = DataModule(
    dataset = "ERA5",
    task = "forecasting",
    root_dir = dataset_path,
    # in_vars = ["temperature", "geopotential", "2m_temperature"],
    # in_vars = ['2m_temperature', 'geopotential_50', 'geopotential_250', 'geopotential_500', 'geopotential_600', 'geopotential_700', 'geopotential_850', 'geopotential_925',  'u_component_of_wind_50', 'u_component_of_wind_250', 'u_component_of_wind_500', 'u_component_of_wind_600', 'u_component_of_wind_700', 'u_component_of_wind_850', 'u_component_of_wind_925', 'v_component_of_wind_50', 'v_component_of_wind_250', 'v_component_of_wind_500', 'v_component_of_wind_600', 'v_component_of_wind_700', 'v_component_of_wind_850', 'v_component_of_wind_925', 'temperature_50', 'temperature_250', 'temperature_500', 'temperature_600', 'temperature_700', 'temperature_850', 'temperature_925', 'specific_humidity_50', 'specific_humidity_250', 'specific_humidity_500', 'specific_humidity_600', 'specific_humidity_700', 'specific_humidity_850', 'specific_humidity_925'],
    in_vars = ['2m_temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity'],
    out_vars = ['2m_temperature', 'geopotential_500', 'temperature_850'],
    train_start_year = Year(1979),
    val_start_year = Year(2016),
    test_start_year = Year(2017),
    end_year = Year(2018),
    pred_range = Days(3),
    subsample = Hours(6),
    batch_size = 32,
    num_workers = 8,
)

print(data_module.train_dataset.inp_transform)
print(data_module.train_dataset[0][0].shape)
print(data_module.train_dataset.out_transform)
print(data_module.train_dataset[0][1].shape)

from climate_learn.models import load_model

model_kwargs = {
    "in_channels": 36, #len(data_module.hparams.in_vars),
    "out_channels": len(data_module.hparams.out_vars),
    "n_blocks": 19
}

optim_kwargs = {
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "warmup_epochs": 1,
    "max_epochs": 64,
}

resnet_model_module = load_model(name = "resnet", task = "forecasting", model_kwargs = model_kwargs, optim_kwargs = optim_kwargs)

# add_description
from climate_learn.models import set_climatology
set_climatology(resnet_model_module, data_module)

from climate_learn.training import Trainer, WandbLogger

resnet_trainer = Trainer(
    seed = 0,
    accelerator = "gpu",
    precision = 16,
    max_epochs = 64,
    # logger = WandbLogger(project = "climate_learn", name = "forecast-vit")
)

resnet_trainer.fit(resnet_model_module, data_module)

resnet_trainer.test(resnet_model_module, data_module)


# from climate_tutorial.utils import visualize

import os
import random
import numpy as np
from datetime import datetime
from plotly.express import imshow
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# more use cases for visualize, make a more intuitive api
# which split of the data? train/val/test? currently test with a random data point
# timestamp that we are visualizing?
# only timestamp works -- can infer the split, we don't have the timestamp info for now -> include it in the dataloader
# number: 5 data points

# add lat long information 
# plotly to zoom in

samples = ["2017-01-01:12", "2017-02-01:18"]
def visualize(model_module, data_module, split = "test", samples = 2):
    # dataset.setup()
    dataset = eval(f"data_module.{split}_dataset")

    if(type(samples) == int):
        idxs = random.sample(range(0, len(dataset)), samples)
    elif(type(samples) == list):
        samples = [np.datetime64(datetime.strptime(dt, "%Y-%m-%d:%H")) for dt in samples]
        idxs = [dataset.time.index(dt) for dt in samples if dt in dataset.time]
    else:
        raise Exception("Invalid type for samples; Allowed int or list[datetime.datetime or np.datetime64]")

    # print(dataset.time[idxs[0]])
    # row_titles = [datetime.strftime(None, "%Y-%m-%d:%H") for idx in idxs]

    if(data_module.hparams.task == "forecasting"):
        col_titles = ["Initial condition", "Ground truth", "Prediction", "Bias"]
    elif(data_module.hparams.task == "downscaling"):
        col_titles = ["Low resolution data", "High resolution data", "Downscaled", "Bias"]
    else:
        raise NotImplementedError

    fig = make_subplots(len(idxs), 4, subplot_titles = col_titles * len(idxs))
    for i, idx in enumerate(idxs):
        x, y, _, _, _ = dataset[idx] # 1, 1, 32, 64
        pred = model_module.forward(x.unsqueeze(0)) # 1, 1, 32, 64

        inv_normalize = model_module.denormalization
        init_condition, gt = inv_normalize(x), inv_normalize(y)
        pred = inv_normalize(pred)
        bias = pred - gt

        for j, tensor in enumerate([init_condition, gt, pred, bias]):
            fig.add_trace(imshow(tensor.detach().squeeze().cpu().numpy(), color_continuous_scale = "rdbu", x = dataset.inp_lon if i == 0 else dataset.out_lon, y = dataset.inp_lat if i == 0 else dataset.out_lat).data[0], row = i + 1, col = j + 1)
            # fig.colorbar(im, ax=ax)

    # fig.tight_layout()
    fig.show()

# visualize(resnet_model_module, data_module)