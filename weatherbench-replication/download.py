from pathlib import Path
import sys
sys.path.insert(0, '/home/seongbin/climate_tutorial')

path = Path("/data0/datasets/weatherbench")

from climate_tutorial.data import download
from climate_tutorial.data import regrider

# download(root = path, source = "weatherbench", variable = "2m_temperature", dataset = "era5", resolution = "5.625")
# download(root = path, source = "weatherbench", variable = "geopotential_500", dataset = "era5", resolution = "5.625")
# download(root = path, source = "weatherbench", variable = "temperature_850", dataset = "era5", resolution = "5.625")
# download(root = path, source = "weatherbench", variable = "geopotential", dataset = "era5", resolution = "5.625")
# download(root = path, source = "weatherbench", variable = "temperature", dataset = "era5", resolution = "5.625")
# download(root = path, source = "esgf", variable = "temperature", dataset = "cmip6")
# download(root = path, source = "esgf", variable = "geopotential", dataset = "cmip6")

regrider(root = path, source = "esgf", variable = "geopotential", dataset = "cmip6", resolution = "5.625")
regrider(root = path, source = "esgf", variable = "temperature", dataset = "cmip6", resolution = "5.625")

# https://esgf-node.llnl.gov/search/cmip6/