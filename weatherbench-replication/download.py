from pathlib import Path
import sys
sys.path.insert(0, '/u/home/s/se0ngbin/climate_tutorial')

path = Path("/u/scratch/s/se0ngbin/weatherbench")

from climate_tutorial.data import download

download(root = path, source = "weatherbench", variable = "2m_temperature", dataset = "era5", resolution = "5.625")
download(root = path, source = "weatherbench", variable = "geopotential_500", dataset = "era5", resolution = "5.625")
download(root = path, source = "weatherbench", variable = "temperature_850", dataset = "era5", resolution = "5.625")
download(root = path, source = "weatherbench", variable = "geopotential", dataset = "era5", resolution = "5.625")
download(root = path, source = "weatherbench", variable = "temperature", dataset = "era5", resolution = "5.625")


