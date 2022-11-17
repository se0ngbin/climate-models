from pathlib import Path
import sys
sys.path.insert(0, '/home/seongbin/climate_learn')

path = Path("/data0/datasets/weatherbench/")

from climate_learn.data import download

download(root = path, source = "climatebench", dataset = "cmip6")

