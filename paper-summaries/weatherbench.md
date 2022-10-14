# WeatherBench - A benchmark dataset for data-driven weather forecasting

## Overview
This paper presents:
- a benchmark dataset for data-driven medium-rnage weather forasting
- simple and clear evaluation metircs to enable direct comparisons between ML models/methods
- baseline scroes from:
	- simple linear regression
	- deep learing
	- purely physical forcasting


## General Vocabulary / Abbreviations
- **NWP:** numerical weather prediction
- **lead time:** time between forecast and actual event
- **geopotential:** gravitational potential at certian alutitude
- **hPa:** unit of pressure. commonly used as vertical coordinate instead of physical height
	- sea level = ~100 hPa
	- decreases roughly exponentially with height
- **Z500:** geopotnetial at 500 hPa pressure
- **2 meter temperature:** temperature 2m above ground
- **climate runs:** long series of consecutive forecasts
- **reanalysis dataset:** provides the best guess of atmospheric state by combining forecast model with available observations

## Numerical Weather Prediction (NWP)
- model currently used for weather/climate predictions
- purely physical computer models; equations are solved on a discrete numerical grid
- many shortcomings; huge amounts of computing power required


## Previous Work using ML
- **post-processing:** correction of statistical biases in the output of physical models
- **statistical forecasting:** prediction of variables not directly output by physical models
	- traditionally done via linear techniques
	- recently random forests and neural networks have been explored - target very specific variables; general predictions still made by physical model
- **nowcasting:** short range (< 6hr) prediction of precipitation by direct extrapolation of radar observations
- **hyprid modeling:** physical model + data-drive components
	- only replace uncertain or computationally expensive components with ML emulators
	- drawbacks: 
		- interaction between physical and ML components poorly understood - can lead to unexpected instabilites / biases
		- difficult to implement technically (must interface ML components with complex climate model code)
		
### Dueben and Bauer (2018)
#### Model(s)
- 1 fully connected neural network, 1 spatially localized network similar to CNN
- predict difference from one time frame to next (iterative)
- third order Adams-Bashford explicit time-stepping scheme used
#### Training Data
- taken from ERA5 archive (2010~2017)
- regridded to 6 degree lat-lng grid
#### Results
- predicted Z500 and 2m temperature up to 120 lead time; 10 month validation period
- CNN predicting only Z500 performed best but was unable to beat low-resolution operational NWP model

### Scher (2018) & Scher  and Messori (2019b)
#### Model(s)
- CNNs with encoder-decoder setup with lead time of several days
	- input: instantaneous 3D model fields at timestep
	- output: same fields at some time later
- 2018: separate network trained for each lead-time up to 14 days
- 2019b: trained only on 1-day forecasts; longer forecasts via iteration
#### Training Data
Used simpified reality setting (via General Circulation Models)
- 2018: highly simplified GCM without hydrological cycle
- 2019b: more realistic and complex GCMs
#### Results
- models evaluated using RMSE and anomally correlation coefficient
- networks trained to directly predict a specific forecast time outperformed iterative networks
- architectures tuned on simplfied GCMs also work on more complex ones
- 2018: achieved high performance; able to create stable climate runs
- 2019b: relatively good short-term forecasting; unable to create stable and realistic climate runs 

### Weyn et al. (2019)
#### Model(s)
-  deep CNNs with encoder-decoder setup (similar to Scher)
- added convolution long short term memory (LSTM) hidden layer
- iterative forecasting
#### Training Data
- taken from Climate Forecast System Reanalysis (1979-2010)
- 2.5 deg horizontal resolution; cropped to northern hemisphere
#### Results
- predicted Z500 and 700-300 hPa thickness
- using 2 input time steps (6hr apart) and predicting 2 outputs iteratively performed better than single step
- best CNN "outperforms a climatology benchmark at up to 120hr lead time and appears to correctly asymptote towards persistence forecasts at longer lead times up to 14 days"


## Choices Made by Authors
- lead times of 3 and 5 days
	- atmospher still reasonably deterministic but also exhibits complex nonlinear behavior
	- delivers cruical information for disaster (flood, heat/cold spells, winds) prepartion
	- benchmark cloesly emulates task performed by physical NWP models
- purely data-driven approach
	- not as computationally expensive as NWP - may enable cheaper forecasts
	- by learning from a diverse set of data sources, may outperform physical models some areas
- focus on upper-level fields of pressure and temperature, where physical models perform very well


## Benchmark Dataset (ERA5 reanalysis dataset)
- can have huge impact 
	- enables quantitive comparison of different algorithms
	- fosters constructive competition
- since dataset is very large, it was regrided to lower resolutions in paper
	- 5.625, 2.8125, 1.40525 degrees resolution
	- regridding done via bilinear interpolation
- selected pressure levels are commonly used by climate models --> useful for pretrianing

### Variables
![variables](https://github.com/se0ngbin/climate-models/blob/main/paper-summaries/images/weatherbench-fields.jpeg)
- soil_type: 7 different soil categories


## Evaluation
- evaluation done for years 2017 and 2018 (2016 for validation); 3 and 5 day lead time
- 5.625 deg resolution
	- predictions at higher resolutions have to be downscaled to evalution resolution
	- baselines at higher resolutions found to be almost identical (little info is lost by evaluating at coarser resolution)
- primary verification fields
	- Z500
	- T850: temperature at 850 hPa, which is high enough to not be affected by diurnal variations, but provides info about broader temperature trends
- metric: RMSE + latitude weighted anomaly correlation coefficient and mean absolute error
	- metric matters less for smooth fields (Z500, T850)
	- metric matters more for itermittent fields (precipitation, etc.)

### Baseline Results
![results](https://github.com/se0ngbin/climate-models/blob/main/paper-summaries/images/weatherbench-results.png)
- **persistence:** fields at initialization time used as forecasts
- **climatology:** single mean over all times in training set
- **weekly climatology:** mean over each of the 52 calendar weeks
- **Operational Integrated Forecast System (IFS):** gold standard of medium-range NWP; computationally very expensive
- **IFS T42, T63:** IFS on coarser horizontal resolutions (T42 < T63 in terms of resolution)
	- less computationally expensive
	- T42: initalized from ERA5
	- T63: initialized from operational IFS
- **direct linear regression:** separate model trained for each of the 4 variables
- **iterative linear regression**
	- single linear regression (fields were concatenated)
	- 6 hour increments
- **simple CNNs**
	- 5 layers
		- each hidden layer with 64 channels with convolutional kernel of size 5 and ELU activations
		- input and output layers have 2 channels, one for Z500 and one for T850
	- trained using the Adam optimizer and mean squared error loss
	- total number of trainable parameters: 313858
	- preiodic convolutions only in lng direction
	- iterative version created by chaining together 6 hourly predictions

### Other Findings
- CNN forecasts for 6h lead time not able to capture wave-like patterns caused by atmospheric tides --> hints at a failure to capture the basic physics of the atmosphere.
	- can be captured by IFS operational forecast
- for 5 days forecast time the CNN model predicts unrealistically smooth fields. Lkely caused by either of the following:
	- the two input fields contain insufficient information
	- at 5 days the atmosphere already shows some chaotic behavior, causing a model trained with a simple RMSE loss to predict smooth fields.

## Link to Paper
- [weatherbench-web-link](https://arxiv.org/abs/2002.00469)

