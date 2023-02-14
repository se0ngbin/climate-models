# WeatherBench Probability: A benchmark dataset for probabilistic medium-range weather forecasting along with deep learning baseline models

## Overview
This paper presents:
- a probabilistic extension to original weatherbench paper 
	- set of probabilistic verification metrics
		- continuous ranked probability score
		- spread-skill ratio
		- rank histograms
	- state-of-the-art operation baseline using ECWMF IFS ensemble forecast
- experiments with 3 probabalistic models:
	- Monte Carlo dropout
	- parametric prediction
	- categorical prediction

## Evaluation Metrics
### Spread-Skill Ratio (ensemble reliability)
![spread formula](images/spread.png)
- measures reliability
- $\text{var}(f_{j,k})$: variance in the ensemble dimension
- perfectly reliable model has ratio of 1
	- smaller values = underdispersion (overconfidence)
	- larger values = overdispersion

### CRPS - continuous ranked probability score (ensemble sharpness/calibration)
![CRPS formula](images/CRPS.png)
- if forecast is deterministic, CPRS is just the mean absolute error

### Rank Histograms


## Probabilistic neural networks
### Some Info
- deep Resnet used as basic network architecture with only the final layer changed
- **input variables**: at time $t, t-6h, t-12h$
	- 3D
		- geopotential, temperature, zonal and meridional wind, humidity
		- 7 vertical levels: 50, 250, 500, 600, 700, 850, and 925 hPa
	- 2D
		- 2m temperature, 6hr accumulated preciptation, top-of-atmospher incoming solar radiation
	- constant (time invariant)
		- land-sea mask, orography, latitude at grid point
	- variables, levels, time steps stacked to create 114 channel input
	- all features normalized except precipitation
- train/val/test split (ERA5 at 5.625 deg [32 x 64 grid points])
	- train: 1979 to 2015
	- validation: 2016
	- test: 2017, 2018


### Monte Carlo Dropout
- dropout rates used: 0, 0.1, 0.2, 0.5
- drop out used during training, but also inference to create a stochastic prediction
	- created "50 random realizations for the test dataset to match the 50-member IFS ensemble"

### Parametric Prediction
- predict parameters of distribution
	- Z500, T850, T2M ~ $N(\mu, \sigma)$ ($\sigma$ is variance in this paper for some reason; worth noting is how variance is used in the closed form CRPS solution below)
- CRPS used as probabilistic loss function
  - closed form, differentiable solution of CRPS: ![formula](images/gaussiancrps.png)
- no stable distribution could be found for precipitation due to nature of 
- created randomly sampled 50-member ensemble for rank histogram (CRPS, spread-skill ratio calculated directly from predicted parametric distributions)


### Categorical Prediction
- divide value range into discrete bins then predict probability of each bin --> multi-class classification problem
- loss function: categorical cross-entropy (log-loss)
- trade-off between bin width (= probabilistic resolution) and number of bins
- due to limits in the number of channels and fluctuations in values, to avoid large bins, $\Delta t$ and $\Delta z$ predicted instead of $t,z$.
	- difference added for evalution
- separate training for each variable (empirically has yielded better results)

## Results
![results](images/wbprobaresultstable.png)
- mean RMSE and CRPS lowest for dropout rate of 0.1
- spread-skill ratio shows that dropout ensemble is severely underdispersive for all variables
- spread skill ratio close to one for parametric and categorical models


### Insights from rank histograms
- very strong U-shape for MC dropout
- parametric predictions are skewed --> high-bias. 
- categorical predictions do not have a bias and are only moderately U-shaped, suggesting good calibration

## Discussion

### Verification caveats
- RMSE/CRPS suffer from "double penalty problem", making them poor metrics for precipitation forecasts
- 5.625 degrees is a very "coarse" grid
- various sources of initial error warrant comparing each method against its own analysis, so as to start form zero error at $t=0$

### Parametric vs. categorial prediction
- both performed well as per verification scores (in line with previous research as well-performing)
- parametric approaches:
	- advantageous for supposedly well-known distributions (i.e. geopotential, temperature), 
	- problematic for unusually-distributed variables (i.e., precipitation)
- categorial approaches:
	- no assumptions on variable distribution
	- more values are estimated (>100) than the 2-3 parameters learned in the aforementioned approaches
	- introduction of "probabilistic resolution" in bin width ==> reframe problem to predict change (not absolute value)

### Spatially and temporally correlated forecasts
- separate distributions at each grid point + time point is fine for some applications, not for others
- Ex.: flood forecasting relies on cumulative rain distribution in entire catchment area
	- rain's variance in its characteristics make adding distributions in space/time unwise



## Link to Paper
- [https://arxiv.org/pdf/2205.00865.pdf](https://arxiv.org/pdf/2205.00865.pdf)
