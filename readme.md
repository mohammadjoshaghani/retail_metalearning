# The source code of the paper : "Retail time series forecasting using an automated deep meta-learning framework"
![](framework.png)

The code used wandb for hyper-parameter optimization. You can connect to your account by:

```bash
wandb login
```
You can run the optimization by:

```bash
python src/experiment.py
```
The directories `fforma` and `M0` contain the source code of the benchmark models.

The base-forecasters' code, in R, can be found from [this repository](https://github.com/Shawn-nau/retail-sales-forecasting-with-meta-learning).