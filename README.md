# Required Packages

To run our code, you will need the following packages:
## System environment
- Python 3.7 to 3.10
## Python packages
- torch==1.9.0+cu102
- numpy==1.19.4
- scikit-learn==0.24.2
- torch_scatter==2.0.8
- cluster==1.5.9
- scipy==1.5.4
- matplotlib==3.4.3
- pandas==1.1.4
- tqdm==4.62.2
- numba==0.53.1
- seaborn==0.11.2
- pynvml==11.2.147
- sparse==0.13.0
- geometric==1.7.2
- einops==0.3.0

# Datasets

We prepared to run our code for Raindrop as well as the baseline methods with two healthcare and one human activity dataset.

## P19 (PhysioNet Sepsis Early Prediction Challenge 2019)

Dataset link: https://doi.org/10.6084/m9.figshare.19514338.v1

## P12 (PhysioNet Mortality Prediction Challenge 2012)

Dataset link: https://doi.org/10.6084/m9.figshare.19514341.v1

## PAM (PAMAP2 Physical Activity Monitoring)

Dataset link: https://doi.org/10.6084/m9.figshare.19514347.v1

We organize the well-processed and ready-to-run data in the same way for three datasets.


# Running the Code
```bash
cd code
python GK.py
``` 
You can run our code with the following command:

Alternatively, if you're using Jupyter Notebook, run `%run GK.py` in a code cell. This method is more flexible and will automatically preserve your experimental results.

You can run baseline models by navigating to `code/baselines`.