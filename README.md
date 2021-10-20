# 360-Degree Gaze Estimation in the Wild Using Multiple Zoom Scales

## About
This code is for the paper `360-Degree Gaze Estimation in the Wild Using Multiple Zoom Scales`. By using this code you agree to terms of the [LICENSE](LICENSE).

## Setting up the environment. 
Use the conda to create a new environment using the given [.yml](https://github.com/ashesh-0/MultiZoomGaze/blob/main/multizoomgaze_env.yml) file.  In case `conda` is not installed on your system, you can install it from [here](https://docs.conda.io/en/latest/miniconda.html). Once conda is installed, use the following code to create an environment with all dependencies installed. 

```
conda env create -f multizoomgaze_env.yml
```

## Setting up the [Gaze360](http://gaze360.csail.mit.edu/iccv2019_gaze360.pdf) database.
Register [here](http://gaze360.csail.mit.edu/download.php) which will then give you access to the database.


## Downloading the pre-trained checkpoint files.
Download the checkpoint files from [here](https://drive.google.com/drive/folders/1ORsJMSiL0b7yEXCPENidhfTdNhUfJtNw).


## Evaluating the model performance on Gaze360 dataset.

**MSA+Seq**
```
python run.py --model_type=NonLstmSinCosModel  --enable_time --checkpoints_path=CKECKPOINT_DIRECTORY/ --source_path=/data/GAZE360/imgs/ --evaluate
```

**MSA**
```
python run.py --model_type=NonLstmSinCosModel --checkpoints_path=CKECKPOINT_DIRECTORY/ --source_path=/data/GAZE360/imgs/ --evaluate
```

**MSA+raw**
```
python run.py --model_type=NonLstmMultiCropModel --checkpoints_path=CKECKPOINT_DIRECTORY/ --source_path=/data/GAZE360/imgs/ --evaluate
```

**Pinball Static**
```
python run.py --model_type=StaticModel --checkpoints_path=/home/ashesh/gaze_final_checkpoints/ --evaluate
```

Here, `checkpoints_path` is the directory where you've saved the trained checkpoint files. `source_path` is the directory which contains gaze360 data.

## Training the model
Just remove the `--evaluate` token from the command for evaluating the model performance which is given in the previous section. In this case, `checkpoints_path` will be the path where your checkpoints will get saved. 