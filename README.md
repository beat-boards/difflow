# difflow

Calculate Beat Saber map dificultiy using TensorFlow Estimators

## Requirements

Anaconda is _strongly_ recommended.

### Training

* Python
* TensorFlow
* NumPy
* pandas
* SciKit Learn
* IPython
* Matplotlib

### Predicting

* Python
* TensorFlow
* NumPy

## Usage

### Training

`python train.py <LEARNING_RATE> <STEPS> <BATCH_SIZE> [<DATA_FILE>]`

`<LEARNING_RATE>` being the learning rate as a float, `<STEPS>` the number of steps to perform as an int and `<BATCH_SIZE>` the batch size as an int. `[<DATA_FILE>]` is an optional path/url to a csv file containing training data. The default will use the provided file.