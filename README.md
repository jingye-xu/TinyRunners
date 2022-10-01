# TinyRunners
This repository stores the codes for joining the 2022 TinyML Design Contest @ ICCAD

## Contents

[1. Training](README.md#training)

[2. C codes for deployment](README.md#c-codes-for-deployment)

[3. How to test our model](README.md#How-to-test-our-model)


## Training

Training repository can alse be seen here:

[https://github.com/jingye-xu/TinyRunners](https://github.com/jingye-xu/TinyRunners)

### Data Preparation

`data` folder stores the compressed `data.gz` file that stores all data.

There are also some functions that can be called individually to help analyze to dataset.

### Model design
Model design and model structure is stored inside `NN/model/`

Besides, the dataload function for training using pytorch stores here.

### Training

To train the model, make sure you are in the root path of this repository, and run command:

```bash
python NN/train.py
```

### Logging and saved model

logging files and saved model are stored at lightning_logs, you can easily use tensorboard to read the logs.

## C Codes for deployment

## How to test our model
