# T5 Model (allowing you to train t5 model for different tasks)😁


**HOW TO USE**

- [configurations](#configuration-settings)
- [Training](#training)
- [Testing](#testing)
- [real time inference](#inference)
- [api](#api-usage)


## CONFIGURATION SETTINGS

first thing first **(i'll want to clone the github repo first)**, check the yaml configurations file located at ``` src/conf/configs.yml ``` the configs file provides all the settings needed for training, testing and inferencing the t5 model, at the configs file, you can change the settings for your suitable need, e.g batch size, training epochs, paths to datas, model name, etc.


## TRAINING
let's see how the config file looks like, 

```yaml
|-- src
|   |--conf
|      |--config.yml:
|                    model_configs:
|                      tokenizer_path: 
|                      model_name: t5-base
|                      path_to_save: /path/to/save model/
|                      model_path: /path/to/model
|
|                    training_configs:
|                      epochs: 20
|                      batch_size: 2
|                      eval_batch_size: 2
|                      reduction_factor: 0.1
|                      patience: 3
|                      optimizer: Adam
|                      weight_decay: 0.0
|                      shuflle_data: True
|                      evaluate_dev: True
|                      use_cuda: False
|                      betas: [0.9,0.98]
|                      eps: 0.000000009
|                      lr : 0.0003
|                      min_lr: 0.00000001
|                      fp16: False 
|                                
|                    data_configs:
|                      train_path: /path/to/training_data.csv
|                      dev_path: /path/to/testing_data.csv
|                      prefix: 'summarize: '
|                      max_input_len: 512
|                      max_output_len: 150
```
at the ```model configs```, if you have a tokenizer and model already, provide the path to tokenizer and the model, the implementation will load the model automatically from the path you provide and continue training, this will allow you to be retraining your model. if however, you want to start training from scratch, leave the ```tokenizer_path``` and the ```model_path``` empty, the implementation will load the tokenizer and model from the huggingfacehub, just make sure you give the name (t5-base or t5-small) at the ```model_name```😅. at the ```training_configs``` you can change the batch_size, epoch, etc.

the implementation expect the training and testing data to be in a csv format with columns {'text':[...],'summary':[...]}, if you want to change the columns, and you know what your doing, go to the dataprocessor.py and change the columns to call, to suit your task (translation, summarization, etc).

assuming the ```config.yml``` is set, go to the src directory and type the following at the terminal:

```cmd
python finetune.py
```
this command will start finetuning your t5 model on your own dataset, and saving the best model at the ```path_to_save``` directory each time the model achieves best result or higher bleu score than previous one.

## TESTING
after you finished training your model, you can test the model by setting the ```dev_path``` to your evaluation dataset, just type the command below at the terminal:

```cmd
python evaluate.py

```

if you want to see all the logging historys, go to the ```logs.log``` file

## INFERENCE
loading...


## API USAGE
loading...
...
