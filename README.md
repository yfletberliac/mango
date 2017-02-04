# Mango
## Question Answering NLP with character-level RNN

... WORK IN PROGRESS ...

The model will first be challenged with the bAbI dataset from FAIR.

### Files :

- **format_babi.py** - processes the dataset for character-level handling
- **/datasets** - repository for datasets

- **/model/dataset.py** - loads the data
- **/model/model.py** - constructs model_fn that will be fed into the Estimator in main.py
- **main.py** - runs TensorFlow instances to train and evaluate the model
- **run.sh** - runs main.py given a dataset and a seed

### Usage :

```bash
bash run.sh
```
