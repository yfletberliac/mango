# Question Answering NLP with character-level RNN

The models will first be challenged with the bAbI dataset from FAIR.

Next, there will be experiments on the SQuAD dataset.

### Usage :

Query-Reduction Network with Char2Word:
```bash
python char2word_qrn.py
```

Query-Reduction Network without Char2Word:
```bash
python qrn.py
```

Entity Neural Network (first, run format_babi.py to preprocess the data):
```bash
bash run.sh
```
