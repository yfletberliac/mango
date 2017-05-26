# Question Answering NLP with character-level RNN

The model will first be challenged with the bAbI dataset from FAIR.

### Usage :

Query-Reduction Network without Char2Word:
```bash
python qrn.py
```

Query-Reduction Network with Char2Word:
```bash
python char2word_qrn.py
```

Entity Neural Network (first, run format_babi.py to preprocess the data):
```bash
bash run.sh
```
