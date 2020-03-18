# Question Answering NLP with character-level RNN

The different models will first be challenged with the bAbI dataset from FAIR and the SQuAD dataset from Stanford.

### Content:

Name | Description
--- | ---
/babi| babi dataset and utilities
/squad | squad dataset and utilities
/qrn | contains the QRN cell
char2word.py | Char2Word-only module (on bAbI dataset)
qrn.py | implementation of the QRN model (on bAbI dataset)
char2word_qrn.py | implementation of the QRN model w/ Char2Word module (on bAbI dataset)
mango_squad.py | implementation of the QRN model w/ Char2Word module (on SQuAD dataset)



### Usage (with the bAbI dataset):

Query-Reduction Network with Char2Word:
```bash
python char2word_qrn.py
```

Query-Reduction Network without Char2Word:
```bash
python qrn.py
```

### Usage (with the SQuAD dataset):

Query-Reduction Network with Char2Word:
```bash
python mango_squad.py
```
