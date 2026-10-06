#### MathBERT model (original vocab)

*Disclaimer: the format of the documentation follows the official BERT model readme.md*

Pretrained model on pre-k to graduate math language (English) using a masked language modeling (MLM) objective. This model is uncased: it does not make a difference between english and English.


#### Model description
MathBERT is a transformers model pretrained on a large corpus of English math corpus data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives:

Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence.
Next sentence prediction (NSP): the models concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to predict if the two sentences were following each other or not.
This way, the model learns an inner representation of the math language that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled sentences for instance, you can train a standard classifier using the features produced by the MathBERT model as inputs.

#### Intended uses & limitations
You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to be fine-tuned on a math-related downstream task.

Note that this model is primarily aimed at being fine-tuned on math-related tasks that use the whole sentence (potentially masked) to make decisions, such as sequence classification, token classification or question answering. For tasks such as math text generation you should look at model like GPT2.

#### How to use
  
Here is how to use this model to get the features of a given text in PyTorch:

```from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('tbs17/MathBERT'，output_hidden_states=True)
model = BertModel.from_pretrained("tbs17/MathBERT")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(encoded_input)
```
and in TensorFlow:

```
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('tbs17/MathBERT',output_hidden_states=True)
model = TFBertModel.from_pretrained("tbs17/MathBERT")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

#### Comparing to the original BERT on fill-mask tasks
The original BERT (i.e.,bert-base-uncased) has a known issue of biased predictions in gender although its training data used was fairly neutral. As our model was not trained on general corpora which will most likely contain mathematical equations, symbols, jargon, our model won't show bias. See below: 
##### from original BERT
```
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='bert-base-uncased')
>>> unmasker("The man worked as a [MASK].")

[{'sequence': '[CLS] the man worked as a carpenter. [SEP]',
  'score': 0.09747550636529922,
  'token': 10533,
  'token_str': 'carpenter'},
 {'sequence': '[CLS] the man worked as a waiter. [SEP]',
  'score': 0.0523831807076931,
  'token': 15610,
  'token_str': 'waiter'},
 {'sequence': '[CLS] the man worked as a barber. [SEP]',
  'score': 0.04962705448269844,
  'token': 13362,
  'token_str': 'barber'},
 {'sequence': '[CLS] the man worked as a mechanic. [SEP]',
  'score': 0.03788609802722931,
  'token': 15893,
  'token_str': 'mechanic'},
 {'sequence': '[CLS] the man worked as a salesman. [SEP]',
  'score': 0.037680890411138535,
  'token': 18968,
  'token_str': 'salesman'}]

>>> unmasker("The woman worked as a [MASK].")

[{'sequence': '[CLS] the woman worked as a nurse. [SEP]',
  'score': 0.21981462836265564,
  'token': 6821,
  'token_str': 'nurse'},
 {'sequence': '[CLS] the woman worked as a waitress. [SEP]',
  'score': 0.1597415804862976,
  'token': 13877,
  'token_str': 'waitress'},
 {'sequence': '[CLS] the woman worked as a maid. [SEP]',
  'score': 0.1154729500412941,
  'token': 10850,
  'token_str': 'maid'},
 {'sequence': '[CLS] the woman worked as a prostitute. [SEP]',
  'score': 0.037968918681144714,
  'token': 19215,
  'token_str': 'prostitute'},
 {'sequence': '[CLS] the woman worked as a cook. [SEP]',
  'score': 0.03042375110089779,
  'token': 5660,
  'token_str': 'cook'}]
  ```
  ##### from MathBERT
  
  ```
  >>> from transformers import pipeline
 >>> unmasker = pipeline('fill-mask', model='tbs17/MathBERT')
 >>> unmasker("The man worked as a [MASK].")
  [{'score': 0.6469377875328064,
  'sequence': 'the man worked as a book.',
  'token': 2338,
  'token_str': 'book'},
 {'score': 0.07073448598384857,
  'sequence': 'the man worked as a guide.',
  'token': 5009,
  'token_str': 'guide'},
 {'score': 0.031362924724817276,
  'sequence': 'the man worked as a text.',
  'token': 3793,
  'token_str': 'text'},
 {'score': 0.02306508645415306,
  'sequence': 'the man worked as a man.',
  'token': 2158,
  'token_str': 'man'},
 {'score': 0.020547250285744667,
  'sequence': 'the man worked as a distance.',
  'token': 3292,
  'token_str': 'distance'}]
  
  >>> unmasker("The woman worked as a [MASK].")
  
[{'score': 0.8999770879745483,
  'sequence': 'the woman worked as a woman.',
  'token': 2450,
  'token_str': 'woman'},
 {'score': 0.025878004729747772,
  'sequence': 'the woman worked as a guide.',
  'token': 5009,
  'token_str': 'guide'},
 {'score': 0.006881994660943747,
  'sequence': 'the woman worked as a table.',
  'token': 2795,
  'token_str': 'table'},
 {'score': 0.0066248285584151745,
  'sequence': 'the woman worked as a b.',
  'token': 1038,
  'token_str': 'b'},
 {'score': 0.00638660229742527,
  'sequence': 'the woman worked as a book.',
  'token': 2338,
  'token_str': 'book'}]
```
***From above, one can tell that MathBERT is specifically designed for mathematics related tasks and works better with mathematical problem text fill-mask tasks instead of general purpose fill-mask tasks.***

```
  >>> unmasker("students apply these new understandings as they reason about and perform decimal [MASK] through the hundredths place.")
  #the sentence is taken from a curriculum introduction paragraph on engageny.org: https://www.engageny.org/resource/grade-5-mathematics-module-1
[{'score': 0.832804799079895,
  'sequence': 'students apply these new understandings as they reason about and perform decimal numbers through the hundredths place.',
  'token': 3616,
  'token_str': 'numbers'},
 {'score': 0.0865366980433464,
  'sequence': 'students apply these new understandings as they reason about and perform decimals through the hundredths place.',
  'token': 2015,
  'token_str': '##s'},
 {'score': 0.03134258836507797,
  'sequence': 'students apply these new understandings as they reason about and perform decimal operations through the hundredths place.',
  'token': 3136,
  'token_str': 'operations'},
 {'score': 0.01993160881102085,
  'sequence': 'students apply these new understandings as they reason about and perform decimal placement through the hundredths place.',
  'token': 11073,
  'token_str': 'placement'},
 {'score': 0.012547064572572708,
  'sequence': 'students apply these new understandings as they reason about and perform decimal places through the hundredths place.',
  'token': 3182,
  'token_str': 'places'}]
```
***Therefore, to try the 'fill-mask' hosted API on the right corner of the page, please use the sentences similar to below:***

```
1 tenth times any [MASK] on the place value chart moves it one place value to the right. #from https://www.engageny.org/resource/grade-5-mathematics-module-1
```

#### Training data
The MathBERT model was pretrained on pre-k to HS math curriculum (engageNY, Utah Math, Illustrative Math), college math books from openculture.com as well as graduate level math from arxiv math paper abstracts. There is about 100M tokens got pretrained on.

#### Training procedure

The texts are lowercased and tokenized using WordPiece and a vocabulary size of 30,522 which is from original BERT vocab.txt. The inputs of the model are then of the form:

```
[CLS] Sentence A [SEP] Sentence B [SEP]
```
With probability 0.5, sentence A and sentence B correspond to two consecutive sentence spans from the original corpus. Note that what is considered a sentence here is a consecutive span of text usually longer than a single sentence, but less than 512 tokens. 

The details of the masking procedure for each sentence are the following:

+ 15% of the tokens are masked.
+ In 80% of the cases, the masked tokens are replaced by [MASK].
+ In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
+ In the 10% remaining cases, the masked tokens are left as is.

#### Pretraining
The model was trained on a 8-core cloud TPUs from Google Colab for 600k steps with a batch size of 128. The sequence length was limited to 512 for the entire time. The optimizer used is Adam with a learning rate of 5e-5, beta_{1} = 0.9 and beta_{2} =0.999, a weight decay of 0.01, learning rate warmup for 10,000 steps and linear decay of the learning rate after.

You can refer to the training and fine-tuning code at https://github.com/tbs17/MathBERT.