# -*- coding: utf-8 -*-
"""

# Fine-tuning Pre-trained Language Models

- **Overview**: In this assignment, we will explore how to take pre-trained language models and adapt them to do different natural language tasks. We'll cover:
   - Applying models to down-stream tasks without extra training
   - Fine-tuning for sequence classification
   - Fine-tuning for sequence pair classification
   - Fine-tuning for token classification

Recommended Readings
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. NAACL 2019.
- [Generalized Language Models](https://lilianweng.github.io/posts/2019-01-31-lm/) Lillian Weng. 2019
- [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/) Jay Alammar. 2019


## To get started, make a copy of this colab notebook into your google drive!

Also make sure you're using the GPU by going to `Runtime` -> `Change Runtime Type` and making sure that `Hardware Accelerator` is set to `GPU`. This will save you significant amounts of time when training models
"""
# Commented out IPython magic to ensure Python compatibility.
# %env CUBLAS_WORKSPACE_CONFIG=:4096:8

# Commented out IPython magic to ensure Python compatibility.
# ## DO NOT CHANGE ANYTHING, JUST RUN
# %%capture
# !pip install penngrader-client

# Commented out IPython magic to ensure Python compatibility.
# %%writefile notebook-config.yaml
#
# grader_api_url: 'https://23whrwph9h.execute-api.us-east-1.amazonaws.com/default/Grader23'
# grader_api_key: 'flfkE736fA6Z8GxMDJe2q8Kfk8UDqjsG3GVqOFOa'

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from collections import defaultdict
from dill.source import getsource
import copy
import torch
import numpy as np
import evaluate
from evaluate import evaluator
from datasets import load_dataset
import datasets
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import pipeline, AutoModelForMaskedLM
import locale
import logging
!cat notebook-config.yaml

# Comment this out if you want to enable huggingface warnings
logging.disable(logging.WARNING)

# This fixes colab's default encoding to match huggingface accelerate
locale.getpreferredencoding = lambda x=False: "UTF-8"

"""# Introduction: HuggingFace Transformers

In this homework we will be making use of the [HuggingFace](https://huggingface.co/) transformers library. This library provides support for downloading, running, and training language models and is an essential tool for professionals that want to deploy NLP technology.

We will be installing four huggingface libraries:

1. [Transformers](https://huggingface.co/docs/transformers/index) (Model inference): `!pip install transformers`
2. [Accelerate](https://huggingface.co/docs/accelerate/index) (Model training): `!pip install accelerate`
3. [Datasets](https://huggingface.co/docs/datasets/index) (Data processing): `!pip install datasets`
4. [Evaluate](https://huggingface.co/docs/evaluate/index) (Model evaluation): `!pip install evaluate`

We highly recommend that you reference the documentation linked above frequently as it will be a very valuable resource throughout this homework
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install -U huggingface_hub fsspec
# !pip install transformers
# !pip install --upgrade accelerate
# !pip install evaluate
# !pip install datasets==3.6.0


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_id = 0 if str(device) == 'cuda' else -1

print(device_id)

"""# Section 1: Zero-Shot Application of Pre-Trained Language Models (15 points)

In this homework we'll be exploring different ways of using pre-trained language models to accomplish natural language tasks. We will be using BERT as our pre-trained language model for this homework.

BERT was introduced in late 2018 by the landmark paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). It was trained via "Masked Language Modeling" a procedure that involves training the model to fill in [MASK] tokens in input sequences.

<!-- We will be using the "base" size BERT model (110 million parameters) and we will use the "cased" version (i.e. the model is trained to distinguish between upper and lower case letters).  -->

As you can see below, according to BERT, there is an 84.7% chance that the masked word is "Italian".

<!-- Note: We are running BERT via a [HuggingFace Pipeline](https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/pipelines) object. Here we are using a `fill-mask` pipeline but many others exist and we will be using them extensively in this assignment. While this is not the only way to query a model with HuggingFace, it is the most convenient. -->
"""

# Download and query the bert-base-cased model
bert_model = pipeline('fill-mask', model='bert-base-cased', device=device_id)
print(bert_model("I went to an [MASK] restaurant and ordered pasta."))

"""In *theory* we should be able to use BERT to do natural language tasks such as Question Answering, Language Identification, Part-of-Speech Tagging, and even Translation by formulating our task in the style of a fill-in-the-blank sentence."""

# Language Identification
print(bert_model("I am currently speaking in the [MASK] language.")[0])

# Factual QA
print(bert_model(
    "The Declaration of Independence was written in the year [MASK].")[0])

# Part of speech tagging
print(bert_model("The word run is a [MASK].")[0])

# Translation
print(bert_model("The French word amour translates to [MASK] in English.")[0])

"""However, it turns out that BERT really isn't very good at doing these tasks without extra training (as you can see below). In Section 1 of this homework we'll evaluate BERT without extra training on sentiment analysis to get an idea of where the base model is at. Then, in the next sections, we'll train the model and see how much we can improve the performance."""

# Language Identification (failed)
print(bert_model("今[MASK]で話しています")[0])

# Factual QA (failed)
print(bert_model("The U.S.A. was founded in the year [MASK].")[0])

# Part of speech tagging (failed)
print(bert_model("The word golf is a [MASK].")[0])

# Translation (failed)
print(bert_model(
    "The French word bonjour translates to [MASK] in English.")[0])

"""### **Sequence Classification with pre-trained BERT**

In this section we'll be evaluating BERT for sentiment analysis without fine-tuning. This is purely for the purposes of demonstration as you'll be able to see the difference between what BERT does before and after fine-tuning.

In order to do a sequence classification task without fine-tuning BERT you need two things. You need
1. A mask-filling template to attach on to the end of the sequence
2. A set of `targets` which consist of the class labels.

Below you'll see an example of this procedure applied to sentiment analysis. For the sentence "I just saw a movie today and it was really great", we see that BERT outputs 2.78% chance of the `[MASK]` token being "positive" and 1.65% chance of it being "negative", thus we give the example a positive label.
"""

bert_model("I just saw a movie today and it was really great. My opinion of the movie is [MASK].", targets=[
           "positive", "negative"])

"""Let's test BERT's ability to recognize sentiment using the [Yelp Reviews](https://huggingface.co/datasets/yelp_review_full) dataset. This dataset consists of 650,000 reviews with their user-assigned star ratings. The task is to determine how many stars (1 to 5) the user gave given the text of their review."""

# Download the yelp dataset from huggingface
dataset = load_dataset("yelp_review_full")

"""### **Dataset Processing** (5 points)

There are many functions we can use to process a HuggingFace dataset object. One of the most useful is the [Map](https://huggingface.co/docs/datasets/process#map) function. This function applies some function `f` to every row of the dataset where `f` takes in a row of the dataset and returns a dictionary containing the new columns of the data (or any edits to existing columns). For example, given a dataset with a column `binary_label` that contains a binary label (0 or 1) you can create a new column with the opposite label by writing either one of the following functions:
```
def reverse_label(row):
  return {'reverse_label': int(not row["binary_label"])}

def reverse_label(row):
  row['reverse_label'] = int(not row[''binary_label"])
  return row
```
and then calling map on the dataset with the function
```
new_data = dataset.map(reverse_label)
```

### **Concatenating the Mask-Fill Template** (5 points)
Your first task will be to concatenate the mask-fill template onto every item in the `text` column of the dataset and add the output to the dataset as a new column named `input`.

Refer back to the documentation for the [Map](https://huggingface.co/docs/datasets/process#map) function if you're stuck
"""


def concatenate_mask(row):
    '''
      Pseudocode:
          1. Concatenate this template string to the 'text' field of the input row.
          2. Add the resulting concatenated string as a new field 'input' in the row dictionary.

      Input:
          row: A dictionary representing a single row of the dataset.
               It contains at least the key "text" which holds a string of text.

      Returns:
          The input row dictionary, but with an added key "input" that contains the text
          with the template appended.
    '''
    # TODO: Concatenate the template to each example and return in a new column "input"
    template = " I give it a score of [MASK] out of 5."
    row["input"] = row["text"] + template
    return row


test_data = dataset['test'].map(concatenate_mask)

"""### **Intro to the BERT Tokenizer**
Before taking in a sequence of text, language models need to break up the sequence into tokens. Since BERT uses the [Transformer](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model) architecture, it has a fixed maximum length of tokens it can process at any one time (that being 512). If you pass in a sequence of text longer than 512 tokens, BERT will throw an error:
"""

# Giving BERT a piece of text with more than 512 tokens produces an error
try:
    bert_model("Hello " * 512 + " my name is [MASK]")
except RuntimeError as e:
    print(e)

"""### **Filtering the data for BERT's max length** (5 points)
Below we have provided code to load the BERT tokenizer. To run the tokenizer you can call it on a piece of text like `tokenizer("Hello students")`. This will return a dictionary with three arrays, `input_ids`, `attention_mask` and `token_type_ids`. The length of all three arrays should be equal to the total number of tokens in the sequence.

Since we need to make sure that no example in our input is longer than 512 tokens, your assignment now is to filter the dataset to only include examples that have 512 or less tokens. You can do this using the [Filter](https://huggingface.co/docs/datasets/process#select-and-filter) function. This function is similar to Map but returns a boolean where `True` indicates we should keep the row and `False` indicates we should remove it.

NOTE: The maximum sequence length of any model can be found through the `tokenizer.model_max_length` variable. If you dislike magic numbers in your code, you can use this instead of hard-coding 512.
"""

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def filter_for_max_length(row):
    '''
      Pseudocode:
          1. Tokenize the "input" field in the row using the BERT tokenizer.
          2. Get the number of tokens in the tokenized input.
          3. Compare the number of tokens to the maximum length allowed by the BERT model.
          4. Return True if the number of tokens is less than or equal to the maximum length, otherwise return False.

      Input:
          row: A dictionary representing a single row of the dataset.
                It contains at least the key "input" which holds a string of text.

      Returns:
          A boolean value indicating whether the length of the tokenized "input" is
          less than or equal to the maximum length allowed by the BERT model.
    '''

    # Return True if the length of the input (in tokens) is less than
    # or equal to the maximum length allowed by the BERT model
    tokenized = tokenizer(
        row["input"], truncation=False, add_special_tokens=True)
    num_tokens = len(tokenized["input_ids"])
    return num_tokens <= tokenizer.model_max_length


filtered_data = test_data.filter(filter_for_max_length)

"""Here we'll be randomly selecting 100 examples from our filtered dataset for testing purposes. We'll be doing this throughout this assignment both for training and testing to ensure runtime stays fast. In the real world, given unlimited time, we would ideally like to use the full datasets."""

# Randomly sample 100 examples from the dataset (do not change!)
sampled_data = filtered_data.shuffle(seed=42).select(range(100))

"""### **Apply BERT to each example** (5 points)
As your final task for this section you must write a function that runs the BERT model on every sequence of text in the dataset's `input` column and outputs the most likely integer star rating (1-5) as predicted by BERT in a new column named `score`.

Refer back to the earlier parts of this Section for how to use BERT and make sure to use the `targets` field to specify the five possible star ratings `[1, 2, 3, 4, 5]`.
"""

model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
model.eval()  # Set model to evaluation mode


def apply_bert(row):
    '''
    Pseudocode:
        1. Use the BERT model to compute the probabilities for each possible score (1 to 5).
        2. Identify the score with the highest probability.
        3. Return the most likely score in a new dictionary with the key "score".

    Input:
        row: A dictionary representing a single row of the dataset.
              It contains at least the key "input" which holds a string of text.

    Returns:
        A dictionary with the key "score" holding the most likely score (1 to 5) as an integer.
    '''
    # Compute the probabilities for each score and return the value of the most likely score
    with torch.no_grad():
        inputs = tokenizer(row["input"], return_tensors="pt")
        mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(
            as_tuple=True)[1]

        outputs = model(**inputs)
        logits = outputs.logits

        # Extract logits at the [MASK] token position
        mask_token_logits = logits[0, mask_token_index, :].squeeze()

        # Scores of interest: tokens for "1" to "5"
        scores = {}
        for i in range(1, 6):
            token = str(i)
            token_id = tokenizer.convert_tokens_to_ids(token)
            scores[i] = mask_token_logits[token_id].item()

        # Get the score with the highest logit
        predicted_score = max(scores, key=scores.get)

        return {"score": predicted_score}


predicted_data = sampled_data.map(apply_bert)

"""### **Evaluating the Model**

Below we've written code to evaluate the model's accuracy. If you've done everything correctly you should get somewhere around 24% accuracy. Given that this task is five-way classification, this is only *barely* better than random chance! Clearly BERT cannot do sentiment analysis without extra training. Let's see if fine-tuning will help.


"""


def accuracy(outputs, reference):
    return sum([o == r for o, r in zip(outputs, reference)]) / len(reference)


print(accuracy(predicted_data["score"], predicted_data["label"]))

"""# Section 2: Training BERT for Sentiment Analysis (20 points)

In this section we'll be training BERT for the same Sentiment Analysis task from Section 1. We'll start by giving a short introduction to the concept of fine-tuning.

### **What is fine-tuning?**

Fine-tuning is the process of further training a pre-trained model to accomplish a particular down-stream task. This is typically done by swapping out the pre-trained model's last layer (its head) for a new randomly initialized head and training both model and head.

### **Model "heads"**

The final output layer of a language model is typically called a "head". The standard head used by models when pre-training is called a "language modeling head". This is a dense linear layer that projects the $D_{enc}$ length encodings of each of the $L_{context}$ tokens to a probability distribution over the vocabulary. A single transformation matrix is learned and applied to all context tokens, which are treated as a batch dimension. The total size of this layer is thus ($D_{enc}$ x $|V|$).
<!-- The head and the network are always trained *together*. We compute the cross-entropy loss $L = -\log(P(w))$ using the output of the head and backpropagate through the head to the rest of the network. -->

### **Adding on a "Classification head"**

In Section 1 we used BERT's language modeling head to perform the Sentiment Analysis task. While we *can* train the model with its original LM head, this is unnecessary, as the LM head outputs probabilities over all tokens rather than just our targets. Thus, in this section we're going to remove the language modeling head and replace it with a classification head.

A classification head is very similar to a language modeling head. It is a dense linear layer that projects the $D_{enc}$ length encodings of each of the $L_{context}$ tokens to a probability distribution over the *$n$ classes* rather than the whole vocabulary. The transformation matrix is again shared, with the context tokens treated as a batch dimension. The total size of this layer is thus ($D_{enc}$ x $n$). This makes it significantly more efficient to train.

### **Using HuggingFace to swap heads**

Loading a model with a particular head is easy using HuggingFace. All you do is load the model as an instance of a given class. The classes we're using in this homework are as follows:
1. [AutoModelForMaskedLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForMaskedLM) - Language modeling Head
2. [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification) - Classification Head
3. [AutoModelForTokenClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForTokenClassification) - Token Classification Head

Feel free to read the [documentation](https://huggingface.co/docs/transformers/model_doc/auto#auto-classes) for a list of other possible heads
"""

# Load in the bert-base-cased model with a classification head (num_labels = number of classes)
classification_head_bert = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=5)

"""### **Training the Model + Classification head**

Let's now train the model + classification head for sentiment analysis. We will use the HuggingFace [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) class. We need to:
1. Tokenize and process the dataset
2. Create a `compute_metrics` function
3. Initialize the `Trainer` and specify the `TrainingArguments`

### **Step 1: Tokenize and Process the dataset** (5 points)

For this section you must implement the `tokenize_function`. This function should run the BERT tokenizer on every example in the dataset and add the output fields `input_ids`, `attention_mask`, and `token_type_ids` as new columns in the dataset.

This function should also also pad all sequences of less than 512 tokens to be exactly 512 tokens using the special `[PAD]` token and truncate sequences longer than 512 to be exactly 512. This is to help GPU parallelization and can be done by specifying certain arguments to the tokenizer (check the [documentation](https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__)!)
"""

# Step 1: Tokenize and Process the dataset
dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(row):
    '''
    Pseudocode:
        1. Use the BERT tokenizer to tokenize the text in the "text" field of the input row.
        2. Ensure that the tokenized output is padded to the maximum length.
        3. Ensure that the tokenized output is truncated to the maximum length if it exceeds it.
        4. Return the tokenized output.

    Input:
        row: A dictionary representing a single row of the dataset.
              It contains at least the key "text" which holds a string of text.

    Returns:
        A dictionary with the tokenized output including keys "input_ids", "attention_mask", and "token_type_ids".
    '''
    # Tokenize the dataset (Remember to pad and truncate to max length)
    return tokenizer(
        row["text"],
        padding="max_length",                   # Pad to max length
        truncation=True,                        # Truncate if needed
        max_length=tokenizer.model_max_length,  # Use model's max token length
        # Return as plain dict for HuggingFace Datasets
        return_tensors=None
    )


# Randomly select 1000 examples from the train and test data and tokenize - do not change!
# (Note: We are subsampling our data just so that training doesn't take too long)
train_data = dataset["train"].shuffle(seed=42).select(
    range(1000)).map(tokenize_function)
eval_data = dataset["test"].shuffle(seed=42).select(
    range(1000)).map(tokenize_function)

"""### **Step 2: Define a `compute_metrics` function** (10 points)

For this function you'll be calculating accuracy between the predictions and true labels. The `probabilities` variable is a 2D numpy array of size ($n$ x $5$) containing probabilities for the 5 class labels for each of the $n$ examples in the input dataset.

You must write code to select the index with the highest probability for each row in the `probabilities` array then calculate the accuracy of the model with respect to the ground truth label. (HINT: the [np.argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html) function is really nice for this)
"""

# Step 2: Define a compute_metrics function


def compute_metrics(eval_pred):
    '''
    Pseudocode:
        1. Extract the predicted probabilities and the true labels from the evaluation predictions.
        2. Compute the predicted labels by taking the argmax of the probabilities along the last axis.
        3. Calculate the accuracy by comparing the predicted labels with the true labels.
        4. Return a dictionary containing the accuracy.

    Input:
        eval_pred: A tuple (probabilities, labels)
                    probabilities: A 2D numpy array of shape (num_examples, num_classes) representing the predicted probabilities for each class.
                    labels: A 1D numpy array of shape (num_examples,) representing the true labels.

    Returns:
        A dictionary with the key "accuracy" and the value being the calculated accuracy.
    '''
    # Get the true labels and predicted probabilities
    probabilities, labels = eval_pred

    # Select the index (class) with the highest probability for each row
    preds = np.argmax(probabilities, axis=1)

    # Compute accuracy as the proportion of correct predictions
    accuracy = np.mean(preds == labels)

    return {'accuracy': accuracy}


"""### **Step 3: Initialize the `Trainer` and Specify `TrainingArguments`**

Here we specify various arguments to the trainer. We choose to evaluate and log every `epoch` (i.e. every pass through the full dataset), we specify the output directory for checkpoints, the learning rate (`5e-05`), the number of epochs (`3`), and the size of batches used for stochastic gradient descent (`8`). In addition, we specify full_determinism for the sake of grading consistency.

In the Trainer we specify our model as the BERT model we loaded earlier, we pass in our argument and filtered datasets, and finally we pass in our `compute_metrics` function.
"""

# Step 3: Specify the TrainingArguments and Initialize the Trainer (Do not change!)
training_args = TrainingArguments(
    eval_strategy="epoch",
    logging_strategy="epoch",
    output_dir="yelp-training",
    learning_rate=5e-05,
    num_train_epochs=3.0,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    full_determinism=True
)

trainer = Trainer(
    model=classification_head_bert,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    compute_metrics=compute_metrics,
)

"""### **Time to Train!**

Once all the above cells have finished running it is time to train the model. On GPU this should only take about 5 minutes. You can check if you're using the GPU by going to `Runtime` -> `Change Runtime Type` and making sure that `Hardware Accelerator` is set to `GPU`
"""

# Train the model!
trainer.train()
trainer.save_model('./yelpBERT')

"""## Sanity-Checking our trained model

In order to run our trained model we need to load it into a `text-classification` pipeline. A quick sanity check should show us that our model outputs sensible results
"""

yelpBERT = pipeline('text-classification', model='./yelpBERT',
                    tokenizer=tokenizer, device=device_id)

# Output sentiment for reviews (LABEL_4 = 5 stars, LABEL_0 = 1 star)
yelpBERT(["This place was amazing!",
          "This place was good",
          "This place was fine, there were good and bad parts.",
          "This place was pretty bad",
          "This place was awful"])

"""## Evaluating the model

In order to get the official evaluation results we will be using the [Evaluator](https://huggingface.co/docs/evaluate/v0.4.0/en/package_reference/evaluator_classes#evaluator) pipeline for a standardized evaluation environment. Please run the following code to get the accuracy of your model on the Yelp dataset. We will be grading your model's accuracy. Your model should be at least higher than 55% to recieve full credit for this section.
"""

task_evaluator = evaluator("text-classification")

eval_results = task_evaluator.compute(
    model_or_pipeline=yelpBERT,
    data=eval_data,
    metric=evaluate.load("accuracy"),
    label_mapping={"LABEL_0": 0, "LABEL_1": 1,
                   "LABEL_2": 2, "LABEL_3": 3, "LABEL_4": 4}
)

print(eval_results)

"""# Section 3: Training BERT for Natural Language Inference (25 points)

In this section you will train BERT to perform a new task -- Natural Language Inference (NLI). NLI is the task of taking two pieces of text (a "premise" and "hypothesis") and determining whether or not the premise is entailed from the hypothesis.


> Natural Language Inference is a task of determining whether the given “hypothesis” and “premise” logically follow (entailment) or unfollow (contradiction) or are undetermined (neutral) to each other. ~ [Oleh Loksyn](https://towardsdatascience.com/natural-language-inference-an-overview-57c0eecf6517)

This can be formulated as a sequence classification task where, given both sequences, we predict one of three labels

(0 = contradiction, 1 = neutral, 2 = entailment)

### **Applying BERT to multiple sequences**

In order to encode multiple sequences of non-contiguous text with BERT we use the `[SEP]` token. This token is a special token (similar to `[MASK]`) that indicates to the model that the sequences on either side of the token are distinct. With the `[SEP]` token we're able to concatenate the premise and hypothesis together into one sequence allowing us to use a standard classification head.

### **Loading BERT**
Your first task is to load in BERT with a classification head. This should be identical to the loading code from Section 2 save for a different number of labels
"""

# Load in the bert-base-cased model with a classification head (remember the right number of labels!)
classification_head_bert = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased",
    num_labels=3
)

"""### **The MNLI Dataset**
For this task we'll be using the [Multi-Genre NLI](https://huggingface.co/datasets/multi_nli) dataset. This dataset was collected by asking crowd workers to annotate 433,000 sentence pairs from various genres (Fiction, Travel, Telephone, Letters, Government) for their textual entailment information.
"""

mnli = load_dataset("multi_nli")

"""### **Tokenize and Process the Dataset** (15 points)

In this section you must implement the three processing functions `is_labeled`, `concatenate` and `tokenize_function` each worth 5 points. These should be fairly similar to the processing functions we've been dealing with so far.
"""


def is_labeled(row):
    '''

    Input:
        row: A dictionary representing a single row of the dataset.
              It contains at least the key "label" which holds an integer.

    Returns:
        A boolean value indicating whether the "label" is not -1.
    '''
    # Filter out all examples that do not have an entailment label (i.e. label is -1)
    return row["label"] != -1


mnli = mnli.filter(is_labeled)


def concatenate(row):
    '''
    Input:
        row: A dictionary representing a single row of the dataset.
              It contains at least the keys "premise" and "hypothesis", both holding strings.

    Returns:
        The input row dictionary, but with an added key "concat" that contains the concatenated string.

    '''
    # Concatenate the text of the "premise" column to the "hypothesis" column
    # using the [SEP] token. Must be in the order "<premise> [SEP] <hypothesis>"
    # Add the output as a new column "concat" to the dataset
    row["concat"] = row["premise"] + " [SEP] " + row["hypothesis"]
    return row


mnli = mnli.map(concatenate)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(row):
    '''
    Input:
        row: A dictionary representing a single row of the dataset.
              It contains at least the key "concat" which holds a string of text.

    Returns:
        A dictionary with the tokenized output including keys "input_ids", "attention_mask", and "token_type_ids".
    '''
    # Tokenize the concatenated data (make sure to pad and truncate to max length!)
    return tokenizer(
        row["concat"],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length
    )


# Randomly select 1000 examples from train and validation and process them (Do not change!)
train_mnli = mnli["train"].shuffle(seed=42).select(
    range(1000)).map(tokenize_function)
eval_mnli = mnli["validation_matched"].shuffle(
    seed=42).select(range(1000)).map(tokenize_function)

"""### **Define the `compute_metrics` function** (5 points)
For this section you'll define the compute_metrics function. We'll be using accuracy again in this section. Feel free to reference your implementation from the previous section to help you out.
"""


def compute_metrics(eval_pred):
    '''
    Input:
        eval_pred: A tuple (logits, labels)
                    logits: A 2D numpy array of shape (num_examples, num_classes) representing the raw predictions from the model.
                    labels: A 1D numpy array of shape (num_examples,) representing the true labels.

    Returns:
        A dictionary with the key "accuracy" and the value being the calculated accuracy.
    '''
    # Extract the outputs of the model and compute accuracy
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = np.mean(preds == labels)

    return {"accuracy": accuracy}


"""### **Initialize the `TrainingArguments` and `Trainer`**"""

training_args = TrainingArguments(
    eval_strategy="epoch",
    logging_strategy="epoch",
    output_dir="mnli-training",
    learning_rate=5e-05,
    num_train_epochs=3.0,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    full_determinism=True
)

trainer = Trainer(
    model=classification_head_bert,
    args=training_args,
    train_dataset=train_mnli,
    eval_dataset=eval_mnli,
    compute_metrics=compute_metrics,
)

"""### **Train the model**"""

trainer.train()
trainer.save_model('./mnliBERT')

"""### **Sanity Check**

Once again, if trained properly, a quick sanity check should provide us with sensible results
"""

mnliBERT = pipeline('text-classification', model='./mnliBERT',
                    tokenizer=tokenizer, device=device_id)

# Output entailment (LABEL_0 = entailment, LABEL_1 = neutral, LABEL_2 = contradiction)
mnliBERT(["I just graduated college. [SEP] The person is a college graduate.",
          "My name is John. [SEP] The sky is blue.",
          "Thank you so much for your help. [SEP] The person was not helped."])

"""### **Evaluate the Model**

Please run the following code to evaluate your model on the MNLI dataset. Your model should be at least 55% accurate to receive full credit.
"""

task_evaluator = evaluator("text-classification")

eval_results = task_evaluator.compute(
    model_or_pipeline=mnliBERT,
    data=eval_mnli,
    input_column="concat",
    label_column="label",
    metric=evaluate.load("accuracy"),
    label_mapping={"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
)

print(eval_results)

"""# Section 4: Training BERT for Named-Entity Recognition (40 points)

In this final section you will train BERT to perform [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) (NER).

NER is the task of tagging and identifying named entities (e.g. People, Organizations, Locations) in text. For example the sentence "Only France and Britain backed Fischler's proposal." would be tagged as

> Only (France, LOC) and (Britain, LOC) backed (Fischler's, PER) proposal.

NER is typically formulated as a token classification task. Given a sequence of tokens we assign a class to each token. Tokens that are not named entities get given the null tag `O` and tokens that are named entities get one of four classes (`ORG`, `PER`, `LOC`, or `MISC`). Thus the sentence from earlier would be given the labels

>(Only, O) (France, LOC) (and, O) (Britain, LOC) (backed, O) (Fischler's, PER) (proposal, O)

Giving every token in the sequence a label allows us to use a token classification head for this task.

### **Beginning-Inside-Outside (BIO) tags**

One downside to the tagging scheme described is that for two consecutive tokens with the same tag, we can't tell whether or not they are the same entity. For example

> ("25-1", O), ("Barcelona", ORG), ("Real", ORG), ("Madrid", ORG)

We know that [Real Madrid](https://en.wikipedia.org/wiki/Real_Madrid_CF) and [Barcelona](https://en.wikipedia.org/wiki/FC_Barcelona) are distinct entities but our tag system combines them together. Thus we must distinguish between tags that *begin* an entity (`B-ORG`) and those that are *inside* an entity (`I-ORG`). With this new system we get

> ("25-1", O), ("Barcelona", B-ORG), ("Real", B-ORG), ("Madrid", I-ORG)

Which allows us to distinguish between the two consecutive entities.

### **Training BERT + Token Classification Head**

In order to train BERT for NER we need to use a token classification head. Token classification heads are dense linear layers that output a probability distribution over $n$ classes *for each token* in the sequence. This is equivalent to having $L_{context}$ different classification heads, each trained to predict an output at a specific index. The $L_{context}$​ tokens are treated as a batch dimension, with the same transformation matrix applied to each token in our context window. The total size of a token classification head is ($D_{enc}$ x $n$).

### **Load the Model** (5 points)
Your first task is to load BERT with a token classification head. Take care that the `num_labels` and `id2label` parameters are set correctly.

HINT: Refer back to Section 2's discussion of model heads to find mention (and documentation) for the token classification head class.
"""

label_names = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG',
               4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
# Load the model with the right num_labels and set id2label to be label_names
token_classification_head_bert = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(label_names),
    id2label=label_names,
    label2id={v: k for k, v in label_names.items()}
)

attributes = copy.deepcopy(vars(token_classification_head_bert))
attributes['config'] = attributes['config'].__dict__
del attributes['_modules']

"""### **The CoNLL Dataset**

We will use the [CoNLL](https://huggingface.co/datasets/conll2003) dataset as our training data for NER. This dataset consists of 20,000 sentences from Reuters news articles that were manually annotated by participants for their named entities. The tags used are the four classes from earlier (`ORG`, `PER`, `LOC`, or `MISC`) with both `B-` and `I-` variants as well as a null tag `O`.
"""

conll = load_dataset("conll2003")

"""### **Token-tag mismatch**
When processing data for a tagging task, we need to ensure that the tags properly match up to all the correct tokens. BERT's tokenizer is a [Byte-Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) (BPE) tokenizer that tends to split up words into sub-word tokens. For example, take the sentence from earlier
> "Only France and Britain backed Fischler's proposal"

We can see below that the tokenization from CoNLL doesn't match the BERT tokenizer. This creates a problem when processing the tags, as we need to keep the correspondence between tags and tokens. It will be up to you to fix this
"""

print(conll['train'][12]['ner_tags'])
print(tokenizer.convert_ids_to_tokens(tokenizer.encode(
    "Only France and Britain backed Fischler's proposal"))[1:-1])

"""### **Redistributing Tags to Sub-word Tokens** (10 points)

Your job is to implement the `tokenize_and_tag` function. This function should do three things
1. Retokenize the tokens from CoNLL using the BERT tokenizer and add the outputs (`input_ids`, `attention_mask`, and `token_type_ids`) as new columns in the dataset.
2. Redistribute the NER tags with respect to the newly tokenized sequence and add the tags as a new `labels` column to the dataset
3. Pad and truncate both the tokens and tags to be length `tokenizer.model_max_length` (512)

If a token is broken up into multiple sub-tokens then all sub-tokens should be given the same NER tag as the original token.

Some example inputs and outputs are as follows (Reminder to consult the `label_names` dictionary for the mapping from number to NER tag):
```
-- Example #1 --
Inputs:
"tokens": ['Israel', 'approves', 'Arafat', "'s", 'flight', 'to', 'West', 'Bank', '.']
"ner_tags": [5, 0, 1, 0, 0, 0, 5, 6, 0]

Outputs:
"bert_tokens": ['[CLS]', 'Israel', 'approve', '##s', 'Ara', '##fa', '##t', "'", 's', 'flight', 'to', 'West', 'Bank', '.', '[SEP]', '[PAD]', '[PAD]', ...]
"input_ids": [101, 3103, 14942, 1116, 25692, 8057, 1204, 112, 188, 3043, 1106, 1537, 2950, 119, 102, 0, 0, ...]
"labels": [-100, 5, 0, 0, 1, 1, 1, 0, 0, 0, 0, 5, 6, 0, -100, -100, -100, ...]

-- Example #2 --
Inputs:
"tokens": ['66', 'Paul', 'Goydos', ',', 'Billy', 'Mayfair', ',', 'Hidemichi', 'Tanaka', '(', 'Japan', ')']
"ner_tags": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 5, 0]

Outputs:
"bert_tokens": ['[CLS]', '66', 'Paul', 'Go', '##yd', '##os', ',', 'Billy', 'May', '##fair', ',', 'Hi', '##de', '##mic', '##hi', 'Tanaka', '(', 'Japan', ')', '[SEP]', '[PAD]', '[PAD]', ...]
"input_ids": [101, 5046, 1795, 3414, 19429, 2155, 117, 4224, 1318, 19803, 117, 8790, 2007, 7257, 3031, 24128, 113, 1999, 114, 102, 0, 0, ...]
"labels": [-100, 0, 1, 2, 2, 2, 0, 1, 2, 2, 0, 1, 1, 1, 1, 2, 0, 5, 0, -100, -100, -100, ...]
```
**NOTE:** Make sure to use the label `-100` for all special tokens (`[CLS]`, `[SEP]`, `[PAD]`). Using this label will ensure that special tokens will be ignored by PyTorch and will not contribute to the loss calculation.
"""


def tokenize_and_tag(row):
    '''
    Pseudocode:
        1. Initialize empty lists for tokens and tags.
        2. For each word and its corresponding tag in the input row:
            a. Tokenize the word using the BERT tokenizer.
            b. Extend the tokens list with the tokenized word.
            c. Extend the tags list with the tag repeated for each sub-token.
        3. Tokenize the entire sequence of tokens using the BERT tokenizer with padding and truncation.
        4. Add the special label -100 to the start of the labels list and pad it to the maximum length.
        5. Add the labels list to the tokenized samples dictionary.
        6. Return the tokenized samples dictionary.

    Input:
        row: A dictionary representing a single row of the dataset.
              It contains at least the keys "tokens" and "ner_tags", both holding lists of strings and integers respectively.

    Returns:
        A dictionary with the tokenized output including keys "input_ids", "attention_mask", "token_type_ids", and "labels".
    '''
    # Redistribute and tokenize the CoNLL dataset using the BERT tokenizer
    # Initialize lists for tokens and tags
    tokens = []
    tags = []

    # Process each word and its tag
    for word, tag in zip(row["tokens"], row["ner_tags"]):
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)
        tags.extend([tag] * len(word_tokens))

    # Add special tokens [CLS] and [SEP]:
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    tags = [-100] + tags + [-100]  # -100 for special tokens

    # Convert tokens to input IDs:
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Create attention mask:
    attention_mask = [1] * len(input_ids)

    # Pad sequences to max length:
    padding_length = tokenizer.model_max_length - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        tags = tags + [-100] * padding_length  # -100 for padding tokens

    # Truncate if necessary:
    input_ids = input_ids[:tokenizer.model_max_length]
    attention_mask = attention_mask[:tokenizer.model_max_length]
    tags = tags[:tokenizer.model_max_length]

    # Create token type ids:
    token_type_ids = [0] * tokenizer.model_max_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": tags
    }


# Process datasets
train_conll = conll["train"].shuffle(seed=42).select(
    range(1000)).map(tokenize_and_tag)
eval_conll = conll["validation"].shuffle(
    seed=42).select(range(1000)).map(tokenize_and_tag)

"""### **Computing Macro-Averaged F1 Score**

We want to compute the accuracy of our model when identifying named entities. However, most sequences consist mostly of `O` tags. This is bad for our evaluation, as it means that predicting a sequence of all `O`s will result in high accuracy despite not actually accomplishing the task.

To solve this problem we will be using the Macro-Averaged F1 Score as our metric. This is the unweighted average of the F1 scores for each individual token class. This is a common metric to use when given an unbalanced multi-class classification task. Feel free to read [this article](https://stephenallwright.com/micro-vs-macro-f1-score/) for a full explanation of Macro-F1 as it will be useful for your implementation.

You are to implement the `calculate_macro_f1` function. This function should:
1. Calculate the full confusion matrix (TP, TN, FP, FN) for each class
2. Compute the precision, recall, and f1 score for each class
3. Macro-Average together the F1 score for all nine classes to get the final score

**NOTE**: Typically for NER a correct entity prediction is considered to be one that exactly matches all `B-` and `I-` tags. However, to make the Macro-F1 calculation easier we are considering all of our nine different tag classes as fully separate.

**TIP**: You can use the [sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) function from scikitlearn to check your implementation. To help debug we have provided the handy `debug_macro_f1` function for you to use. This function generates random test cases and compares the output of your implementation to that of scikitlearn's. If the two do not match it prints out the failed example. This should be helpful when debugging.
"""


def calculate_macro_f1(preds, labels):
    '''
    Pseudocode:
        1. Map prediction and label IDs to their corresponding tag names.
        2. Compute confusion matrices for each class (TP, TN, FP, FN).
        3. Compute precision, recall, and F1 score for each class.
        4. Macro-average the precision, recall, and F1 scores across all classes.
        5. Return a dictionary containing the macro-averaged precision, recall, and F1 scores.

    Input:
        preds: A list of lists, where each sublist contains predicted label IDs for a sequence.
        labels: A list of lists, where each sublist contains true label IDs for a sequence.

    Returns:
        A dictionary with keys "precision", "recall", and "macro-f1", representing the macro-averaged scores.
    '''
    # Filter out -100 and convert ids to tags
    label_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG',
                 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}

    # Flatten predictions and labels
    flat_preds = [p for pred in preds for p in pred]
    flat_labels = [l for label in labels for l in label]

    # Filter out -100 labels and get corresponding predictions
    preds_filtered = []
    labels_filtered = []
    for p, l in zip(flat_preds, flat_labels):
        if l in label_map:
            preds_filtered.append(p)
            labels_filtered.append(l)

    # Get all unique classes that appear in the data
    classes = sorted(set(labels_filtered + preds_filtered))

    # Initialize dictionaries to store TP, FP, FN for each class
    tp = {cls: 0 for cls in classes}
    fp = {cls: 0 for cls in classes}
    fn = {cls: 0 for cls in classes}

    # Compute confusion matrices for each class
    for pred, label in zip(preds_filtered, labels_filtered):
        if pred == label:
            tp[label] += 1
        else:
            fp[pred] += 1
            fn[label] += 1

    # Compute precision, recall, and f1 for each class
    precisions = []
    recalls = []
    f1_scores = []

    for cls in classes:
        precision = tp[cls] / \
            (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        recall = tp[cls] / (tp[cls] + fn[cls]
                            ) if (tp[cls] + fn[cls]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Macro-Average them together
    macro_precision = sum(precisions) / \
        len(precisions) if len(precisions) > 0 else 0
    macro_recall = sum(recalls) / len(recalls) if len(recalls) > 0 else 0
    macro_f1 = sum(f1_scores) / len(f1_scores) if len(f1_scores) > 0 else 0

    return {
        "precision": macro_precision,
        "recall": macro_recall,
        "macro-f1": macro_f1
    }


def debug_macro_f1(num_tests, len_examples):
    # For each random test case
    for i in range(num_tests):
        # Generate two random arrays for the predictions and labels
        rand_preds = np.random.randint(0, 9, len_examples).tolist()
        rand_labels = np.random.randint(0, 9, len_examples).tolist()

        # Calculate Macro-F1 score with your implementation and scikitlearn
        sklearn_f1 = f1_score(rand_labels, rand_preds, average='macro')
        your_f1 = calculate_macro_f1([rand_preds], [rand_labels])

        # If the two implementations differ, print the example and the scores
        if abs(sklearn_f1 - your_f1['macro-f1']) <= 0.001:
            print(f'preds: {rand_preds}')
            print(f'labels: {rand_labels}')
            print(sklearn_f1, your_f1['macro-f1'])


debug_macro_f1(1, 10)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=2)
    return calculate_macro_f1(preds, labels)


"""### **Training the model**"""

training_args = TrainingArguments(
    eval_strategy="epoch",
    logging_strategy="epoch",
    output_dir="conll-training",
    learning_rate=5e-05,
    num_train_epochs=3.0,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    full_determinism=True
)

trainer = Trainer(
    model=token_classification_head_bert,
    args=training_args,
    train_dataset=train_conll,
    eval_dataset=eval_conll,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model('./nerBERT')

"""### **Sanity Checking the NER Predictions**
In order to check our model we need to load it into a `token-classification` pipeline.

If your model has been trained correctly then you will see in the following example that it correctly predicts the "Pennsylvania" in "University of Pennsylvania" as an `ORG` but "Philadelphia" as a `LOC`.
"""

nerBERT = pipeline('token-classification', model='./nerBERT',
                   tokenizer=tokenizer, device=device_id)

nerBERT(["Chris Callison-Burch is a professor at the University of Pennsylvania in Philadelphia",
         "Joe Biden is the 46th President of the United States"], aggregation_strategy="simple")

"""### **Evaluating NER with SeqEval**

The official library for evaluating a model on the CoNLL task is [SeqEval](https://github.com/chakki-works/seqeval). We'll be downloading it and using it to officially score our model. The SeqEval `overall_f1` is identical to your Macro-F1 implementation with the slight caveat that it uses the traditional NER definition of a correct tag prediction (i.e. one that exactly matches all `B-` and `I-` tags). Thus, don't worry if the official result doesn't match your macro-f1 implementation.

For this task, in order to receive full credit, your implementation must get a Macro-F1 score of at least 0.82.
"""

!pip install seqeval

task_evaluator = evaluator("token-classification")

eval_results = task_evaluator.compute(
    model_or_pipeline=nerBERT,
    data=eval_conll,
    input_column="tokens",
    label_column="ner_tags",
    tokenizer=tokenizer,
    metric=evaluate.load("seqeval")
)

for key in eval_results:
    print(f"{key}: {eval_results[key]}")
