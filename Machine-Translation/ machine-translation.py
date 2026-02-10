# -*- coding: utf-8 -*-
"""
# Text Simplification as Machine Translation

- **Overview**: In this homework assignment, you will learn about text simplification and will fine-tune GPT to translate complex English sentences into simple ones. We'll cover:

  - Exploring text completexity and how to measure it
  - Write simpler versions of sentences manually
  - Use prompting to simplify text with GPT
  - Fine-tune GPT to translate between complex and simple English

- **Grading**: We will use the auto-grading system called `PennGrader`. To complete the homework assignment, you should implement anything marked with `#TODO` and run the cell with `#PennGrader` note.


## Related Readings (Optional)

- [Simple-qe: Better automatic quality estimation for text simplification](https://arxiv.org/abs/2012.12382) Reno Kriz, Marianna Apidianaki, and Chris Callison-Burch, arXiv preprint arXiv:2012.12382, 2020.

- [Complexity-weighted loss and diverse reranking for sentence simplification](https://arxiv.org/abs/1904.02767) Reno Kriz, João Sedoc, Marianna Apidianaki, Carolina Zheng, Gaurav Kumar, Eleni Miltsakaki, and Chris Callison-Burch, arXiv preprint arXiv:1904.02767, 2019.

- [Simplification using paraphrases and context-based lexical substitution](https://aclanthology.org/N18-1019) Reno Kriz, Eleni Miltsakaki, Marianna Apidianaki, and Chris Callison-Burch, Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 207-217, 2018.

- [Optimizing statistical machine translation for text simplification](https://www.aclweb.org/anthology/Q16-1029) Wei Xu, Courtney Napoles, Ellie Pavlick, Quanze Chen, and Chris Callison-Burch, Transactions of the Association for Computational Linguistics 4, pp. 401-415, 2016.

- [Simple PPDB: A paraphrase database for simplification](https://aclanthology.org/P16-2024/) Ellie Pavlick, and Chris Callison-Burch, Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 143-148, 2016.

- [Problems in current text simplification research: New data can help](https://aclanthology.org/Q15-1021/) Wei Xu, Chris Callison-Burch, and Courtney Napoles, Transactions of the Association for Computational Linguistics 3, pp. 283-297, 2015.

"""

# Commented out IPython magic to ensure Python compatibility.
# ## DO NOT CHANGE ANYTHING, JUST RUN
# %%capture
# !pip install penngrader-client

# Commented out IPython magic to ensure Python compatibility.
# %%writefile notebook-config.yaml
#
# grader_api_url: 'https://23whrwph9h.execute-api.us-east-1.amazonaws.com/default/Grader23'
# grader_api_key: 'flfkE736fA6Z8GxMDJe2q8Kfk8UDqjsG3GVqOFOa'



"""# Section 1: Exploring Text Complexity
**Background:** Text simplification is the process of converting complex sentences into simple ones that are still faithful to the originl meaning. It can be used in cases where a complex document needs to be translated into a simpler form for a particular audience, like children, non-native speakers, and people who face reading difficulties. Even computational tasks like machine translation and text summarization can benefit from simplifying text first.

Over the years, research on text simplification has evolved from simple, rule-based approaches like substituting complex words with simpler ones, or splitting long sentences into more easily understandable shorter ones, to more complex methods that leverage large language models to understand complex pieces of information and generate simpler versions.

Here, we will attempt and evaluate different methods and finally use modern techniques like fine-tuning GPT to "translate" complex text into simple text.

First, let's install and import the required libraries.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install openai textstat nltk pandas jsonlines wandb scikit-learn cmudict==1.0.33
#
# import os
# from openai import OpenAI
# import textstat
# import nltk
# from nltk.corpus import wordnet as wn
# from nltk.corpus import stopwords
# import pandas as pd
# from random import sample
# import numpy as np
# import json
# import time
# import seaborn as sns
# import string
# import glob
# from collections import Counter
# from sklearn.metrics.pairwise import cosine_similarity

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

from getpass import getpass
import os

print('Enter OpenAI API key:')
openai_api_key = getpass()

os.environ['OPENAI_API_KEY']=openai_api_key
client = OpenAI()

"""### What makes a sentence simple?

**Problem 1.1**

Before we use computers to perform this task, let's first try to do it ourselves to understand it better. Pick five complex sentences you may have recently read (or find some on the Internet), and write your own simplified versions of them.

Example:
"""

original_sentence = "The diversification in occupational roles has become a crucial attribute in today's rapidly evolving job market."

simplified_sentence = "Having a variety of job roles is important for today's fast-changing job market."

example_sentence_pair = (original_sentence, simplified_sentence)

##################################################

pair_1 = ("Although the experiment yielded significant results, the researchers cautioned that further studies would be necessary to confirm their findings.", "The experiment showed results, but more studies are needed.")
pair_2 = ("Because the algorithm was trained on biased data, its predictions consistently favored one demographic over another.", "The algorithm gave biased results because of the data.")
pair_3 = ("While most participants reported an increase in satisfaction, a subset expressed concerns about the interventions long term effectiveness.", "Most people were satisfied, but some had concerns.")
pair_4 = ("Even though the new policy was designed to reduce emissions, it inadvertently increased energy costs for small businesses.", "The policy cut emissions but raised energy costs.")
pair_5 = ("Since the model assumes normality in the error terms, its accuracy deteriorates when applied to data with heavy tailed distributions.", "The model works poorly with unusual data.")

##################################################

pairs = [pair_1, pair_2, pair_3, pair_4, pair_5]

"""**Problem 1.2**

After completing the above task, list five general changes (not specifically to the pairs above) that you might make to a complex sentence to transform it into a simpler one. Your answers should be included in your separate "***writeup.pdf***" file and will be manually graded (5 points).
"""

# TODO in writeup.pdf file

"""### Measuring complexity

While it may be clear to you that the transformed sentences are simpler than the original ones, we need a way to quanitify this so that we can automate the process.

One way to think about this is to measure the 'readability' of a text. The most well-known readability metric are the Flesch–Kincaid tests. The test have two types: Flesch Reading-Ease and Flesch–Kincaid Grade Level.

The Flesch Reading-Ease test is scored on a scale of 1-100, with higher scores indicating that the text is easier to read. The score is calculated by the following formula:

![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/bd4916e193d2f96fa3b74ee258aaa6fe242e110e)


The table below shows a breakdown of the different levels and how they translate to school levels.


| Score | School level (US) | Notes                                                           |
|-------|------------------|-----------------------------------------------------------------|
| 100.00–90.00 | 5th grade  | Very easy to read. Easily understood by an average 11-year-old student. |
| 90.0–80.0 | 6th grade | Easy to read. Conversational English for consumers.                   |
| 80.0–70.0 | 7th grade | Fairly easy to read.                                                |
| 70.0–60.0 | 8th & 9th grade | Plain English. Easily understood by 13- to 15-year-old students.  |
| 60.0–50.0 | 10th to 12th grade | Fairly difficult to read.                                      |
| 50.0–30.0 | College | Difficult to read.                                                 |
| 30.0–10.0 | College graduate | Very difficult to read. Best understood by university graduates. |
| 10.0–0.0 | Professional | Extremely difficult to read. Best understood by university graduates. |

\
The Flesch–Kincaid Grade Level, on the other hand, calculates the approximate U.S. grade level that corresponds to the complexity of the given text. It is therefore inversely related to the Flesch Reading-Ease, since lower school grade levels correspond to text that is easier to read, and provides a more interpretable understanding of the complexity of a text. It is calculated by the following formula:

![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/8e68f5fc959d052d1123b85758065afecc4150c3)

Notice how, in both tests, having more words per sentence, and having more syllables per word, both increase the complexity of text.

**Problem 1.3**

Using the appropriate functions in the `textstat` module, evaluate the complexity of your sentences and confirm that the simplified ones score accordingly on both metrics. You can find the documentation for it [here](https://github.com/textstat/textstat). Look for functions that measure the **Flesch Reading-Ease** and **Flesch–Kincaid Grade Level**.

Your solution should be formatted as follows:

```
reading_ease = [(pair1_complex_sentence_score, pair1_simple_sentence_score), ...]
grade_level = [(pair1_complex_sentence_score, pair1_simple_sentence_score), ...]
```
"""

##################################################

pairs = [
    ("Although the experiment yielded significant results, the researchers cautioned that further studies would be necessary to confirm their findings.", "The experiment showed results, but more studies are needed."),
    ("Because the algorithm was trained on biased data, its predictions consistently favored one demographic over another.", "The algorithm gave biased results because of the data."),
    ("While most participants reported an increase in satisfaction, a subset expressed concerns about the interventions long term effectiveness.", "Most people were satisfied, but some had concerns."),
    ("Even though the new policy was designed to reduce emissions, it inadvertently increased energy costs for small businesses.", "The policy cut emissions but raised energy costs."),
    ("Since the model assumes normality in the error terms, its accuracy deteriorates when applied to data with heavy tailed distributions.", "The model works poorly with unusual data."),
]

reading_ease = []
grade_level = []

for complex_sentence, simple_sentence in pairs:
    reading_ease.append((
        textstat.flesch_reading_ease(complex_sentence),
        textstat.flesch_reading_ease(simple_sentence)
    ))
    grade_level.append((
        textstat.flesch_kincaid_grade(complex_sentence),
        textstat.flesch_kincaid_grade(simple_sentence)
    ))

print("reading_ease =", reading_ease)
print("grade_level =", grade_level)

##################################################

"""## Evaluating simplification quality

While measuring the change in Flesch-Kincaid Grade Level is one measure of how good a simplification is, another way to measure its quality is to see how faithful it is to the original sentence. While we want the simplification to be concise, we also want it to retain the meaning of the original.

### Measuring semantic similarity

In order to measure the similarity of two sentences, we can compute the difference between their vector embeddings. We studied word embeddings earlier in the class, which let us represent words as vectors we can use to compute how similar two words are to each other. The same can be done for sentences, and even documents, using large language models (LLMs). Let's see how this works.

**Problem 1.4**

Define a function that gets the embeddings for a given piece of text from the OpenAI API. Use the `text-embedding-ada-002` model, which accepts a list of strings as input and returns a list embeddings in response. You can look at the API documentation [here](https://platform.openai.com/docs/guides/embeddings/use-cases).
"""

# Create a client using the environment's API key
client = OpenAI()

def get_embedding(text, model="text-embedding-ada-002"):
    embeddings = None
    response = client.embeddings.create(input=[text], model=model)
    embeddings = response.data[0].embedding

    return embeddings

# Generate embeddings for all sentence pairs
embeddings = [(get_embedding(i), get_embedding(j)) for (i, j) in pairs]

"""**Problem 1.5**

Now use the cosine similarity function we imported from `scikit-learn` to compare the embeddings of all pairs of sentences. You can read its documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html).

Completely identical sentences should have a similarity of 1.0, while sentences roughly on the same topic might have a cosine similarity of 0.9. Ideally, we want our simplified sentences to be as close to their original, complex versions as possible.
"""

# Compute cosine similarities
sim = [
    cosine_similarity(
        np.array(complex_emb).reshape(1, -1),
        np.array(simple_emb).reshape(1, -1)
    )[0][0]
    for (complex_emb, simple_emb) in embeddings
]

# Output result
print("cosine_similarities =", sim)

"""# Section 2: Rule-based Simplification

Given what we have learned from the above exercise, let's now try to write a simple rule-based system for simplifying sentences.

One of the ways to simplify a sentence is to use simpler words in the place of complex ones. Use the wordnet lexical database to replace complex words in a sentence with their synonyms that are simpler. Some heuristics you might want to consider are the word length, number of syllables, and frequency in the English corpus.

**Problem 2.1**

Using WordNet, list all the synonyms of the following words:
language, research, complex, simple, model. Your answer should be a list of length five, where each element is an alphabetically sorted list containing all the synonyms of the respective word.


```
[
  ['word1_synonym1', 'word1_synonym2', ... ],
  ['word2_synonym1', 'word2_synonym2', ... ],
  ...
]
```


You can see the NLTK WordNet documentation [here](https://www.nltk.org/howto/wordnet.html).
"""

words = ['language', 'research', 'complex', 'simple', 'model']

def get_synonyms(word):
    synonyms = set()
    synonyms.add(word.lower())

    # Get all synsets for the word
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            synonym = lemma.name().lower()
            synonyms.add(synonym)

    # Return sorted list
    return sorted(list(synonyms))

# Get synonyms for all words
synonyms = [get_synonyms(word) for word in words]

"""Once we have a list of possible replacements for a word, we need a way to pick the simplest one. One heuristic for this is how common a word is in the English language -- common words tend to be simpler than rare ones. To get the frequency of a word, you can use the Google n-grams corpus from your text classification homework.

Let's download and unzip the Google n-gram dataset first.
"""

!curl -L -o ngram_counts.txt.gz http://www.cis.upenn.edu/~cis5300/18sp/data/ngram_counts.txt.gz

!gzip -d ngram_counts.txt.gz

"""

```
`# This is formatted as code`
```

**Problem 2.2**

Write a function to extract the frequency of a given word from the n-gram dataset. It might help to store the n-grams in a data structure that allows efficient retrieval. You should return the frequency as an integer. If a word does not appear in the n-gram dataset, its frequency should be 0."""

from collections import defaultdict
import gzip

ngram_counts = defaultdict(int)

# Read the ngram file:
with open('ngram_counts.txt', 'r', encoding='utf-8') as f:
    for line in f:
        word, count = line.strip().split('\t')
        ngram_counts[word.lower()] = int(count)

def get_freq(word):
    """Return the frequency of a word from the n-gram dataset"""
    return ngram_counts.get(word.lower(), 0)

# Test words from Problem 2.1
words = ['language', 'research', 'complex', 'simple', 'model']
freqs = [get_freq(word) for word in words]

"""**Problem 2.3**

Now put both of these pieces together to write a function that, given a sentence, replaces each word with its simplest i.e. most common synonym. You will find all possible synonyms for each word in the sentence and replace them with the most frequently occuring one.

Example input: Science is a subject that requires meticulous attention to detail and rigorous experimentation

Example output: Science be a case that take meticulous care to point and strict experiment
"""

def synonym_simplify(text):
    words = text.split()
    simplified = []

    for word in words:
        # Get base word without any trailing punctuation
        base = word
        punct = ''
        # Separate word from trailing punctuation
        while len(base) > 0 and not base[-1].isalpha():
            punct = base[-1] + punct
            base = base[:-1]

        if not base:  # If the word was all punctuation
            simplified.append(word)
            continue

        # Get synonyms and find most frequent one
        synonyms = get_synonyms(base)
        if not synonyms:  # If no synonyms found
            simplified.append(word)
            continue

        most_common = max(synonyms, key=get_freq)

        # Preserve original capitalization
        if base[0].isupper():
            most_common = most_common.capitalize()

        # Recombine with punctuation
        simplified_word = most_common + punct
        simplified.append(simplified_word)

    return ' '.join(simplified)

simplified = synonym_simplify("The diversification in occupational roles has become a crucial attribute in today's rapidly evolving job market")
print(simplified)

"""# Section 3: Simplification via Prompting

Since you have seen that simplification is a hard task to do using hard-coded rules, let's try using large language models to simplify text for us. We will use GPT-4o-mini for this task.

## Zero-shot

In zero-shot prompting, we simply give the model a piece of input, called a prompt, which may contain a question or instruction, and ask the model to process it and return an output.

**Problem 3.1**

Write a prompt that asks GPT to simplify a given sentence. The input (complex) sentence will be appended to the end of the prompt. In response, GPT should return the resulting (simple) sentence.
"""

def simplify_zeroshot(prompt, sentence):

  messages = [
        {"role": "user", "content": prompt + sentence},  # appending the sentence to the prompt
  ]

  response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=messages,
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      # stop=["\n"]
  )

  # We recommend putting a short wait after each call,
  # since the rate limit for the platform is 60 requests/min.
  # (This increases to 3000 requests/min after you've been using the platform for 2 days).
  time.sleep(1)

  # the response from OpenAI's API is a JSON object that contains
  # the completion to your prompt plus some other information. Here's how to access
  # just the text of the completion.
  return response.choices[0].message.content.strip()

##################################################
prompt = "Please simplify the following sentence by using simpler vocabulary and clearer phrasing, while preserving its original meaning:\n"
##################################################

simplify_zeroshot(prompt, original_sentence)

"""Here is some information about the different arguments that we gave to the `client.completions.create` call:
 * `model` – Model ID used to generate the response, like gpt-4o or gpt-4o-mini, which we use here.
 * `messages` - A list of messages comprising the conversation so far.
 * `temperature` - controls how much of the probability distribution the model will use when it is generating each token. 1.0 means that it samples from the complete probability distrubiton, 0.7 means that it drops the bottom 30% of the least likely tokens when it is sampling. 0.0 means that it will perform deterministically and always output the single most probable token for each context.
 * `top_p` - is an alternative way of controling the sampling.
 * `frequency_penalty` and `presence_penalty` are two ways of reduing the model from repeating the same words in one output.  You can set these to be >0 if you're seeing a lot of repetition in your output.
 * `max_tokens` is the maximum length in tokens that will be output by calling the function.  A token is a subword unit.  There are roughly 2 or 3 tokens per word on average.
 * `stop` is a list of stop sequences.  The model will stop generating output once it generates one of these strings, even if it hasn't reached the max token length. By default this is set to a special token `<|endoftext|>`.

You can read more about [the Chat Completions API call in the documentation](https://platform.openai.com/docs/api-reference/chat).
"""

"""Try this prompt on all five of your complex sentences and calculate the readibility metrics on both the simplified forms. Confirm that the simplified sentences are more readable than the originals, and compare the level of simplification performed by GPT against your manual attempt."""

simple_sentences_gpt = [simplify_zeroshot(prompt, i) for i, j in pairs]
simple_sentences_gpt

"""**Problem 3.2**

You can also vary how much you want GPT to simplify a sentence. Write two prompts, one which simplifies the input sentence a little, and another which simplifies the input sentence comparatively more.

You can try doing this at different levels and check whether the Flesch-Kincaid scores change accordingly.

"""

##################################################
# Prompt for slight simplification
prompt_less_simple = "Simplify the following sentence just a little by using slightly simpler vocabulary and phrasing, but keep the structure and details mostly intact:\n"

# Prompt for strong simplification
prompt_more_simple = "Greatly simplify the following sentence using very basic vocabulary and shorter sentence structure, while still preserving the core meaning:\n"

##################################################

print(simplify_zeroshot(prompt_less_simple, original_sentence))
print(simplify_zeroshot(prompt_more_simple, original_sentence))

"""You can also give GPT specific instructions on how it should simplify a sentence. Try writing prompts explicitly instructing GPT to make the specific changes you listed above in Problem 1.2 and see whether that works."""

##################################################
# Prompt 1: Simplify by reducing sentence length and breaking into shorter sentences
prompt_change_1 = "Break this long sentence into 2-3 shorter sentences while preserving the meaning: "

# Prompt 2: Replace complex words with simpler synonyms
prompt_change_2 = "Replace all complex words with simpler synonyms while keeping the same meaning: "

# Prompt 3: Remove unnecessary clauses and modifiers
prompt_change_3 = "Remove unnecessary clauses and modifiers to make this sentence more concise: "

# Prompt 4: Change passive voice to active voice
prompt_change_4 = "Convert this sentence from passive to active voice while simplifying: "

# Prompt 5: Reduce abstract concepts to concrete terms
prompt_change_5 = "Replace abstract concepts with more concrete, everyday language: "

##################################################

##################################################

print("Simplification by breaking into shorter sentences:")
print(simplify_zeroshot(prompt_change_1, original_sentence))

print("\nSimplification using simpler synonyms:")
print(simplify_zeroshot(prompt_change_2, original_sentence))

print("\nSimplification by removing unnecessary clauses:")
print(simplify_zeroshot(prompt_change_3, original_sentence))

print("\nSimplification using active voice:")
print(simplify_zeroshot(prompt_change_4, original_sentence))

print("\nSimplification using concrete terms:")
print(simplify_zeroshot(prompt_change_5, original_sentence))

"""## Few-shot

One way to make GPT do a better job at a task is to give it a few examples describing what we want instead of just giving it an instruction. We can do this by providing a few pairs of complex and simple sentences (in that order, and separated by a hypen) in the prompt, and GPT will learn the mapping between them. We can then give it the target complex sentence as part the prompt (followed by a hyphen), and GPT will complete it by responding with its simplified version.

```
'''
complex_sentence_1 - simple_sentence_1
complex_sentence_2 - simple_sentence_2
...
complex_sentence_5 -
'''
```

**Problem 3.3**

Write a prompt in the above format (minus the last sentence) that uses the first four of your sentences as few-shot examples.
"""

def simplify_fewshot(prompt, sentence):

  prompt = f"{prompt.strip()}\n{sentence} - "  # appending the sentence to the prompt

  messages = [
        {"role": "user", "content": prompt},
  ]

  response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=messages,
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["\n"]
  )
  time.sleep(1)

  return response.choices[0].message.content.strip()


##################################################

prompt = f"""
{pairs[0][0]} - {pairs[0][1]}
{pairs[1][0]} - {pairs[1][1]}
{pairs[2][0]} - {pairs[2][1]}
{pairs[3][0]} - {pairs[3][1]}
"""

##################################################

simplify_fewshot(prompt, pairs[-1][0])

"""# Section 4: Fine-tuning

In addition to zero-shot and few-shot learning, another way of getting large language models to do your tasks is via a process called "fine tuning".  In fine-tuning the model updates its parameters so that it performs well on many training examples.  The training examples are in the form of original complex sentences paired with their gold standard simplified forms.

Large language models are pre-trained to perform well on general tasks like text completion but not on the specific task that you might be interested in.  The models can be fine tuned to perform you task, starting with the model parameters that are good for the general setting, and then updating them to be good for your task.

We'll walk through how to fine-tune GPT for text simplification.

### Data

We'll be using the Newsela dataset (Xu et. al., 2015). This dataset contains sentences from news articles that have been simplified at 4 different grade levels for different audiences.
"""

!gdown 10HX4ekdQdDyZyTldtj0qfly437B9cL5C

!unzip newsela-auto.zip

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# sentences_df = pd.read_csv('newsela-auto/newsela-auto/all_data/aligned-sentence-pairs-all.tsv', sep='\t', names=['id_simple', 'completion', 'id_complex', 'prompt'], on_bad_lines='warn')
# sentences_df = sentences_df[sentences_df['prompt'].str.len() > 50]

sentences_df

"""The sentences are provided at five levels of simplicity, numbered [0-4], where 0 denotes the original, complex sentence, and 4 denotes its simplest form.

As you see above, each line contains the id of the sentence, which also contains this simplicity marker.

In `brain-gender.en-1-4-0`, its simplicity level is 1.

In `bbking-obit.en-4-21-0`, its simplicity level is 4.

### Select a subset of the data for fine-tuning

**Problem 4.1**

Keep only the rows with the most complex prompts (i.e. their simplicity level is 0) simplest completions (i.e. their simplicity level is 4). Then randomly sample 100 sentences from the dataframe (use the random_state value given above).

Your solution should be a dataframe with 100 rows and the same 4 columns as above.
"""

random_state = 42
num_sample = 100

import pandas as pd
import re

# Assume sentences_df is already loaded and has these columns:
# ['id_simple', 'completion', 'id_complex', 'prompt']

# Define a function to extract simplicity level from the ID pattern (e.g. bbking-obit.en-4-20-0)
def extract_simplicity_level(id_str):
    match = re.search(r'en-(\d)-', id_str)
    return int(match.group(1)) if match else None

# Extract levels
sentences_df["prompt_simplicity"] = sentences_df["id_complex"].apply(extract_simplicity_level)
sentences_df["completion_simplicity"] = sentences_df["id_simple"].apply(extract_simplicity_level)

# Filter for prompt level 0 and completion level 4
filtered_df = sentences_df[
    (sentences_df["prompt_simplicity"] == 0) &
    (sentences_df["completion_simplicity"] == 4)
]

# Random sample of 100 rows with the original 4 columns
sentences_df_sample = filtered_df.sample(n=100, random_state=42)[['id_simple', 'completion', 'id_complex', 'prompt']].reset_index(drop=True)

# Show the result
# import ace_tools as tools; tools.display_dataframe_to_user(name="Problem 4.1 Sample", dataframe=final_sample)

sentences_df_sample

"""## Format data for fine-tuning

Below, we'll format data to fine-tune GPT.  The OpenAI API documentation has a [guide to fine-tuning models](https://platform.openai.com/docs/guides/fine-tuning) that you should read.   The basic format of fine-tuning data is a JSONL file (one JSON object per line) with each example in the dataset should be a conversation in the same format as the Chat Completions API, specifically a list of messages where each message has a role and content.

```
{"messages": [{"role": "user", "content": "<prompt text>"}, {"role": "assistant", "content": "<ideal generated text>"}]}
{"messages": [{"role": "user", "content": "<prompt text>"}, {"role": "assistant", "content": "<ideal generated text>"}]}
{"messages": [{"role": "user", "content": "<prompt text>"}, {"role": "assistant", "content": "<ideal generated text>"}]}
...
```

Where our complex sentences form the `<prompt_text>` and the simplified sentences are the `<ideal generated text>`

Do NOT use `system` messages when constucting the messages for your document, this is for the autograder

**Problem 4.2**

Format the dataframe you created above in the JSON format and write it in a file with the name defined below.
"""

fine_tuning_sentences_filename = 'newsela_sentences_finetuning_data.jsonl'

import json

# Assume sentences_df_sample already exists with these columns:
# ['id_simple', 'completion', 'id_complex', 'prompt']

# Step 1: Format each row into Chat Completions API structure
formatted_data = []
for _, row in sentences_df_sample.iterrows():
    entry = {
        "messages": [
            {"role": "user", "content": row["prompt"].strip()},
            {"role": "assistant", "content": row["completion"].strip()}
        ]
    }
    formatted_data.append(entry)

# Step 2: Write to JSONL file (one object per line)
with open(fine_tuning_sentences_filename, "w") as f:
    for item in formatted_data:
        json.dump(item, f)
        f.write("\n")

# Optional: Print confirmation
print(f"Saved {len(formatted_data)} examples to {fine_tuning_sentences_filename}")

with open(fine_tuning_sentences_filename) as f:
  newsela_json = f.read().strip().split('\n')

"""You can verify that the file looks okay by printing out the first ten lines:"""

!head {fine_tuning_sentences_filename}

"""You can also count the total number of lines, words, and characters in the file:"""

!wc {fine_tuning_sentences_filename}

"""## Run the fine-tuning API

Next, we'll make the fine tuning call via the python library. There are 2 sizes of GPT-4 models. They go in alphabetical order from largest to smallest:

*   `gpt-4o-2024-08-06`
*   `gpt-4o-mini-2024-07-18`

As the model sizes increase, so does their quality and their cost. `gpt-4o` is the highest quality and highest cost model. We recommend starting by fine-tuning smaller models to debug your code first so that you don't rack up costs. Once you're sure that your code is working as expected then you can fine-tune a `gpt-4o` model, although `gpt-4o-mini` is perfectly capable and recommended based on the price.

The size of the dataset we've created has been chosen so as to not be too expensive for you to fine-tune your models on and should only cost a few dollars. We encourage experimenting with the simpler, cheaper models before using the more capable and expensive ones.

**Problem 4.3**

Run the fine-tuning API to create your fine-tuned model. You can read the documentation [here](https://platform.openai.com/docs/guides/fine-tuning/create-a-fine-tuned-model).

You may validate your training file, if you would like, using the script in this [link](https://cookbook.openai.com/examples/chat_finetuning_data_prep).

Although, this is NOT required.

Once you have the dataset created, the file needs to be uploaded using the Files API in order to be used with a fine-tuning jobs:
"""

##################################################

fine_tuning_file_id = None

from openai import OpenAI

client = OpenAI()

# Upload the training file
file_response = client.files.create(
    file=open(fine_tuning_sentences_filename, "rb"),
    purpose="fine-tune"
)

fine_tuning_file_id = file_response.id
print("Uploaded file ID:", fine_tuning_file_id)

##################################################

"""After ensuring you have uploaded the file, the next step is to create a fine-tuning job."""

##################################################

fine_tuning_job_id = None
fine_tune_response = client.fine_tuning.jobs.create(
    training_file=fine_tuning_file_id,
    model="gpt-3.5-turbo"
)

fine_tuning_job_id = fine_tune_response.id
print("Fine-tuning job ID:", fine_tuning_job_id)

##################################################

"""You may monitor the status of your fine-tuning job using the below code:"""

response = client.fine_tuning.jobs.list_events(fine_tuning_job_id)

events = response.data
events.reverse()

for event in events:
    print(event.message)

"""Once your fine-tune is completed, you should see a message in the output of the above code saying
`The job has successfully completed`.
"""

# you may also check the status of your fine-tuning job in the UI using this link
print(f"https://platform.openai.com/finetune/{fine_tuning_job_id}")

"""You can retrieve your model ID, by running this command"""

# prints your fine-tuned model ID
client.fine_tuning.jobs.retrieve(fine_tuning_job_id).fine_tuned_model

"""The above output will contain the ID of your model, which you should should copy down for use later. It will look something like this:

```
ft:gpt-4o-mini-2024-07-18:university-of-pennsylvania::BBYu06Qj
```

If you forget to write it down, you can list your fine-tuned runs and models with the following command:
"""

client.fine_tuning.jobs.list(limit=10)

fine_tuned_model = 'ft:gpt-3.5-turbo-0125:upenn::C0MWVj3z' # replace with your REAL ft-model ID from above

"""## Test your fine-tuned model

**Problem 4.4**

Write a function that uses your fine-tuned model to generate simplified sentences. The function will look the same as the zeroshot example above, except that you will provide a prompt but instead the specific model ID of your fine-tuned model to use instead of OpenAI's default models.
"""

# # TODO

def generate_simple_sentence(complex_sentence, finetuned_model):

  response = None

  ##################################################

  messages = [
      {"role": "user", "content": complex_sentence}
  ]

  response = client.chat.completions.create(
      model=finetuned_model,
      messages=messages,
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
  )

  time.sleep(1)

  ##################################################

  return response.choices[0].message.content.strip()

finetuned_model = "ft:gpt-3.5-turbo-0125:upenn::C0MWVj3z"  # remember to replace this with your own model ID

fine_tune_test = generate_simple_sentence(original_sentence, finetuned_model)

"""# Section 5: Instruction Fine-tuning

In this section, we'll try simplifying whole documents instead of sentences. We'll also instruct GPT to simplify a given document to a particular degree.

### Data

We'll again be using the Newsela dataset (Xu et. al., 2015), but this time a different versions which contains the whole documents instead of individual sentences. Let's start by downloading the data.
"""


# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !unzip newsela_share_2020.zip

article_list = pd.read_csv('newsela_share_2020/documents/articles_metadata.csv')

article_list

"""If we look at the file `articles_metadata.csv` in the documents folder, we see that it has a list of the articles and their title, language, grade level, and filename. We can use this spreadsheet to select a subset of articles to finetune our model on.

Notice how there are 5 versions of each article, where the version number [0-5] corresponds to the grade level [12, 8, 6, 5, 3], with higher version numbers representing increasingly simple versions of the article.

## Analyzing readibility, text length, and vocabulary

Let's look at how text readibility correlates with the length of a document and its sentences. We'll also take a look at the top words for simple and complex documents. You won't have to implement anything in here, this is just to show you how complex and simple documents differ along different dimensions.
"""

article_path = 'newsela_share_2020/documents/articles/'
files_complex = glob.glob(article_path+'*.en.0.txt')
files_simple = glob.glob(article_path+'*.en.4.txt')

articles = []

for fname in files_complex+files_simple:
  with open(fname) as f:
    if fname.endswith('.en.0.txt'):
      articles.append([f.read().strip(), 'complex', fname.split('/')[-1]])
    else:
      articles.append([f.read().strip(), 'simple', fname.split('/')[-1]])

articles_df = pd.DataFrame(articles, columns=['text', 'type', 'filename'])

articles_df['readibility'] = articles_df['text'].apply(textstat.flesch_reading_ease)
articles_df['length'] = articles_df['text'].apply(lambda x: len(x.split()))
articles_df['sent_length'] = articles_df['text'].apply(lambda x: np.mean([len(i.split()) for i in x.split('.')]))
articles_df = articles_df.sort_values(by='length')[:-1]
articles_df

"""### Readibilty vs. document length

Note how long documents tend to be less readible than short ones because they may contain more detail.
"""

sns.regplot(data=articles_df, x='readibility', y='length')

"""### Readibility vs. average sentence length
Similarly, longer sentences also tend to be less readible than shorter ones because they can be hard to parse for a reader.
"""

sns.regplot(data=articles_df, x='readibility', y='sent_length')

"""### Comparing word frequencies in simple documents vs complex documents

Let's look at which words are most overrepresented in each class of document. First we'll count the frequencies of all words and calculate their unigram probabilities per class. Then we'll filter them to keep only words which exist in the English language. Finally, for words that appear in both classes, we'll divide their frequencies to find the most over-represented words in each class.
"""

complex_articles = ' '.join(articles_df[articles_df.filename.str.contains('.en.0.txt')].text.to_list())
simple_articles = ' '.join(articles_df[articles_df.filename.str.contains('.en.4.txt')].text.to_list())

complex_articles =  complex_articles.translate(str.maketrans('', '', string.punctuation)).lower()
simple_articles =  simple_articles.translate(str.maketrans('', '', string.punctuation)).lower()

complex_words = Counter(complex_articles.split())
simple_words = Counter(simple_articles.split())

sw = set(stopwords.words('english'))

complex_words = {w:f for w, f in complex_words.items() if wn.synsets(w) and w not in sw}
simple_words = {w:f for w, f in simple_words.items() if wn.synsets(w) and w not in sw}

total_complex = sum(complex_words.values())
total_simple = sum(simple_words.values())

for word in complex_words:
  complex_words[word] /= total_complex
for word in simple_words:
  simple_words[word] /= total_simple

odds_ratio = {}

for word in complex_words:
  if word in simple_words:
    odds_ratio[word] = complex_words[word] / simple_words[word]

odds_ratio = sorted(odds_ratio.items(), key=lambda x: x[1], reverse=True)

"""Top words in complex documents:"""

odds_ratio[:10]

"""Top words in simple documents:"""

odds_ratio[-10:]

"""Notice how the words most commonly found in complex documents are, naturally, more complex and formal than the words found in the simple documents. This minor differences add up at the document level to increase an article's complexity.

## Select a subset of the data for fine-tuning

**Problem 5.1**

Keep only English articles for which we have the 5 levels of complexity [0-4], from the most complex (version number 0) to the least complex (version number 4). Then select the 10 slugs which have the shortest original/complex documents (version 0) and keep all versions [0-4] of their documents. Your solution should have 50 total rows in this dataframe (10 slugs x 5 levels per article), with the same 6 columns as the original article metadata file.
"""

random_state = 42
num_sample = 100

# Step 1: Filter English-language articles
english_articles = article_list[article_list["language"] == "en"]

# Step 2: Keep only slugs that have all 5 versions [0–4]
version_counts = english_articles.groupby("slug")["version"].nunique()
valid_slugs = version_counts[version_counts == 5].index
filtered_articles = english_articles[english_articles["slug"].isin(valid_slugs)]

# Step 3: Merge with articles_df to get document lengths
# Ensure filename columns match in format
merged = filtered_articles.merge(
    articles_df[["filename", "length"]],
    on="filename",
    how="left"
)

# Step 4: Select version 0 articles and find the 10 with the shortest length
version_0_articles = merged[merged["version"] == 0]
shortest_slugs = version_0_articles.nsmallest(10, "length")["slug"].tolist()

# Step 5: Get all 5 versions for those 10 slugs
document_df_sample = merged[merged["slug"].isin(shortest_slugs)]

# Step 6: Keep only the original 6 article metadata columns
document_df_sample = document_df_sample[["slug", "language", "title", "grade_level", "version", "filename"]]

# Step 7: Sort by slug and version
document_df_sample = document_df_sample.sort_values(by=["slug", "version"]).reset_index(drop=True)

# Final check
assert document_df_sample.shape == (50, 6)

"""## Format data for fine-tuning

Below, we'll format data to fine-tune GPT. Instead of just providing the complex document as the prompt and the simple document as the completion text, we'll add an instruction within the prompt to tell GPT to simplify the document to the particular grade level of the completion text as follows:

```

{"messages": [{"role": "user", "content": "Simplify this document for a student of grade 6: <grade 12 text>"}, {"role": "assistant", "content": "<grade 6 text>"}]}
{"messages": [{"role": "user", "content": "Simplify this document for a student of grade 3: <grade 12 text>"}, {"role": "assistant", "content": "<grade 3 text>"}]}
{"messages": [{"role": "user", "content": "Simplify this document for a student of grade 5: <grade 12 text>"}, {"role": "assistant", "content": "<grade 5 text>"}]}
...
```

Where our complex sentences form the `<grade 12 text>` and the simplified sentences are the `<grade 4/5/6/8 text>`.

Make sure you fine-tuning dataset follows the above format exactly, the autograder expects the `Simplify this document for a student of grade ` part.

Also, as before, we do NOT use `system` messages here.

**Problem 5.2**

Use the dataframe you created above to create a fine-tuning dataset and write a JSON file in the above format with the name defined below. Note that since all prompts will be grade 12 documents and all completions will be grade 4/5/6/8 documents, this file will contain 40 documents in total. Use a mapping of version to grade:

```
{
    grade 8: version 1
    grade 6: version 2
    grade 5: version 3
    grade 4: version 4
}
```

Remember to remove any leading or trailing spaces in the documents.
"""

fine_tuning_documents_filename = 'newsela_documents_finetuning_data.jsonl'

import pandas as pd
import json

# Mapping of version to target grade level
version_to_grade = {
    1: 8,
    2: 6,
    3: 5,
    4: 4
}

# Step 1: Filter prompt (version 0 = grade 12) and completion (versions 1–4)
prompts = document_df_sample[document_df_sample["version"] == 0].copy()
completions = document_df_sample[document_df_sample["version"].isin(version_to_grade.keys())].copy()

# Step 2: Clean whitespace from titles
prompts["title"] = prompts["title"].str.strip()
completions["title"] = completions["title"].str.strip()

# Step 3: Merge prompts with completions by slug
merged = prompts.merge(completions, on="slug", suffixes=("_prompt", "_completion"))

# Step 4: Format as multi-turn conversation JSON for fine-tuning
formatted_data = []
for _, row in merged.iterrows():
    grade = version_to_grade[row["version_completion"]]
    prompt_text = f"Simplify this document for a student of grade {grade}: {row['title_prompt']}"
    completion_text = row["title_completion"]

    formatted_data.append({
        "messages": [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": completion_text}
        ]
    })

# Step 5: Write to a .jsonl file
fine_tuning_documents_filename = "newsela_documents_finetuning_data.jsonl"
with open(fine_tuning_documents_filename, "w") as f:
    for item in formatted_data:
        json.dump(item, f)
        f.write("\n")

# Optional: Show confirmation
print(f"Wrote {len(formatted_data)} entries to {fine_tuning_documents_filename}")

with open(fine_tuning_documents_filename) as f:
  documents_json = f.read().strip().split('\n')

!head '{fine_tuning_documents_filename}'

!wc '{fine_tuning_documents_filename}'

"""## Run the fine-tuning API

**Problem 5.3**

Run the fine-tuning API to create your fine-tuned model. You can read the documentation [here](https://platform.openai.com/docs/guides/fine-tuning/create-a-fine-tuned-model).

You may validate your training file, if you would like, using the script in this [link](https://cookbook.openai.com/examples/chat_finetuning_data_prep).

Although, this is NOT required.

Once you have the data validated, the file needs to be uploaded using the Files API in order to be used with a fine-tuning jobs:
"""

import openai

# Upload the JSONL fine-tuning file
file_path = "newsela_documents_finetuning_data.jsonl"

with open(file_path, "rb") as f:
    response = openai.files.create(file=f, purpose="fine-tune")
    fine_tuning_file_id = response.id

print("Uploaded file ID:", fine_tuning_file_id)


# Start the fine-tuning job
response = openai.fine_tuning.jobs.create(
    training_file=fine_tuning_file_id,
    model="gpt-3.5-turbo",  # or another base model
)

fine_tuning_job_id = response.id
print("Fine-tuning job ID:", fine_tuning_job_id)

##################################################
fine_tuning_file_id = "file-SJC47zQWWyi6YRf1oCDMoZ"
##################################################

"""After ensuring you have uploaded the file, the next step is to create a fine-tuning job."""

##################################################
fine_tuning_job_id = "ftjob-KVSVEJYDjyZD23cdP8xKCcFA"
##################################################

"""You may monitor the status of your fine-tuning job using the below code:"""

response = client.fine_tuning.jobs.list_events(fine_tuning_job_id)

events = response.data
events.reverse()

for event in events:
    print(event.message)

"""Once your fine-tune is completed, you should see a message in the output of the above code saying
`The job has successfully completed`.
"""

# you may also check the status of your fine-tuning job in the UI using this link
print(f"https://platform.openai.com/finetune/{fine_tuning_job_id}")

"""You can retrieve your model ID, by running this command"""

# prints your fine-tuned model ID
client.fine_tuning.jobs.retrieve(fine_tuning_job_id).fine_tuned_model

"""The above output will contain the ID of your model, which you should should copy down for use later. It will look something like this:

```
ft:gpt-4o-mini-2024-07-18:university-of-pennsylvania::BBYu06Qj
```

If you forget to write it down, you can list your fine-tuned runs and models with the following command:
"""

client.fine_tuning.jobs.list(limit=10)

fine_tuned_model_doc = 'ft:gpt-3.5-turbo-0125:upenn::C0wrxUHD' # replace with your REAL ft-model ID from above

"""## Test your fine-tuned model

**Problem 5.4**

Use your fine-tuned model to generate simplified sentences.
"""

# # TODO

# def generate_simple_documents(complex_document, instruction_prompt, finetuned_model):

#   response = None

#   ##################################################

#   ##################################################

#   return response.choices[0].message.content.strip()

# finetuned_model_doc = "model_ID" # replace with your REAL ft-model ID from above
# instruction_prompt = 'Simplify this document for a student of grade 1:'

# fine_tune_test_doc = generate_simple_documents(original_sentence, instruction_prompt, finetuned_model_doc)

import openai

# Required for new SDK versions (≥ 1.0)
client = openai.OpenAI()

def generate_simple_documents(complex_document, instruction_prompt, finetuned_model):
    response = client.chat.completions.create(
        model=finetuned_model,
        messages=[
            {"role": "user", "content": f"{instruction_prompt} {complex_document}"}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Replace with your actual fine-tuned model ID
finetuned_model_doc = fine_tuned_model_doc

# Example instruction and complex text
instruction_prompt = "Simplify this document for a student of grade 1:"
original_sentence = "The Constitution of the United States was written to establish a system of government based on democratic principles."

# Generate simplified output
fine_tune_test_doc = generate_simple_documents(original_sentence, instruction_prompt, finetuned_model_doc)

# Print the result
print(fine_tune_test_doc)

"""Now that we have finetuned a new model to simplify documents, we can test it out in the same manner as before.

# Section 6: Evaluation

In this section, you will attempt to evaluate how well each of the models you built (zeroshot, fewshot, finetuning) perform at this task. You will do a series of evaluations for 10 random sentences where you will be presented with two completions for a sentence and asked to pick which is better. Your responses will decide which model wins the most out of all three. Finally, you will write your thoughts about what you learned from this exercise at the end.
"""

input_sentences = pd.DataFrame(sentences_df.sample(10, random_state=random_state).prompt).reset_index().drop(columns=['index'])

import openai
import pandas as pd

client = openai.OpenAI()  # requires OPENAI_API_KEY env variable

##################################################

zeroshot_prompt = "Simplify this document for a student of grade 4:"
fewshot_prompt = """Simplify this document for a student of grade 4:
Example 1:
Original: Renewable energy helps protect the planet by reducing pollution.
Simplified: Clean energy is better for the Earth because it makes less pollution.

Example 2:
Original: Gravity keeps the planets in orbit around the sun.
Simplified: Gravity makes planets go around the sun.

Now simplify this:
"""

fine_tuned_model = fine_tuned_model_doc  # replace with your real model ID

##################################################

# Define simplification functions
def simplify_zeroshot(prompt, sentence):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"{prompt} {sentence}"}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def simplify_fewshot(fewshot, sentence):
    fewshot_text = fewshot + f"Original: {sentence}\nSimplified:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": fewshot_text}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def generate_simple_sentence(sentence, finetuned_model):
    response = client.chat.completions.create(
        model=finetuned_model,
        messages=[
            {"role": "user", "content": f"Simplify this document for a student of grade 4: {sentence}"}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Prepare 10 input sentences for evaluation
input_sentences = pd.DataFrame({
    "prompt": [
        "The legislative branch is responsible for making laws and consists of the House of Representatives and the Senate.",
        "Water scarcity affects billions of people worldwide due to climate change and mismanagement.",
        "Photosynthesis is the process by which green plants convert sunlight into energy.",
        "The economic recession led to high unemployment and business closures across the country.",
        "Gravity is a natural force that pulls objects toward each other.",
        "The Supreme Court's decision set a legal precedent that affects future rulings.",
        "Renewable energy sources like wind and solar are crucial to combating climate change.",
        "Mitochondria are often referred to as the powerhouses of the cell.",
        "Cultural exchange helps societies appreciate and learn from each other’s traditions.",
        "Vaccines help prevent the spread of infectious diseases and protect communities."
    ]
})

# Generate completions from all three models
input_sentences['completion_1'] = input_sentences['prompt'].apply(lambda x: simplify_zeroshot(zeroshot_prompt, x))
input_sentences['completion_2'] = input_sentences['prompt'].apply(lambda x: simplify_fewshot(fewshot_prompt, x))
input_sentences['completion_3'] = input_sentences['prompt'].apply(lambda x: generate_simple_sentence(x, fine_tuned_model))

ratings = {1:0, 2:0, 3:0}
annot = []

for index, row in input_sentences.iterrows():
  s1, s2 = 1, 2
  for i in range(1,4):
    if i == 2:
      s1, s2 = 2, 3
    elif i == 3:
      s1, s2 = 1, 3

    print('-'*50)
    print(f'\nTask {index*3+i} of 30')
    print('Original sentence:\t', row['prompt'])
    print('Simplification 1:\t', row['completion_'+str(s1)])
    print('Simplification 2:\t', row['completion_'+str(s2)])
    r = int(input('Which simplification is better? Enter 1 or 2: '))
    if i == 1:
      ratings[r]+=1
    if i == 2:
      ratings[r+1]+=1
    elif i == 3:
      if r == 1:
        ratings[1]+=1
      else:
        ratings[3]+=1
    annot.append((row['prompt'], row['completion_'+str(s1)], row['completion_'+str(s2)], r))

annot_df = pd.DataFrame(annot, columns=['input', 'simplification_1', 'simplification_2', 'choice'])
annot_df.to_csv('annotations.csv', sep='\t')

eval_score = pd.DataFrame(ratings.items(), columns=['Model', 'Score'])
eval_score['Model'] = eval_score['Model'].replace([1, 2, 3], ['Zero-shot', 'Few-shot', 'Fine-tuning'])

sns.barplot(data=eval_score, x='Model', y='Score')

"""**Problem 6.1** What did you learn from the above exercise? Comment on the overall performance of the three systems and compare the quality of their outputs against each other. Were the results expected or unexpected? And why? Write your thoughts below. Your answers should be included in your separate **"writeup.pdf"** file and will be manually graded (5 points).

Do not be worried if your fine-tuned model performs worse than your zero-shot or few-shot; fine-tuning does not always lead to better performance. For a funny example of where fine-tuning did not work as expected, see [this](https://rosslazer.com/posts/fine-tuning/).
"""

# TODO in writeup.pdf file

##################################################

##################################################

"""# Final Thoughts

Text simplification is an fascinating application of natural language processing and can teach us much about transforming sequences of text into new sequences of text which exhibit some desired property, whether it be a different language or style of writing.

In this homework, we thought about how we would approach this task as humans, how we can measure simplicitly programmatically, and we even tried to think of and implement some logic to simplify a sentence. We then explored how we can use large language models to perform this task, both by instructing them explicitly, simply just giving them examples of what we expect, and putting both approaches together. Finally, we evaluated how well each of these approaches did in order to gain insight into the best way to solve this problem.

We hope you had fun doing this homework, and we encourage you to use the techniques you learned here at other tasks you find interesting.

# Submission
Here are the deliverables you need to submit to GradeScope:
- Code:
    - This notebook and py file: rename to `homework9.ipynb` and `homework9.py`. You can download the notebook and py file by going to the top-left corner of this webpage, `File -> Download -> Download .ipynb/.py`
- PDF:
  - Your `writeup.pdf` with answers to problems 1.2 and 6.1
- Annotations:
    - The `annotations.csv` file from your human evaluation exercise.
"""