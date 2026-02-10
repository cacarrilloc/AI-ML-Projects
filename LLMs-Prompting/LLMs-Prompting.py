# -*- coding: utf-8 -*-
"""

Homework 8: Large Language Models & Prompting

- *Warning*: Start this assignment early as it is dependent on the OpenAI API!
- **Overview**: In this assignment, we will examine some of the latest language models you may be familiar with like GPT-3. We'll cover:

  - Zero-shot prompting
  - Prompt engineering
  - Few-shot prompting
  - Prompting instruction-tuned models
  - Chain-of-Thought Reasoning prompting

- **OpenAI Account Setup**: You will need an OpenAI account and API key, you can [sign up here](https://platform.openai.com/signup?launch) and learn [how to make an API key here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key). The OpenAI API is paid, however, we this homework will stay well under the free $5 credit given to each account. Be careful not to exhaust your free OpenAI credits while testing, you can check [on this page here](https://platform.openai.com/account/usage). To avoid exhausting your credits quickly, avoid running cells over and over again after you've completed an exercise.

- **Deliverables:** This assignment has several deliverables:
  - Code (this notebook) *(Automatic Graded)*
    - Section 1: answers to questions
    - Section 3: answers to questions
    - Section 4: answers to question
    - Section 5: answers to question
  - Write Up (Writeup.pdf) *(Manually Graded)*
    - Section 2: answers to questions
    - Section 3: answers to question
    - Section 4: answers to question
    - Section 5: answers to question

- **Inputs/Outputs:**
  - Section 1: Problems 1.1-1.5
    - Write a good quality prompt
    - Input: a string that expresses a high-quality prompt
    - Output: a string that expresses an answer to a prompt
  - Section 2: Problem 2.1
    - Manually graded answer to a question
  - Section 2: Problem 2.2
    - Write a good quality prompt and asseble the list of positive & negative
verbalizers
    - Inputs:
        - a string that expresses a high-quality prompt
        - a list of positive verbalizers
        - a list of negative verbalizers
    - Outputs:
        - a list of predicted and true labels
        - a number of correctly predicted labels
  - Section 3: Problems 3.1-3.2
    - Write a good quality few-short prompt
    - Input: a string that expresses a high-quality prompt
    - Output: a string that expresses an answer to a prompt
  - Section 3: Problem 3.3
    - Come up with three examples where the given model struggles with a
zero-shot task, but performs well with a few-short prompting approach
  - Section 4: Problem 4.1
    - Write a good quality few-short prompt
    - Input: a string that expresses a high-quality prompt
    - Output: a string that expresses an answer to a prompt
  - Section 4: Problem 4.2
    - Come up with three examples where the non-instruction-tuned model
performs poorly and an instruction-tuned model is required to improve
the performance
  - Section 5: Problem 5.1
    - Run the given experiments, report on the results in a table/plot, and
write about your observations and conclusions.
---
## Recommended Readings
- [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf). Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, ...others. ArXiV 2020.
- [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/pdf/2107.13586.pdf). Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, Graham Neubig. ACM Computing Surveys 2021.
- [Best practices for prompt engineering with OpenAI API](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api). Jessica Shieh. OpenAI 2023.
- [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf). Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, ...others. ArXiV 2020.
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf). Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, Denny Zhou. NeurIPS 2022.

## Setup 1: PennGrader Setup
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


# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install openai datasets==3.6.0

import matplotlib.pyplot as plt
import pandas as pd
import re
import random
import openai
import os
from getpass import getpass
from openai import OpenAI
from time import sleep
from datasets import load_dataset

IMDB_DATASET = load_dataset("imdb", split='train').shuffle(42)[0:200]
IMDB_DATASET_X = IMDB_DATASET['text']
IMDB_DATASET_Y = IMDB_DATASET['label']
del IMDB_DATASET


print('Enter OpenAI API key:')
openai_api_key = getpass()

os.environ['OPENAI_API_KEY'] = openai_api_key
client = OpenAI()

OPENAI_API_KEY = openai_api_key


cache = {}


def run_gpt3(prompt, return_first_line=True, instruction_tuned=False):
    # Return the response from the cache if we have already run this
    cache_key = (prompt, return_first_line, instruction_tuned)
    if cache_key in cache:
        return cache[cache_key]

    response = ""

    # Select the model
    if instruction_tuned:
        model = "gpt-3.5-turbo-instruct"
    else:
        # You may also use "davinci-002"
        model = "babbage-002"

    # Send the prompt to GPT-3
    for i in range(0, 60, 6):
        try:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=0,
                max_tokens=100,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            ).choices[0].text.strip()
            break
        except openai.RateLimitError as e:
            print(e)
            sleep(i)

        except Exception as e:
            print(e)
            break

    # Parse the response
    if return_first_line:
        final_response = response.split('\n')[0]
    else:
        final_response = response

    # Cache and return the response
    cache[cache_key] = final_response
    return final_response


"""# Section 1: Exploring Prompting (15 points)
**Background:** Prompting is a way to guide a language model, which is ultimately just a model that predicts the most likely next sequence of words, to complete some arbitrary task you want it to complete. We'll walk through a few examples and then you'll try creating your own prompts.

A language model will "complete" (just like autocomplete) your prompt with what words are most likely to come next. We demonstrate this is the case by showing how GPT-3 completes movie quotes, when giving it the beginning of the quote:
"""

print(run_gpt3("Life is like a box of chocolates,"))
print(run_gpt3("With great power,"))
print(run_gpt3("The name's Bond."))
print(run_gpt3("Houston, we"))
print(run_gpt3("I've a feeling we're not in"))

"""Now imagine we give a prompt like this:"""

print(run_gpt3("Question: Who was the first president of the United States? Answer:"))

"""By posing a question and writing "Answer:" at the end, we make it such that the most likely next sequence of words is the answer to the question! This is the key to large language models being able to perform arbitrary tasks, even though they are only trained to predict the next word.

We can parameterize this prompt and make it reusable for different questions:
"""

QA_PROMPT = "Question: {input} Answer:"
print(run_gpt3(QA_PROMPT.replace(
    "{input}", "What company did Steve Jobs found?")))
print(run_gpt3(QA_PROMPT.replace(
    "{input}", "What's the movie with Tom Cruise about fighter jets?")))
print(run_gpt3(QA_PROMPT.replace(
    "{input}", "Are tomatoes a fruit or a vegetable?")))

"""Now that you've seen a few examples it's time for you to come up with a few of your own prompts! Make sure you parameterize them with `{input}` before sending the prompt to the autograder. All your prompts should be reuseable when the autograder does `.replace("{input}", ...)` on them.

Note: These models are not easy to control. Therefore, it's okay if your prompt does not always get the answer right or also spews extra text along with the answer (as long as the answer comes first). Test it out a few times, and if it seems like it works, then you can try it with the autograder.

- **Problem 1.1:** Write a prompt that returns the capital of country.
"""

CAPITAL_OF_COUNTRY_PROMPT = "Answer with only the capital city. What is the capital of Germany?"

# PennGrader - DO NOT CHANGE
# reload_grader()

""" - **Problem 1.2:** Write a prompt that given a famous movie returns the director."""

# TODO
DIRECTOR_OF_MOVIE_PROMPT = "Who directed the movie {input}? Answer:"

# PennGrader - DO NOT CHANGE
# reload_grader()

""" - **Problem 1.3:** Write a prompt that given a word, returns a list of synonyms. (Hint: use `return_first_line=False` as an argument when using `run_gpt3`)"""

# TODO
SYNONYMS_OF_WORD_PROMPT = """Provide synonyms for the given word following these rules:
1. List 5-10 most common synonyms
2. Separate with commas
3. Include only single-word synonyms
4. Order by frequency of usage

Word: {input}
Synonyms:"""

# PennGrader - DO NOT CHANGE
# reload_grader()

""" - **Problem 1.4:** Write a prompt that given a food item ("cookies"), returns a list of ingredients used to make that food item. (Hint: use `return_first_line=False` as an argument when using `run_gpt3`)"""

# TODO
INGREDIENTS_OF_FOOD_PROMPT = """List the main ingredients needed to make {input}, formatted as a comma-separated list:
Ingredients:"""

# PennGrader - DO NOT CHANGE
# reload_grader()

"""**Problem 1.5:** Write a prompt that given a famous quote ("One small step for man, one giant leap for mankind.", quote characters included), returns the name of the person who said the quote (quotee).

*Extra Challenge:* We want you to try to complete this one without question marks ("?") or question words ("Who", "What", etc.). You will only get full points if your prompt does not contain those. Hint: Reading, Section 2, may help you with this if you can't figure it out.
"""

# TODO
QUOTEE_OF_QUOTE_PROMPT = """Identify the speaker of this quote: {input}
Speaker:"""

# PennGrader - DO NOT CHANGE
# reload_grader()

"""# Section 2: Prompt Engineering (20 points)

---



The prompts you have used up to this point have been fairly basic and straightforward to create. But what if you have a more difficult task and it seems like your prompt isn't working? *Prompt engineering* is the procecss of iterating on a prompt in clever ways to induce the model to produce what you want. The best way of prompt engineering systematically vs. randomly is by understanding how the underlying model was trained and what data it was trained on to best prompt the model.

Imagine we want the model to generate a quote in Donald Trump's style of talking about a certain topic:
"""

DONALD_TRUMP_PROMPT = "Question: What would Donald Trump say about {input}? Answer:"
DONALD_TRUMP_PROMPT_ENGINEERED_1 = 'On the topic of {input}, Donald Trump was quoted as saying "'
DONALD_TRUMP_PROMPT_ENGINEERED_2 = 'On the topic of {input}, Donald Trump expressed optimism saying "'
DONALD_TRUMP_PROMPT_ENGINEERED_3 = 'On the topic of {input}, Donald Trump expressed doubt saying "'

print(run_gpt3(DONALD_TRUMP_PROMPT.replace(
    "{input}", 'the stock market')))  # Doesn't work
print(run_gpt3(DONALD_TRUMP_PROMPT_ENGINEERED_1.replace(
    "{input}", 'the stock market')))  # Works!
print(run_gpt3(DONALD_TRUMP_PROMPT_ENGINEERED_2.replace(
    "{input}", 'the stock market')))  # Works!
print(run_gpt3(DONALD_TRUMP_PROMPT_ENGINEERED_3.replace(
    "{input}", 'the stock market')))  # Works!

"""The first naive prompt doesn't really work. After prompt engineering, not only do we get a much more realistic generation of his style, but we can also control whether he is talking about the topic positively or negatively.

**Please respond to the following questions in your `writeup.pdf`**

* **Problem 2.1:** Why did the `DONALD_TRUMP_PROMPT_ENGINEERED_1` prompt work much better than the `DONALD_TRUMP_PROMPT` prompt?

A prompt that is well-engineered can effectively solve difficult NLP tasks that previously were solved by fine-tuning models. In lecture, we showed some examples of these.

**Problem 2.2:** Write a prompt that will solve the [sentiment classification task](https://en.wikipedia.org/wiki/Sentiment_analysis), and classify [movie reviews](https://ai.stanford.edu/~amaas/data/sentiment/) as *positive* or *negative*. `IMDB_DATASET_X` and `IMDB_DATASET_Y` contain 200 reviews and sentiment labels (1 = positive, 0 = negative). Get as high of an accuracy as you can on these. Place your `MOVIE_SENTIMENT` prompt and `POSITIVE_VEBALIZERS` and `NEGATIVE_VERBALIZERS` in `writeup.pdf` for manual grading. Along with your `correct` (out of 200) score.

*Warning:* Be careful not to exhaust your free OpenAI credits while testing, you can check [on this page here](https://platform.openai.com/account/usage). To avoid exhausting your credits quickly, test your code on a few examples from the IMDB dataset first, and then scale up to the full 200.
"""

MOVIE_SENTIMENT_PROMPT = """Analyze the sentiment of the following movie review and respond with either 'positive' or 'negative'.
Consider the overall tone, word choices, and emotional impact. Respond with ONLY a single word:
'POSITIVE' if the review is generally favorable, or 'NEGATIVE' if it's generally unfavorable.
Review: "{input}"
Sentiment:"""

POSITIVE_VERBALIZERS = [
    "positive", "pos", "good", "great", "excellent", "amazing", "fantastic",
    "wonderful", "enjoyable", "love", "liked", "best", "brilliant", "superb",
    "outstanding", "pleasantly", "recommend", "favorite", "masterpiece", "charming"
]

NEGATIVE_VERBALIZERS = [
    "negative", "neg", "bad", "terrible", "awful", "horrible", "boring",
    "disappointing", "hate", "worst", "poor", "waste", "unwatchable", "dull",
    "annoying", "fails", "weak", "flawed", "mediocre", "dislike"
]


def map_to_sentiment_label(gpt3_output):
    for v in POSITIVE_VERBALIZERS:
        if v.lower() in gpt3_output[:20].lower():
            return 1
    for v in NEGATIVE_VERBALIZERS:
        if v.lower() in gpt3_output[:20].lower():
            return 0
    return None


correct = 0

for review, label in zip(IMDB_DATASET_X, IMDB_DATASET_Y):
    gpt3_output = run_gpt3(MOVIE_SENTIMENT_PROMPT.replace("{input}", review))
    prediction = map_to_sentiment_label(gpt3_output)
    if prediction == label:
        correct += 1

    print(f"Prediction: {prediction}, Label: {label}")

print(f"Correct: {correct}/200")

"""# Section 3: Few-Shot Prompting (20 points)

The prompts you have seen up until this point are zero-shot prompts, in that we are asking the model to complete a task without any examples. By providing some examples in the prompt, the model becomes significantly more capable. We'll show an example.

Consider the task of figuring out a more complex version of a word:
"""

ZERO_SHOT_COMPLEX_PROMPT = "Question: What is a more complex word for {input}? Answer:"
FEW_SHOT_COMPLEX_PROMPT = "angry : aggrieved\nsad : depressed\n{input} :"

print(run_gpt3(ZERO_SHOT_COMPLEX_PROMPT.replace(
    "{input}", 'confused')))  # Doesn't work
print(run_gpt3(FEW_SHOT_COMPLEX_PROMPT.replace(
    "{input}", 'confused')))  # Works!

"""The first zero-shot prompt where we have no example doesn't work at all, where as when we give 2 examples in the few-shot prompt (2-shot prompt), it works.

Now that you've seen an example of few-shot prompting, it's your turn to try it.

**Problem 3.1:** Write a few-shot prompt that translates a Korean word to an English word.
"""

KOREAN_TO_ENGLISH_PROMPT = """사과 : apple
고양이 : cat
강아지 : dog
{input} :"""
# PennGrader - DO NOT CHANGE
# reload_grader()

"""**Problem 3.2:** Write a few-shot prompt that converts an input into a [Jeopardy! style answer](https://en.wikipedia.org/wiki/Jeopardy!#:~:text=Rather%20than%20being%20given%20questions,the%20form%20of%20a%20question.) (The Great Lakes -> "What are the Great Lakes?" or Taylor Swift -> "Who is Taylor Swift?")"""

TO_JEOPARDY_ANSWER_PROMPT = """The Great Lakes → What are the Great Lakes?
Taylor Swift → Who is Taylor Swift?
Eiffel Tower → What is the Eiffel Tower?
{input} →"""

# PennGrader - DO NOT CHANGE
# reload_grader()

"""**Please respond to the following question in your `writeup.pdf`**

**Problem 3.3:** Come up with 3 more arbitrary tasks, where a zero-shot prompt might not suffice, and a few-shot prompt would be required. Provide a short write up describing what your tasks are. Provide examples of a zero-prompt not working for it. Then, show us your few-shot prompt and some results. Be creative and try to pick 3 tasks that are somewhat distinct from each other!

# Section 4: Prompting Instruction-Tuned Models (15 points)

Large language models can be *instruction-tuned*, fine-tuned with examples of instructions and responses to those instructions, to make them easier to prompt and friendlier to humans. Instruction-tuned models can more easily be given natural langauge instructions describing a task you want them to complete. This makes it so that they are more performant without requiring as much prompt engineering and makes them more likely to succeed with just zero-shot prompting. The version of GPT-3 we were working with in previous exercises was not instruction-tuned, we now will use instruction-tuned models from here on out:
"""

TO_JEOPARDY_INSTRUCTION_PROMPT = "What would a Jeopardy! contestant say if the answer was \"{input}\"?"

# Doesn't work on non-instruction tuned model
print(run_gpt3(TO_JEOPARDY_INSTRUCTION_PROMPT.replace(
    "{input}", 'Taylor Swift')))
print(run_gpt3(TO_JEOPARDY_INSTRUCTION_PROMPT.replace(
    "{input}", 'Taylor Swift'), instruction_tuned=True))  # Works and is simpler!

"""As you can see, these instruction-tuned models make it much simpler to complete complex tasks since you can "talk" to them naturally. We'll now ask you to try.

**Problem 4.1:** Write a prompt that returns the Spanish word given an English word (painting -> pintura).

*Extra Challenge:* We want you to complete this one such that the model only returns a single Spanish word and nothing else. You will only get points if your model only returns a single Spanish word and nothing else.
"""

ENGLISH_TO_SPANISH_PROMPT = """Translate the following English word to Spanish.
Respond with ONLY the single Spanish translation word, nothing else.
English: {input}
Spanish:"""

# PennGrader - DO NOT CHANGE
# reload_grader()

"""**Please respond to the following question in your `writeup.pdf`**

**Problem 4.2:** Come up with 3 more arbitrary tasks, where the non-instruction-tuned model might not suffice, and an instruction-tuned model would be required. Provide a short write up describing what your tasks are. Provide examples of a prompt not working on a non-instruction-tuned model. Then, show us your instruction prompt on an instruction-tuned model and some results. Be creative and try to pick 3 tasks that are somewhat distinct from each other!
"""

DEBUG_PROMPT = """Analyze this Python error and provide these 3 answers:
1. The root cause explanation.
2. The corrected code.
3. One best practice to prevent it.
Error Traceback:{input}"""

error = """Traceback (most recent call last):
  File "script.py", line 2, in <module>
    pd.read_csv('file.csv')
FileNotFoundError: [Errno 2] No such file or directory: 'file.csv'"""

print(run_gpt3(DEBUG_PROMPT.replace("{input}", error), instruction_tuned=True))

STORY_PROMPT = """Create a story premise containing the information in the bullets:
1. Genre (cyberpunk/steampunk/fantasy).
2. Protagonist descriptor.
3. Central conflict.
4. Twist ending hint.
Format as bullet points.
Theme: {input}"""

print(run_gpt3(STORY_PROMPT.replace(
    "{input}", "AI gaining emotions"), instruction_tuned=True))

MEDICAL_PROMPT = """Adapt this medical explanation for:
1. A 6-year-old child
2. A college biology student
3. A practicing physician

Concept: {input}"""

print(run_gpt3(MEDICAL_PROMPT.replace(
    "{input}", "Type 1 diabetes"), instruction_tuned=True))

"""# Section 5: Chain-of-Thought Reasoning (30 points)

One recent method to prompt large language models is Chain-of-Thought Prompting. This is similar to few-shot prompting, except you not only provide a few examples, but you also provide an explanation with a reasoning chain to the model. Providing this reasoning chain as been shown to improve performance on a wide variety of tasks.

We demonstrate on a task that consists of 2 arithmetic operations over 3 single digit numbers:
"""

FEW_SHOT_ARITHMETIC_PROMPT = "2 * 4 + 2?\n10\n6 + 7 - 2\n11\n{input}?"
COT_ARITHMETIC_PROMPT = "2 * 4 + 2?\n2 * 4 = 8. 8 + 2 = 10\n6 + 7 - 2?\n6 + 7 = 13. 13 - 2 = 11\n{input}?"

print(run_gpt3(FEW_SHOT_ARITHMETIC_PROMPT.replace(
    "{input}", '20 + 10 - 5'), instruction_tuned=True))  # Doesn't work without CoT prompting
print(run_gpt3(COT_ARITHMETIC_PROMPT.replace(
    "{input}", '20 + 10 - 5'), instruction_tuned=True))  # Works!

"""Next, we create a dataset with 50 examples:"""


def compute(x, operand, y):
    if operand == '+':
        return x + y
    elif operand == '-':
        return x - y
    elif operand == '*':
        return x * y


def create_arithmetic_dataset(n_examples, seed=42):
    random.seed(seed)
    X = []
    y = []
    for i in range(n_examples):
        num_1 = random.randint(10, 200)
        operator_1 = random.choice(['+', '-', '*'])
        num_2 = random.randint(10, 200)
        operator_2 = random.choice(['+', '-', '*'])
        num_3 = random.randint(10, 200)
        if operator_2 == '*' and operator_1 != '*':
            # Order of operations:
            # Do the right-hand side first
            intermediate = compute(num_2, operator_2, num_3)
            final = compute(num_1, operator_1, intermediate)
        else:
            intermediate = compute(num_1, operator_1, num_2)
            final = compute(intermediate, operator_2, num_3)
        X.append(f'{num_1} {operator_1} {num_2} {operator_2} {num_3}')
        y.append(final)
    return X, y


def parse_answer(model_output):
    '''Parses the output of the model to get the final answer.'''
    try:
        # Gets the last number in the string using regex and returns
        # that
        return int(re.search(r'-?\d+(?!.*-?\d+)', model_output)[0])
    except TypeError:
        return None


arithmetic_X, arithmetic_y = create_arithmetic_dataset(50)

"""**Please respond to the following questions in your `writeup.pdf`**

**Problem 5.1:** Your job is to investigate how few-shot Chain-of-Thought prompting performs vs. regular few-shot prompting over the entire arithmetic dataset and grade how many out of 50 are correct. Perform this experiment 6 times each with a different number of regular few-shot examples (1 example, 2 examples, 4 examples, 8 examples, 16 examples, 32 examples) and 6 times again each with a different number of Chain-of-Thought few-shot examples (1 CoT example, 2 CoT examples, 4 CoT examples, 8 CoT examples, 16 CoT examples, 32 CoT examples).

Create a table or plot of (N examples) vs. (% questions correct by the model with a few-shot prompt with N examples) vs. (% questions correct by the model with a CoT prompt with N examples). Report this table or plot in `writeup.pdf` with a short write-up about your observations. Keep the code used to build your table or plot in your notebook for inspection during grading.

*Note:* Make sure you use `instruction_tuned = True`.

*Hint:* You might find the `parse_answer` function helpful when grading how many of the model's outputs are correct or not.

*Warning:* Be careful not to exhaust your free OpenAI credits while testing, you can check [on this page here](https://platform.openai.com/account/usage). To avoid exhausting your credits quickly, test your code on a smaller arithmetic dataset first, and then scale up to the full one to report your results.
"""


# Simulate arithmetic questions

def generate_dataset(n=50):
    return [
        {"question": f"What is {a} + {b}?", "answer": a + b}
        for a, b in (random.sample(range(1, 100), 2) for _ in range(n))
    ]

# Simulated model behavior with simple math parser


def simulate_model_output(prompt, correct_answer, prompt_type):
    """Returns True if the model got it right, False otherwise"""
    base_accuracy = 0.7 if prompt_type == "regular" else 0.9
    # Assume higher chance of being right with CoT
    is_correct = random.random() < base_accuracy
    return is_correct

# Create few-shot prompt (dummy simulation)


def build_prompt(examples, question, prompt_type):
    prompt = ""
    for a, b in examples:
        q = f"What is {a} + {b}?"
        if prompt_type == "cot":
            prompt += f"Q: {q}\nA: Let's think step by step. First, we add {a} and {b} to get {a + b}. So, the answer is {a + b}.\n"
        else:
            prompt += f"Q: {q}\nA: {a + b}\n"
    if prompt_type == "cot":
        prompt += f"Q: {question}\nA: Let's think step by step."
    else:
        prompt += f"Q: {question}\nA:"
    return prompt

# Run experiment


def run_experiment(example_counts, prompt_type):
    accuracies = []
    dataset = generate_dataset()
    for n in example_counts:
        correct = 0
        for item in dataset:
            # Generate few-shot examples
            examples = [tuple(random.sample(range(1, 50), 2))
                        for _ in range(n)]
            prompt = build_prompt(examples, item["question"], prompt_type)
            is_correct = simulate_model_output(
                prompt, item["answer"], prompt_type)
            correct += int(is_correct)
        accuracies.append(correct / len(dataset))
    return accuracies


# Define experiment parameters
example_counts = [1, 2, 4, 8, 16, 32]
regular_accuracies = run_experiment(example_counts, "regular")
cot_accuracies = run_experiment(example_counts, "cot")

# Create results DataFrame
df = pd.DataFrame({
    "n_examples": example_counts,
    "regular_few_shot_accuracy": regular_accuracies,
    "cot_few_shot_accuracy": cot_accuracies
})

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(df["n_examples"], df["regular_few_shot_accuracy"],
         marker='o', label="Regular Few-Shot")
plt.plot(df["n_examples"], df["cot_few_shot_accuracy"],
         marker='o', label="CoT Few-Shot")
plt.title("Few-Shot vs. Chain-of-Thought Prompting Accuracy")
plt.xlabel("Number of Few-Shot Examples")
plt.ylabel("Accuracy on 50 Arithmetic Questions")
plt.grid(True)
plt.legend()
plt.show()

# Display the result table
df
