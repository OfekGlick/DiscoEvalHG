---
license: bsd
task_categories:
- text-classification
language:
- en
tags:
- Discourse
- Discourse Evaluation
- NLP
pretty_name: DiscoEval
size_categories:
- 100K<n<1M
---

# DiscoEval Benchmark Datasets

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Dataset Sources](#dataset-sources)
  - [Supported Tasks](#supported-tasks)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Additional Information](#additional-information)
  - [Benchmark Creators](#benchmark-creators)
  - [Citation Information](#citation-information)
- [Loading Data Examples](#loading-data-examples)
  - [Loading Data for Sentence Positioning Task with the Arxiv data source](#loading-data-for-sentence-positioning-task-with-the-arxiv-data-source)

## Dataset Description

- **Repository:** [DiscoEval repository](https://github.com/ZeweiChu/DiscoEval)
- **Paper:** [Evaluation Benchmarks and Learning Criteria for Discourse-Aware Sentence Representations](https://arxiv.org/pdf/1909.00142)

### Dataset Summary

The DiscoEval is an English-language Benchmark that contains a test suite of 7
tasks to evaluate whether sentence representations include semantic information
relevant to discourse processing. The benchmark datasets offer a collection of 
tasks designed to evaluate natural language understanding models in the context 
of discourse analysis and coherence.

### Dataset Sources

- **Arxiv**: A repository of scientific papers and research articles.
- **Wikipedia**: An extensive online encyclopedia with articles on diverse topics.
- **Rocstory**: A dataset consisting of fictional stories.
- **Ubuntu IRC channel**: Conversational data extracted from the Ubuntu Internet Relay Chat (IRC) channel.
- **PeerRead**: A dataset of scientific papers frequently used for discourse-related tasks.
- **RST Discourse Treebank**: A dataset annotated with Rhetorical Structure Theory (RST) discourse relations.
- **Penn Discourse Treebank**: Another dataset with annotated discourse relations, facilitating the study of discourse structure.


### Supported Tasks

1. **Sentence Positioning**
   - **Datasets Sources**: Arxiv, Wikipedia, Rocstory
   - **Description**: Determine the correct placement of a sentence within a given context of five sentences. To form the input when training classifiers encode the five sentences to vector representations \\(x_i\\). As input to the classfier we include \\(x_1\\) and the contcatination of \\(x_1 - x_i\\) for all \\(i\\): \\([x_1, x_1 - x_2, x_1-x_3,x_1-x_4,x_1-x_5]\\)

2. **Binary Sentence Ordering**
   - **Datasets Sources**: Arxiv, Wikipedia, Rocstory
   - **Description**: Determining whether two sentences are in the correct consecutive order, identifying the more coherent structure. To form the input when training classifiers, we concatenate the embeddings of both sentences with their element-wise difference: \\([x_1, x_2, x_1-x_2]\\)

3. **Discourse Coherence**
   - **Datasets Sources**: Ubuntu IRC channel, Wikipedia
   - **Description**: Determine whether a sequence of six sentences form a coherent paragraph. To form the input when training classifiers, encode all sentences to vector representations and concatenate all of them: \\([x_1, x_2, x_3, x_4, x_5, x_6]\\)

4. **Sentence Section Prediction**
   - **Datasets Sources**: Constructed from PeerRead
   - **Description**: Determine the section or category to which a sentence belongs within a scientific paper, based on the content and context. To form the input when training classifiers, simply input the sentence embedding.

5. **Discourse Relations**
   - **Datasets Sources**: RST Discourse Treebank, Penn Discourse Treebank
   - **Description**: Identify and classify discourse relations between sentences or text segments, helping to reveal the structure and flow of discourse. To form the input when training classifiers, refer to the [original paper](https://arxiv.org/pdf/1909.00142) for instructions


### Languages

The text in all datasets is in English. The associated BCP-47 code is `en`.


## Dataset Structure

### Data Instances

All tasks are classification tasks, and they differ by the number of sentences per example and the type of label.

An example from the Sentence Positioning task would look as follows:
```
{
'sentence_1': 'Dan was overweight as well.',
'sentence_2': 'Dan's parents were overweight.',
'sentence_3': 'The doctors told his parents it was unhealthy.',
'sentence_4': 'His parents understood and decided to make a change.',
'sentence_5': 'They got themselves and Dan on a diet.'
'label': '1'
}
```
The label is '1' since the first sentence should go at position number 1 (counting from zero)

Another example from the Binary Sentence Ordering task would look as follows:
```
{
'sentence_1': 'When she walked in, she felt awkward.',
'sentence_2': 'Janet decided to go to her high school's party.',
'label': '0'
}
```
The label is '0' because this is not the correct order of the sentences. It should be sentence_2 and then sentence_1.

For more examples, you can refer the [original paper]((https://arxiv.org/pdf/1909.00142).

### Data Fields
In this benchmark, all data fields are string, including the labels.

### Data Splits

The data is split into training, validation and test set for each of the tasks in the benchmark.

|       Task and Dataset      | Train   | Valid | Test |
| -----                       | ------ | ----- | ---- |
| Sentence Positioning: Arxiv| 10000 |  4000 | 4000|
| Sentence Positioning: Rocstory| 10000 |  4000 | 4000|
| Sentence Positioning: Wiki| 10000 |  4000 | 4000|
| Binary Sentence Ordering: Arxiv| 20000 |  8000 | 8000|
| Binary Sentence Ordering: Rocstory| 20000 |  8000 | 8000|
| Binary Sentence Ordering: Wiki| 20000 |  8000 | 8000|
| Discourse Coherence: Chat| 5816 |  1834 | 2418|
| Discourse Coherence: Wiki| 10000 |  4000 | 4000|
| Sentence Section Prediction       | 10000 |  4000 | 4000 |
| Discourse Relation: Penn Discourse Tree Bank: Implicit    | 8693  |  2972 | 3024 |
| Discourse Relation: Penn Discourse Tree Bank: Explicit    | 9383  |  3613 | 3758 |
| Discourse Relation: RST Discourse Tree Bank    | 17051  |  2045 | 2308 |

## Additional Information

### Benchmark Creators

This benchmark was created by Mingda Chen, Zewei Chu and Kevin Gimpel during work done at the University of Chicago and the Toyota Technologival Institute at Chicago.

### Citation Information

```

@inproceedings{mchen-discoeval-19,
                title = {Evaluation Benchmarks and Learning Criteria for Discourse-Aware Sentence Representations},
                author = {Mingda Chen and Zewei Chu and Kevin Gimpel},
                booktitle = {Proc. of {EMNLP}},
                year={2019}
              }
```

## Loading Data Examples

### Loading Data for Sentence Positioning Task with the Arxiv data source

```python
from datasets import load_dataset

# Load the Sentence Positioning dataset
dataset = load_dataset(path="OfekGlick/DiscoEval", name="SParxiv")

# Access the train, validation, and test splits
train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]

# Example usage: Print the first few training examples
for example in train_data[:5]:
    print(example)
```

The other possible inputs for the `name` parameter are:
`SParxiv`, `SProcstory`, `SPwiki`, `SSPabs`, `PDTB-I`, `PDTB-E`, `BSOarxiv`, `BSOrocstory`, `BSOwiki`, `DCchat`, `DCwiki`, `RST`