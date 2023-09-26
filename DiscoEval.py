# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import io
import datasets
import constants
import pickle
import logging
from huggingface_hub import snapshot_download, hf_hub_url, hf_hub_download

_CITATION = """\
@InProceedings{mchen-discoeval-19,
                title = {Evaluation Benchmarks and Learning Criteria for Discourse-Aware Sentence Representations},
                author = {Mingda Chen and Zewei Chu and Kevin Gimpel},
                booktitle = {Proc. of {EMNLP}},
                year={2019}
              }
"""

_DESCRIPTION = """\
This dataset contains all tasks of the DiscoEval benchmark for sentence representation learning.
"""

_HOMEPAGE = "https://github.com/ZeweiChu/DiscoEval"


# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "DiscoEval": "https://huggingface.co/.zip",
}


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class DiscoEvalSentence(datasets.GeneratorBasedBuilder):
    """DiscoEval Benchmark"""
    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=constants.SPARXIV,
            version=VERSION,
            description="Sentence positioning dataset from arXiv",
        ),
        datasets.BuilderConfig(
            name=constants.SPROCSTORY,
            version=VERSION,
            description="Sentence positioning dataset from ROCStory",
        ),
        datasets.BuilderConfig(
            name=constants.SPWIKI,
            version=VERSION,
            description="Sentence positioning dataset from Wikipedia",
        ),
        datasets.BuilderConfig(
            name=constants.DCCHAT,
            version=VERSION,
            description="Discourse Coherence dataset from chat",
        ),
        datasets.BuilderConfig(
            name=constants.DCWIKI,
            version=VERSION,
            description="Discourse Coherence dataset from Wikipedia",
        ),
        datasets.BuilderConfig(
            name=constants.RST,
            version=VERSION,
            description="The RST Discourse Treebank dataset ",
        ),
        datasets.BuilderConfig(
            name=constants.PDTB_E,
            version=VERSION,
            description="The Penn Discourse Treebank - Explicit dataset.",
        ),
        datasets.BuilderConfig(
            name=constants.PDTB_I,
            version=VERSION,
            description="The Penn Discourse Treebank - Implicit dataset.",
        ),
        datasets.BuilderConfig(
            name=constants.SSPABS,
            version=VERSION,
            description="The SSP dataset.",
        ),
    ]

    DEFAULT_CONFIG_NAME = constants.SPARXIV  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset

        if self.config.name in [constants.SPARXIV, constants.SPROCSTORY, constants.SPWIKI]:
            features_dict = {
                constants.TEXT_COLUMN_NAME[i]: datasets.Value('string')
                for i in range(constants.SP_TEXT_COLUMNS + 1)
            }
            features_dict[constants.LABEL_NAME] = datasets.ClassLabel(names=constants.SP_LABELS)
            features = datasets.Features(features_dict)

        elif self.config.name in [constants.DCCHAT, constants.DCWIKI]:
            features_dict = {
                constants.TEXT_COLUMN_NAME[i]: datasets.Value('string')
                for i in range(constants.DC_TEXT_COLUMNS + 1)
            }
            features_dict[constants.LABEL_NAME] = datasets.ClassLabel(names=constants.DC_LABELS)
            features = datasets.Features(features_dict)

        elif self.config.name in [constants.RST]:
            features_dict = {
                constants.TEXT_COLUMN_NAME[i]: [datasets.Value('string')]
                for i in range(constants.RST_TEXT_COLUMNS + 1)
            }
            features_dict[constants.LABEL_NAME] = datasets.ClassLabel(names=constants.RST_LABELS)
            features = datasets.Features(features_dict)

        elif self.config.name in [constants.PDTB_E]:
            features_dict = {
                constants.TEXT_COLUMN_NAME[i]: datasets.Value('string')
                for i in range(constants.PDTB_E_TEXT_COLUMNS + 1)
            }
            features_dict[constants.LABEL_NAME] = datasets.ClassLabel(names=constants.PDTB_E_LABELS)
            features = datasets.Features(features_dict)

        elif self.config.name in [constants.PDTB_I]:
            features_dict = {
                constants.TEXT_COLUMN_NAME[i]: datasets.Value('string')
                for i in range(constants.PDTB_I_TEXT_COLUMNS + 1)
            }
            features_dict[constants.LABEL_NAME] = datasets.ClassLabel(names=constants.PDTB_I_LABELS)
            features = datasets.Features(features_dict)

        elif self.config.name in [constants.SSPABS]:
            features_dict = {
                constants.TEXT_COLUMN_NAME[i]: datasets.Value('string')
                for i in range(constants.SSPABS_TEXT_COLUMNS + 1)
            }
            features_dict[constants.LABEL_NAME] = datasets.ClassLabel(names=constants.SSPABS_LABELS)
            features = datasets.Features(features_dict)

        else:  # This is an example to show how to have different features for "first_domain" and "second_domain"
            features = datasets.Features(
                {
                    "sentence": datasets.Value("string"),
                    "option2": datasets.Value("string"),
                    "second_domain_answer": datasets.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive


        # urls = _URLS[self.config.name]
        # data_dir = dl_manager.download_and_extract(urls)

        if self.config.name in [constants.SPARXIV, constants.SPROCSTORY, constants.SPWIKI]:
            # subfolder = os.path.join(constants.SP_DATA_DIR, constants.SP_DIRS[self.config.name])
            data_dir = constants.SP_DATA_DIR + "/" + constants.SP_DIRS[self.config.name]
            snapshot_download(
                repo_id="OfekGlick/DiscoEval",
                repo_type="dataset",
                local_dir='./',
                ignore_patterns=["*.py", "*.gitignore", "*.gitattributes", "*.DS_Store", "*.md"],
            )
            # train_url = hf_hub_download(
            #     repo_id="OfekGlick/DiscoEval",
            #     filename=constants.SP_TRAIN_NAME,
            #     subfolder=subfolder,
            #     repo_type="dataset",
            #     local_dir='./',
            # )
            #
            # valid_url = hf_hub_download(
            #     repo_id="OfekGlick/DiscoEval",
            #     filename=constants.SP_VALID_NAME,
            #     subfolder=subfolder,
            #     repo_type="dataset",
            #     local_dir='./',
            # )
            # text_url = hf_hub_download(
            #     repo_id="OfekGlick/DiscoEval",
            #     filename=constants.SP_TEST_NAME,
            #     subfolder=subfolder,
            #     repo_type="dataset",
            #     local_dir='./',
            # )
            # data_dir = dl_manager.download_and_extract(urls)
            train_name = constants.SP_TRAIN_NAME
            valid_name = constants.SP_VALID_NAME
            test_name = constants.SP_TEST_NAME

        elif self.config.name in [constants.DCCHAT, constants.DCWIKI]:
            data_dir = os.path.join(constants.DC_DATA_DIR, constants.DC_DIRS[self.config.name])
            train_name = constants.DC_TRAIN_NAME
            valid_name = constants.DC_VALID_NAME
            test_name = constants.DC_TEST_NAME

        elif self.config.name in [constants.RST]:
            data_dir = constants.RST_DATA_DIR
            train_name = constants.RST_TRAIN_NAME
            valid_name = constants.RST_VALID_NAME
            test_name = constants.RST_TEST_NAME

        elif self.config.name in [constants.PDTB_E, constants.PDTB_I]:
            data_dir = os.path.join(constants.PDTB_DATA_DIR, constants.PDTB_DIRS[self.config.name])
            train_name = constants.PDTB_TRAIN_NAME
            valid_name = constants.PDTB_VALID_NAME
            test_name = constants.PDTB_TEST_NAME

        elif self.config.name in [constants.SSPABS]:
            data_dir = constants.SSPABS_DATA_DIR
            train_name = constants.SSPABS_TRAIN_NAME
            valid_name = constants.SSPABS_VALID_NAME
            test_name = constants.SSPABS_TEST_NAME

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, train_name),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, valid_name),
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, test_name),
                    "split": "test"
                },
            ),
        ]




    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        logger = logging.getLogger(__name__)
        logger.info(f"Current working dir: {os.getcwd()}")
        logger.info("generating examples from = %s", filepath)
        print(f"Current working dir: {os.getcwd()}")
        print(f"Current working dir: {os.listdir(os.getcwd())}")


        if self.config.name in [constants.SPARXIV, constants.SPROCSTORY, constants.SPWIKI,
                                constants.DCWIKI, constants.DCCHAT,
                                constants.PDTB_E, constants.PDTB_I,
                                constants.SSPABS]:
            with io.open(filepath, mode='r', encoding='utf-8') as f:
                for key, line in enumerate(f):
                    line = line.strip().split("\t")
                    example = {constants.TEXT_COLUMN_NAME[i]: sent for i, sent in enumerate(line[1:])}
                    example[constants.LABEL_NAME] = line[0]
                    yield key, example

        elif self.config.name in [constants.RST]:
            data = pickle.load(open(filepath, "rb"))
            for key, line in enumerate(data):
                example = {constants.TEXT_COLUMN_NAME[i]: sent for i, sent in enumerate(line[1:])}
                example[constants.LABEL_NAME] = line[0]
                yield key, example

        # TODO: implement other datasets
        else:
            yield 0, {
                "sentence": 'example sentences',
                "option2": 'another example sentence',
                "second_domain_answer": "" if split == "test" else 'label',
            }


if __name__ == '__main__':
    temp = os.path.join(constants.SP_DATA_DIR, constants.SP_DIRS[constants.SPARXIV])
    ofek = 5
