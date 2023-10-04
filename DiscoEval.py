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
        datasets.BuilderConfig(
            name=constants.BSOARXIV,
            version=VERSION,
            description="The BSO Task with the arxiv dataset.",
        ),
        datasets.BuilderConfig(
            name=constants.BSOWIKI,
            version=VERSION,
            description="The BSO Task with the wiki dataset.",
        ),
        datasets.BuilderConfig(
            name=constants.BSOROCSTORY,
            version=VERSION,
            description="The BSO Task with the rocstory dataset.",
        ),
    ]

    def _info(self):
        if self.config.name in [constants.SPARXIV, constants.SPROCSTORY, constants.SPWIKI]:
            features_dict = {
                constants.TEXT_COLUMN_NAME[i]: datasets.Value('string')
                for i in range(constants.SP_TEXT_COLUMNS)
            }
            features_dict[constants.LABEL_NAME] = datasets.ClassLabel(names=constants.SP_LABELS)
            features = datasets.Features(features_dict)

        elif self.config.name in [constants.BSOARXIV, constants.BSOWIKI, constants.BSOROCSTORY]:
            features_dict = {
                constants.TEXT_COLUMN_NAME[i]: datasets.Value('string')
                for i in range(constants.BSO_TEXT_COLUMNS)
            }
            features_dict[constants.LABEL_NAME] = datasets.ClassLabel(names=constants.BSO_LABELS)
            features = datasets.Features(features_dict)

        elif self.config.name in [constants.DCCHAT, constants.DCWIKI]:
            features_dict = {
                constants.TEXT_COLUMN_NAME[i]: datasets.Value('string')
                for i in range(constants.DC_TEXT_COLUMNS)
            }
            features_dict[constants.LABEL_NAME] = datasets.ClassLabel(names=constants.DC_LABELS)
            features = datasets.Features(features_dict)

        elif self.config.name in [constants.RST]:
            features_dict = {
                constants.TEXT_COLUMN_NAME[i]: [datasets.Value('string')]
                for i in range(constants.RST_TEXT_COLUMNS)
            }
            features_dict[constants.LABEL_NAME] = datasets.ClassLabel(names=constants.RST_LABELS)
            features = datasets.Features(features_dict)

        elif self.config.name in [constants.PDTB_E]:
            features_dict = {
                constants.TEXT_COLUMN_NAME[i]: datasets.Value('string')
                for i in range(constants.PDTB_E_TEXT_COLUMNS)
            }
            features_dict[constants.LABEL_NAME] = datasets.ClassLabel(names=constants.PDTB_E_LABELS)
            features = datasets.Features(features_dict)

        elif self.config.name in [constants.PDTB_I]:
            features_dict = {
                constants.TEXT_COLUMN_NAME[i]: datasets.Value('string')
                for i in range(constants.PDTB_I_TEXT_COLUMNS)
            }
            features_dict[constants.LABEL_NAME] = datasets.ClassLabel(names=constants.PDTB_I_LABELS)
            features = datasets.Features(features_dict)

        elif self.config.name in [constants.SSPABS]:
            features_dict = {
                constants.TEXT_COLUMN_NAME[i]: datasets.Value('string')
                for i in range(constants.SSPABS_TEXT_COLUMNS)
            }
            features_dict[constants.LABEL_NAME] = datasets.ClassLabel(names=constants.SSPABS_LABELS)
            features = datasets.Features(features_dict)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.name in [constants.SPARXIV, constants.SPROCSTORY, constants.SPWIKI]:
            data_dir = constants.SP_DATA_DIR + "/" + constants.SP_DIRS[self.config.name]
            train_name = constants.SP_TRAIN_NAME
            valid_name = constants.SP_VALID_NAME
            test_name = constants.SP_TEST_NAME

        elif self.config.name in [constants.BSOARXIV, constants.BSOWIKI, constants.BSOROCSTORY]:
            data_dir = constants.BSO_DATA_DIR + "/" + constants.BSO_DIRS[self.config.name]
            train_name = constants.BSO_TRAIN_NAME
            valid_name = constants.BSO_VALID_NAME
            test_name = constants.BSO_TEST_NAME

        elif self.config.name in [constants.DCCHAT, constants.DCWIKI]:
            data_dir = constants.DC_DATA_DIR + "/" + constants.DC_DIRS[self.config.name]
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

        urls_to_download = {
            "train": data_dir + "/" + train_name,
            "valid": data_dir + "/" + valid_name,
            "test": data_dir + "/" + test_name,
        }
        logger = logging.getLogger(__name__)
        data_dirs = dl_manager.download_and_extract(urls_to_download)
        logger.info(f"Data directories: {data_dirs}")
        downloaded_files = dl_manager.download_and_extract(data_dirs)
        logger.info(f"Downloading Completed")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files['train'],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files['valid'],
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files['test'],
                    "split": "test"
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        logger = logging.getLogger(__name__)
        logger.info(f"Current working dir: {os.getcwd()}")
        logger.info("generating examples from = %s", filepath)
        if self.config.name == constants.RST:
            data = pickle.load(open(filepath, "rb"))
            for key, line in enumerate(data):
                example = {constants.TEXT_COLUMN_NAME[i]: sent for i, sent in enumerate(line[1:])}
                example[constants.LABEL_NAME] = line[0]
                yield key, example

        else:
            with io.open(filepath, mode='r', encoding='utf-8') as f:
                for key, line in enumerate(f):
                    line = line.strip().split("\t")
                    example = {constants.TEXT_COLUMN_NAME[i]: sent for i, sent in enumerate(line[1:])}
                    example[constants.LABEL_NAME] = line[0]
                    yield key, example
