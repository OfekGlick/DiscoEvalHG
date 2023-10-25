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
import DiscoEvalConstants
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
            name=DiscoEvalConstants.SPARXIV,
            version=VERSION,
            description="Sentence positioning dataset from arXiv",
        ),
        datasets.BuilderConfig(
            name=DiscoEvalConstants.SPROCSTORY,
            version=VERSION,
            description="Sentence positioning dataset from ROCStory",
        ),
        datasets.BuilderConfig(
            name=DiscoEvalConstants.SPWIKI,
            version=VERSION,
            description="Sentence positioning dataset from Wikipedia",
        ),
        datasets.BuilderConfig(
            name=DiscoEvalConstants.DCCHAT,
            version=VERSION,
            description="Discourse Coherence dataset from chat",
        ),
        datasets.BuilderConfig(
            name=DiscoEvalConstants.DCWIKI,
            version=VERSION,
            description="Discourse Coherence dataset from Wikipedia",
        ),
        datasets.BuilderConfig(
            name=DiscoEvalConstants.RST,
            version=VERSION,
            description="The RST Discourse Treebank dataset ",
        ),
        datasets.BuilderConfig(
            name=DiscoEvalConstants.PDTB_E,
            version=VERSION,
            description="The Penn Discourse Treebank - Explicit dataset.",
        ),
        datasets.BuilderConfig(
            name=DiscoEvalConstants.PDTB_I,
            version=VERSION,
            description="The Penn Discourse Treebank - Implicit dataset.",
        ),
        datasets.BuilderConfig(
            name=DiscoEvalConstants.SSPABS,
            version=VERSION,
            description="The SSP dataset.",
        ),
        datasets.BuilderConfig(
            name=DiscoEvalConstants.BSOARXIV,
            version=VERSION,
            description="The BSO Task with the arxiv dataset.",
        ),
        datasets.BuilderConfig(
            name=DiscoEvalConstants.BSOWIKI,
            version=VERSION,
            description="The BSO Task with the wiki dataset.",
        ),
        datasets.BuilderConfig(
            name=DiscoEvalConstants.BSOROCSTORY,
            version=VERSION,
            description="The BSO Task with the rocstory dataset.",
        ),
    ]

    def _info(self):
        if self.config.name in [DiscoEvalConstants.SPARXIV, DiscoEvalConstants.SPROCSTORY, DiscoEvalConstants.SPWIKI]:
            features_dict = {
                DiscoEvalConstants.TEXT_COLUMN_NAME[i]: datasets.Value('string')
                for i in range(DiscoEvalConstants.SP_TEXT_COLUMNS)
            }
            features_dict[DiscoEvalConstants.LABEL_NAME] = datasets.ClassLabel(names=DiscoEvalConstants.SP_LABELS)
            features = datasets.Features(features_dict)

        elif self.config.name in [DiscoEvalConstants.BSOARXIV, DiscoEvalConstants.BSOWIKI, DiscoEvalConstants.BSOROCSTORY]:
            features_dict = {
                DiscoEvalConstants.TEXT_COLUMN_NAME[i]: datasets.Value('string')
                for i in range(DiscoEvalConstants.BSO_TEXT_COLUMNS)
            }
            features_dict[DiscoEvalConstants.LABEL_NAME] = datasets.ClassLabel(names=DiscoEvalConstants.BSO_LABELS.values())
            features = datasets.Features(features_dict)

        elif self.config.name in [DiscoEvalConstants.DCCHAT, DiscoEvalConstants.DCWIKI]:
            features_dict = {
                DiscoEvalConstants.TEXT_COLUMN_NAME[i]: datasets.Value('string')
                for i in range(DiscoEvalConstants.DC_TEXT_COLUMNS)
            }
            features_dict[DiscoEvalConstants.LABEL_NAME] = datasets.ClassLabel(names=DiscoEvalConstants.DC_LABELS.values())
            features = datasets.Features(features_dict)

        elif self.config.name in [DiscoEvalConstants.RST]:
            features_dict = {
                DiscoEvalConstants.TEXT_COLUMN_NAME[i]: [datasets.Value('string')]
                for i in range(DiscoEvalConstants.RST_TEXT_COLUMNS)
            }
            features_dict[DiscoEvalConstants.LABEL_NAME] = datasets.ClassLabel(names=DiscoEvalConstants.RST_LABELS)
            features = datasets.Features(features_dict)

        elif self.config.name in [DiscoEvalConstants.PDTB_E]:
            features_dict = {
                DiscoEvalConstants.TEXT_COLUMN_NAME[i]: datasets.Value('string')
                for i in range(DiscoEvalConstants.PDTB_E_TEXT_COLUMNS)
            }
            features_dict[DiscoEvalConstants.LABEL_NAME] = datasets.ClassLabel(names=DiscoEvalConstants.PDTB_E_LABELS)
            features = datasets.Features(features_dict)

        elif self.config.name in [DiscoEvalConstants.PDTB_I]:
            features_dict = {
                DiscoEvalConstants.TEXT_COLUMN_NAME[i]: datasets.Value('string')
                for i in range(DiscoEvalConstants.PDTB_I_TEXT_COLUMNS)
            }
            features_dict[DiscoEvalConstants.LABEL_NAME] = datasets.ClassLabel(names=DiscoEvalConstants.PDTB_I_LABELS)
            features = datasets.Features(features_dict)

        elif self.config.name in [DiscoEvalConstants.SSPABS]:
            features_dict = {
                DiscoEvalConstants.TEXT_COLUMN_NAME[i]: datasets.Value('string')
                for i in range(DiscoEvalConstants.SSPABS_TEXT_COLUMNS)
            }
            features_dict[DiscoEvalConstants.LABEL_NAME] = datasets.ClassLabel(names=DiscoEvalConstants.SSPABS_LABELS.values())
            features = datasets.Features(features_dict)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.name in [DiscoEvalConstants.SPARXIV, DiscoEvalConstants.SPROCSTORY, DiscoEvalConstants.SPWIKI]:
            data_dir = DiscoEvalConstants.SP_DATA_DIR + "/" + DiscoEvalConstants.SP_DIRS[self.config.name]
            train_name = DiscoEvalConstants.SP_TRAIN_NAME
            valid_name = DiscoEvalConstants.SP_VALID_NAME
            test_name = DiscoEvalConstants.SP_TEST_NAME

        elif self.config.name in [DiscoEvalConstants.BSOARXIV, DiscoEvalConstants.BSOWIKI, DiscoEvalConstants.BSOROCSTORY]:
            data_dir = DiscoEvalConstants.BSO_DATA_DIR + "/" + DiscoEvalConstants.BSO_DIRS[self.config.name]
            train_name = DiscoEvalConstants.BSO_TRAIN_NAME
            valid_name = DiscoEvalConstants.BSO_VALID_NAME
            test_name = DiscoEvalConstants.BSO_TEST_NAME

        elif self.config.name in [DiscoEvalConstants.DCCHAT, DiscoEvalConstants.DCWIKI]:
            data_dir = DiscoEvalConstants.DC_DATA_DIR + "/" + DiscoEvalConstants.DC_DIRS[self.config.name]
            train_name = DiscoEvalConstants.DC_TRAIN_NAME
            valid_name = DiscoEvalConstants.DC_VALID_NAME
            test_name = DiscoEvalConstants.DC_TEST_NAME

        elif self.config.name in [DiscoEvalConstants.RST]:
            data_dir = DiscoEvalConstants.RST_DATA_DIR
            train_name = DiscoEvalConstants.RST_TRAIN_NAME
            valid_name = DiscoEvalConstants.RST_VALID_NAME
            test_name = DiscoEvalConstants.RST_TEST_NAME

        elif self.config.name in [DiscoEvalConstants.PDTB_E, DiscoEvalConstants.PDTB_I]:
            data_dir = os.path.join(DiscoEvalConstants.PDTB_DATA_DIR, DiscoEvalConstants.PDTB_DIRS[self.config.name])
            train_name = DiscoEvalConstants.PDTB_TRAIN_NAME
            valid_name = DiscoEvalConstants.PDTB_VALID_NAME
            test_name = DiscoEvalConstants.PDTB_TEST_NAME

        elif self.config.name in [DiscoEvalConstants.SSPABS]:
            data_dir = DiscoEvalConstants.SSPABS_DATA_DIR
            train_name = DiscoEvalConstants.SSPABS_TRAIN_NAME
            valid_name = DiscoEvalConstants.SSPABS_VALID_NAME
            test_name = DiscoEvalConstants.SSPABS_TEST_NAME

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
        if self.config.name == DiscoEvalConstants.RST:
            data = pickle.load(open(filepath, "rb"))
            for key, line in enumerate(data):
                example = {DiscoEvalConstants.TEXT_COLUMN_NAME[i]: sent for i, sent in enumerate(line[1:])}
                example[DiscoEvalConstants.LABEL_NAME] = line[0]
                yield key, example

        else:
            with io.open(filepath, mode='r', encoding='utf-8') as f:
                for key, line in enumerate(f):
                    line = line.strip().split("\t")
                    example = {DiscoEvalConstants.TEXT_COLUMN_NAME[i]: sent for i, sent in enumerate(line[1:])}
                    if self.config.name in (DiscoEvalConstants.PDTB_E, DiscoEvalConstants.PDTB_I):
                        example[DiscoEvalConstants.LABEL_NAME] = line[0]
                    elif self.config.name in (DiscoEvalConstants.DCCHAT, DiscoEvalConstants.DCWIKI):
                        example[DiscoEvalConstants.LABEL_NAME] = DiscoEvalConstants.DC_LABELS[line[0]]
                    elif self.config.name == DiscoEvalConstants.SSPABS:
                        example[DiscoEvalConstants.LABEL_NAME] = DiscoEvalConstants.SSPABS_LABELS[line[0]]
                    elif self.config.name in (DiscoEvalConstants.SPWIKI, DiscoEvalConstants.SPROCSTORY, DiscoEvalConstants.SPARXIV):
                        example[DiscoEvalConstants.LABEL_NAME] = DiscoEvalConstants.SP_LABELS[line[0]]
                    elif self.config.name in (DiscoEvalConstants.BSOARXIV, DiscoEvalConstants.BSOWIKI, DiscoEvalConstants.BSOROCSTORY):
                        example[DiscoEvalConstants.LABEL_NAME] = DiscoEvalConstants.BSO_LABELS[line[0]]
                    yield key, example
