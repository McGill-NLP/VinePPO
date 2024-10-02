import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from datasets import Dataset, DatasetDict

from treetune.common import Params
from treetune.tasks import Task
import logging

logger = logging.getLogger(__name__)

choices = ["A", "B", "C", "D"]

def export_mmlu_from_csv_to_hf_dataset_and_save_in_data(data_dir):
    print(f'Exporting MMLU from {data_dir} to HF dataset')
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(data_dir, "test")) if "_test.csv" in f])
    print(f'Found {len(subjects)} subjects: {subjects}')
    print(subjects)

    all_dev_df = None
    all_test_df = None

    for subject in subjects:
        dev_path = Path(data_dir) / "dev" / f"{subject}_dev.csv"
        test_path = Path(data_dir) / "test" / f"{subject}_test.csv"
        dev_df = pd.read_csv(dev_path, header=None)  # this is used for few shot learning, 5 examples per subject
        test_df = pd.read_csv(test_path, header=None)

        # change  zeroth column name to question
        dev_df = dev_df.rename(columns={0: 'question'})
        test_df = test_df.rename(columns={0: 'question'})
        # change columns 1st to 4th name to choice_{i}
        dev_df = dev_df.rename(columns={i: f'{choices[i - 1]}' for i in range(1, 5)})  # just changing name of columns to A, B, C, D
        test_df = test_df.rename(columns={i: f'{choices[i - 1]}' for i in range(1, 5)})  # just changing name of columns to A, B, C, D
        # change column 5th name to answer
        dev_df = dev_df.rename(columns={5: 'answer'})
        test_df = test_df.rename(columns={5: 'answer'})
        # add a column for the subject
        dev_df['subject'] = subject
        test_df['subject'] = subject

        # add to the all df
        if all_dev_df is None:
            all_dev_df = dev_df
        else:
            all_dev_df = pd.concat([all_dev_df, dev_df])

        if all_test_df is None:
            all_test_df = test_df
        else:
            all_test_df = pd.concat([all_test_df, test_df])

    print(f'all_dev_df: {all_dev_df}')
    print(f'all_test_df: {all_test_df}')
    # save to hf dataset
    dev_dataset = Dataset.from_pandas(all_dev_df)
    test_dataset = Dataset.from_pandas(all_test_df)
    dataset_dict = DatasetDict({
        'dev': dev_dataset,
        'test': test_dataset,
    })

    dataset_dict.save_to_disk(Path(data_dir).parent / "mmlu")

def format_example(example, include_answer=False, include_subject=False):
    prompt = ""
    if include_subject:
        prompt += f"The following is a {example['subject']} question:\n"
    prompt += example['question']
    k = len(choices)
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], example[f'{choices[j]}'])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(example['answer'])
    return prompt

@Task.register("mmlu", exist_ok=True)
class MMLUTask(Task):
    def __init__(self, *args, **kwargs):
        assert kwargs['hf_num_proc'] == 1, 'for now, mmlu task only supports num_procs=1, idk why, but it runs to a broken pipe on my end with 4'
        super().__init__(*args, **kwargs)
        logger.info('# ----- mmlu benchmark, not using few shots ----- #')

    def evaluate_predictions(self, predictions, references):
        assert len(predictions) == len(references)
        logger.info('evaluating mmlu predictions')
        logger.info(f'len predictions: {len(predictions)}')
        logger.info(f'first 20 predictions: {predictions[:20]}')
        logger.info(f'first 20 references: {references[:20]}')

        # TODO make it per record per subject, not for now, should figure out the references and how to grab the subjects

        correct_count = 0
        no_answer_count = 0
        total_count = len(predictions)
        for pred, ref in zip(predictions, references):
            assert len(pred) == 1, 'just a single answer is expected'
            pred_answer = pred[0]
            assert pred_answer in ['A', 'B', 'C', 'D', 'no-answer'], f'pred_answer: {pred_answer}'
            gold_answer = ref['answer']
            assert gold_answer in ['A', 'B', 'C', 'D'], f'gold_answer: {gold_answer}'

            if pred_answer == gold_answer:
                correct_count += 1
            elif pred_answer == 'no-answer':
                no_answer_count += 1

        acc = correct_count / total_count
        no_answer_frac = no_answer_count / total_count
        logger.info(f'correct_count: {correct_count}')
        logger.info(f'no_answer_count: {no_answer_count}')

        return {
            'acc': acc,
            'no_answer_frac': no_answer_frac,
        }

    def get_datasets(self, split):
        assert split == 'test', 'mmlu task: only test split is supported, loading hf_test dataset'
        return super().get_datasets(split)

    def build_dataset(
        self,
    ) -> DatasetDict:
        datasets = super().build_dataset()
        datasets = datasets.map(
            self._preprocess_example, num_proc=1, desc="Preprocessing examples"
        )
        return datasets

    def _preprocess_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        question_with_choices_and_subject = format_example(example, include_answer=False, include_subject=True)
        example.update({'question_with_choices_and_subject': question_with_choices_and_subject})
        return example


if __name__ == '__main__':
    # run to generate the mmlu huggingface dataset given that you downloaded the mmlu data from the original github repo https://github.com/hendrycks/test/tree/master
    # the download link as of now is https://people.eecs.berkeley.edu/~hendrycks/data.tar
    export_mmlu_from_csv_to_hf_dataset_and_save_in_data(data_dir='../../../data/mmlu-data')
    task = MMLUTask.from_params(Params({'load_dataset_dict': True,
                                        'dataset_dict_path': '../../../data/mmlu',
                                        'hf_num_proc': 1,
                                        }))
    task.get_datasets('test')
