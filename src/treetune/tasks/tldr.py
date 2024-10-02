import random
from typing import Dict, Any, Optional

from datasets import DatasetDict

from treetune.tasks.base_task import Task
from treetune.tokenization_utils import Tokenizer


@Task.register("tldr_preprocessed", exist_ok=True)
class TLDR(Task):
    pass
