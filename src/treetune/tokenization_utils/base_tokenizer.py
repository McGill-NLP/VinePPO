from transformers import PreTrainedTokenizerBase

from treetune.common import Registrable


class Tokenizer(PreTrainedTokenizerBase, Registrable):
    pass
