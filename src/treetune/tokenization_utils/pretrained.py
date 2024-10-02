from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from treetune.common import JsonDict
from treetune.tokenization_utils.base_tokenizer import Tokenizer


class DIPreTrainedTokenizer(Tokenizer):
    @classmethod
    def from_di(
        cls, hf_model_name: str, pretrained_args: Optional[JsonDict] = None, **kwargs
    ) -> PreTrainedTokenizerFast:
        if pretrained_args is None:
            pretrained_args = {}

        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name, use_fast=True, **pretrained_args
        )
        return tokenizer


Tokenizer.register("pretrained", constructor="from_di", exist_ok=True)(
    DIPreTrainedTokenizer
)
