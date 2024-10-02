import json
import uuid
from collections import defaultdict
from hashlib import md5
from typing import Dict, List, Optional

import evaluate
from datasets import Dataset
from evaluate import EvaluationModule

from treetune import logging_utils
from treetune.analyzers import Analyzer

logger = logging_utils.get_logger(__name__)

_bleu_metric: List[EvaluationModule] = [None]


def get_bleu_metric() -> EvaluationModule:
    if _bleu_metric[0] is None:
        _bleu_metric[0] = evaluate.load(
            "bleu",
            experiment_id=uuid.uuid4().hex,
        )
    return _bleu_metric[0]


def set_bleu_metric(bleu_metric: EvaluationModule):
    _bleu_metric[0] = bleu_metric


def avg_bleu_of_pairs_of_sentences(
    sentences: List[str], bleu_metric: Optional[EvaluationModule] = None
) -> float:
    """
    Args:
        sentences: list of direct texts of the children of a node
        bleu_metric: the bleu metric to use
    Returns:
        average bleu score of all pairs of sentences
    """
    assert (
        len(sentences) > 1
    ), "There should be at least 2 sentences to compute bleu score"
    if bleu_metric is None:
        bleu_metric = get_bleu_metric()

    preds = []
    refs = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sen_1 = sentences[i]
            sen_2 = sentences[j]
            preds.append(sen_1)
            refs.append(sen_2)
    bleu_full_stats = bleu_metric.compute(predictions=preds, references=refs)
    bleu = bleu_full_stats["bleu"]
    return bleu


def compute_tree_bleus(tree: Dict) -> List[Dict]:
    """
    Compute the average bleu score of all pairs of sentences of the children of each node
    Args:
        tree: the tree to compute bleu scores
    Returns:
        a list of dict, each dict contains the average bleu score of all pairs of sentences of the children of a node
    """
    node_bleus = []

    def dfs(node, depth):
        if "children" not in node or len(node["children"]) == 0:
            return  # leaf node

        children_texts = [c["text"] for c in node["children"]]
        if len(children_texts) > 1:
            avg_bleu = avg_bleu_of_pairs_of_sentences(children_texts)
            node_bleus.append({"avg_bleu": avg_bleu, "depth": depth})
        for c in node["children"]:
            dfs(c, depth + 1)

    dfs(tree, 0)

    return node_bleus


def compute_tree_blue_stats(tree: Dict) -> Dict[str, float]:
    """
    Compute the average bleu score of all pairs of sentences of the children of each node
    Args:
        tree: the tree to compute bleu scores
    Returns:
        a dict, contains the average bleu score of all pairs of sentences of the children of the root node
    """
    node_bleus = compute_tree_bleus(tree)
    depth_counter = defaultdict(int)
    for d in node_bleus:
        depth_counter[d["depth"]] += 1

    num_nodes = len(node_bleus)
    num_depths = len(depth_counter.keys())
    avg = sum([d["avg_bleu"] for d in node_bleus]) / num_nodes
    avg_depth_normalized = (
        sum([d["avg_bleu"] / depth_counter[d["depth"]] for d in node_bleus])
        / num_depths
    )

    return {
        "avg_tree_bleu": avg,
        "avg_tree_bleu_depth_normalized": avg_depth_normalized,
        "_node_bleus": node_bleus,
    }


@Analyzer.register("tree_diversity")
class TreeDiversityAnalyzer(Analyzer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.result_dir = self.runtime._get_result_dir()
        assert self.result_dir is not None, "Result directory is not set."
        if not self.result_dir.exists():
            raise ValueError(f"Result directory {self.result_dir} does not exist.")

        self.bleu = evaluate.load(
            "bleu",
            experiment_id=md5(
                str(self.get_analysis_root_dir()).encode("utf-8")
            ).hexdigest(),
        )
        set_bleu_metric(self.bleu)

    def get_analysis_id(self) -> str:
        return super().get_analysis_id() + self.result_dir.name + "_tree_diversity"

    def analyze(self):
        super().analyze()

        logger.info(f"Analyzing trees of {self.result_dir}...")

        # Load the mutated dataset
        output_dataset = Dataset.load_from_disk(self.result_dir)

        from treetune.inference_strategies.tree_inference_strategy import TREE_COLNAME

        assert (
            TREE_COLNAME in output_dataset.features
        ), "The dataset does not contain the generated reasoning tress."

        trees = output_dataset[TREE_COLNAME]
        trees = [json.loads(t) for t in trees]
        metrics = self.analyze_trees_diversity(trees)

        self.log_metrics(metrics)

    def analyze_trees_diversity(self, trees):
        tree_stats = []
        for tree in trees:
            tree_stats.append(compute_tree_blue_stats(tree))
        tree_avg_stats = self.avg_stats(tree_stats)
        return tree_avg_stats

    @staticmethod
    def avg_stats(stat_list):
        avg_stats = defaultdict(float)
        stat_keys = stat_list[0].keys()
        for key in stat_keys:
            if key.startswith("_"):
                continue
            avg_stats[key] = sum([s[key] for s in stat_list]) / len(stat_list)
        return avg_stats
