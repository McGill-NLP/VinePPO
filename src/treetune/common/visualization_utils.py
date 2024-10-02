# color functions
from treetune.common import JsonDict


def visualize_tree(tree: JsonDict):
    import graphviz
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    MIN_SCORE = 0
    MAX_SCORE = 10
    CMAP = plt.get_cmap("Greens")

    def normalize_score(score, min_score, max_score):
        return (score - min_score) / (max_score - min_score)

    def score_to_color(score, cmap, min_score, max_score):
        score = float(score)
        normalized_score = normalize_score(
            score, min_score=min_score, max_score=max_score
        )
        rgba_color = cmap(normalized_score)
        hex_color = colors.rgb2hex(rgba_color)  # convert RGBA to Hex color
        return hex_color

    def dfs(v, g, counter):
        if len(v) == 0:
            return
        text = v["text"]
        score = v.get("score", 0)
        color = score_to_color(score, CMAP, MIN_SCORE, MAX_SCORE)
        elaborate = v.get("elaborate", "no-elaboration")
        node_name = str(counter)
        #     g.node(node_name, label=f'{text}\nscore:{score}\nelaborate:{elaborate}', tooltip_text=text, shape='rectangle', fillcolor=color, style='filled')
        g.node(
            node_name,
            label=f"{text}",
            tooltip_text=text,
            shape="rectangle",
            fillcolor=color,
            style="filled",
        )
        if "children" in v:
            for child in v["children"]:
                counter += 1
                child_name = str(counter)
                if len(child) == 0 or len(child["text"].strip()) == 0:
                    continue

                counter = dfs(child, g, counter)
                g.edge(node_name, child_name)
        if "answer" in v:
            color = "orange"
            g.node(
                f"{counter}_answer",
                label=v["answer"],
                tooltip_text=v["answer"],
                shape="circle",
                fillcolor=color,
                style="filled",
            )
            g.edge(node_name, f"{counter}_answer")
        return counter

    g = graphviz.Digraph("G", format="svg")
    dfs(tree, g, 0)

    return g
