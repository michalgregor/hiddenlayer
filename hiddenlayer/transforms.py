"""
HiddenLayer

Transforms that apply to and modify graph nodes.
 
Written by Waleed Abdulla
Licensed under the MIT License
"""

import re
import copy
from .graph import Node
from . import ge



###########################################################################
# Transforms
###########################################################################

class Fold():
    def __init__(self, pattern, op, name=None):
        # TODO: validate that op and name are valid
        self.pattern = ge.GEParser(pattern).parse()
        self.op = op
        self.name = name

    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break

            # Replace pattern with new node
            if self.op == "__first__":
                combo = matches[0]
            elif self.op == "__last__":
                combo = matches[-1]
            else:
                combo = Node(uid=graph.sequence_id(matches),
                                name=self.name or " &gt; ".join([l.title for l in matches]),
                                op=self.op or self.pattern,
                                output_shape=matches[-1].output_shape)
                combo._caption = "/".join(filter(None, [l.caption for l in matches]))
            graph.replace(matches, combo)
        return graph


class FoldId():
    def __init__(self, id_regex, op, name=None):
        # TODO: validate op and name are valid
        self.id_regex = re.compile(id_regex)
        self.op = op
        self.name = name

    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        # Group nodes by the first matching group of the regex
        groups = {}
        for node in graph.nodes.values():
            m = self.id_regex.match(node.id)
            if not m:
                continue
            
            assert m.groups(), "Regular expression must have a matching group to avoid folding unrelated nodes."
            key = m.group(1)
            if key not in groups:
                groups[key] = []
            groups[key].append(node)
            
        # Fold each group of nodes together
        for key, nodes in groups.items():
            # Replace with a new node
            # TODO: Find last node in the sub-graph and get the output shape from it
            combo = Node(uid=key,
                         name=self.name,
                         op=self.op)
            graph.replace(nodes, combo)
        return graph


class Prune():
    def __init__(self, pattern):
        self.pattern = ge.GEParser(pattern).parse()

    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break
            # Remove found nodes
            graph.remove(matches)
        return graph


class PruneBranch():
    def __init__(self, pattern):
        self.pattern = ge.GEParser(pattern).parse()

    def tag(self, node, tag, graph, conditional=False):
        # Return if the node is already tagged
        if hasattr(node, "__tag__") and node.__tag__ == "tag":
            return
        # If conditional, then tag the node if and only if all its
        # outgoing nodes already have the same tag.
        if conditional:
            # Are all outgoing nodes already tagged?
            outgoing = graph.outgoing(node)
            tagged = filter(lambda n: hasattr(n, "__tag__") and n.__tag__ == tag,
                            outgoing)
            if len(list(tagged)) != len(outgoing):
                # Not all outgoing are tagged
                return
        # Tag the node
        node.__tag__ = tag
        # Tag incoming nodes
        for n in graph.incoming(node):
            self.tag(n, tag, graph, conditional=True)

    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break
            # Tag found nodes and their incoming branches
            for n in matches:
                self.tag(n, "delete", graph)
            # Find all tagged nodes and delete them
            tagged = [n for n in graph.nodes.values()
                      if hasattr(n, "__tag__") and n.__tag__ == "delete"]
            graph.remove(tagged)
        return graph

class PruneBranchId():
    def __init__(self, id_regex, direction="incoming"):
        self.id_regex = re.compile(id_regex)
        self.direction = direction

    def tag(self, node, tag, graph, conditional=False):
        # Return if the node is already tagged
        if hasattr(node, "__tag__") and node.__tag__ == "tag":
            return

        if self.direction == "incoming":
            graph_incoming = graph.incoming
            graph_outgoing = graph.outgoing
        elif self.direction == "outgoing":
            graph_outgoing = graph.incoming
            graph_incoming = graph.outgoing
        else:
            raise ValueError("Direction must be one of ('incoming', 'outgoing').")

        # If conditional, then tag the node if and only if all its
        # outgoing nodes already have the same tag.
        if conditional:
            # Are all outgoing nodes already tagged?
            outgoing = graph_outgoing(node)
            tagged = filter(lambda n: hasattr(n, "__tag__") and n.__tag__ == tag,
                            outgoing)
            if len(list(tagged)) != len(outgoing):
                # Not all outgoing are tagged
                return

        # Tag the node
        node.__tag__ = tag
        # Tag incoming nodes
        for n in graph_incoming(node):
            self.tag(n, tag, graph, conditional=True)

    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        while True:
            matches = []
            for node in graph.nodes.values():
                m = self.id_regex.match(node.id)
                if m:
                    matches.append(node)

            if not len(matches):
                break

            # Tag found nodes and their incoming branches
            for n in matches:
                self.tag(n, "delete", graph)
            # Find all tagged nodes and delete them
            tagged = [n for n in graph.nodes.values()
                      if hasattr(n, "__tag__") and n.__tag__ == "delete"]
            graph.remove(tagged)
        return graph

def make_caption_dict(caption, keys=['g', 's']):
    caption_dict = {}
    matches = re.findall("(({})=([1-9]+))".format("|".join(keys)), caption)

    if matches:
        caption_dict = {m[1]: m[2] for m in matches}

    return caption_dict

def update_caption_dict(caption_dict1, caption_dict2):
    caption_dict = caption_dict1.copy()

    for k, v in caption_dict2.items():
        try:
            caption_dict[k] = None if caption_dict[k] != v else v
        except KeyError:
            caption_dict[k] = None

    for k, v in caption_dict1.items():
        if not k in caption_dict2:
            caption_dict[k] = None

    return caption_dict

def update_caption(caption, caption_dict):
    caption = copy.copy(caption)

    for k, v in caption_dict.items():
        caption, n = re.subn(
            "({}=[1-9]+)".format(k),
            "{}={}".format(k, v) if not v is None else " {}".format(k),
            caption
        )

        if not n:
            caption += " {}={}".format(k, v) if not v is None else " {}".format(k)

    return caption

class FoldDuplicates():
    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        matches = True
        while matches:
            for node in graph.nodes.values():
                pattern = ge.SerialPattern([ge.NodePattern(node.op), ge.NodePattern(node.op)])
                matches, _ = pattern.match(graph, node)
                if matches:
                    combo_name = node.name
                    combo_caption = node.caption
                    # replace number x number by MxN if folding layers with
                    # different kernel sizes
                    m_name = re.search("([1-9]+x[1-9]+)", combo_name)
                    caption_dict = make_caption_dict(combo_caption)
                    
                    for rep_node in matches:
                        caption_dict = update_caption_dict(
                            caption_dict,
                            make_caption_dict(rep_node.caption)
                        )

                        if m_name:
                            m_rep = re.search("([1-9]+x[1-9]+)", rep_node.name)
                            if m_rep and m_name.group(1) != m_rep.group(1):
                                combo_name = re.sub("([1-9]+x[1-9]+)", "MxN", combo_name)
                                break

                    combo_caption = update_caption(combo_caption, caption_dict)

                    # Use op and name from the first node, and output_shape from the last
                    combo = Node(uid=graph.sequence_id(matches),
                                name=combo_name,
                                op=node.op,
                                output_shape=matches[-1].output_shape)
                    combo._caption = combo_caption
                    combo.repeat = sum([n.repeat for n in matches])
                    graph.replace(matches, combo)
                    break
        return graph


class Rename():
    def __init__(self, op=None, name=None, to=None):
        assert op or name, "Either op or name must be provided"
        assert not(op and name), "Either op or name should be provided, but not both"
        assert bool(to), "The to parameter is required" 
        self.to = to
        self.op = re.compile(op) if op else None
        self.name = re.compile(name) if name else None
    
    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        for node in graph.nodes.values():
            if self.op:
                node.op = self.op.sub(self.to, node.op)
            # TODO: name is not tested yet
            if self.name:
                node.name = self.name.sub(self.to, node.name)
        return graph


# Transforms to simplify graphs by folding layers that tend to be 
# used together often, such as Conv/BN/Relu.
# These transforms are used AFTER the framework specific transforms
# that map TF and PyTorch graphs to a common representation.
SIMPLICITY_TRANSFORMS = [
    Fold("Conv > Conv > BatchNorm > Relu", "ConvConvBnRelu"),
    Fold("Conv > BatchNorm > Relu", "ConvBnRelu"),
    Fold("Conv > BatchNorm", "ConvBn"),
    Fold("Conv > Relu", "ConvRelu"),
    Fold("Linear > Relu", "LinearRelu"),
    # Fold("ConvBnRelu > MaxPool", "ConvBnReluMaxpool"),
    # Fold("ConvRelu > MaxPool", "ConvReluMaxpool"),
    FoldDuplicates(),
]
