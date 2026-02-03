#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict, OrderedDict
import onnx
from onnx import helper, ModelProto, NodeProto, GraphProto, numpy_helper
from onnx.external_data_helper import load_external_data_for_model

# ------- Config -------
# Prefix pattern: transformer/block_list.<idx>/<kind>/
BLOCK_RE = re.compile(r"^transformer/block_list\.(\d+)/(attn_norm|attention|proj_norm|proj)/")

# Map subgraph "kind" -> custom op_type
OPTYPE_MAP = {
    "attn_norm": "Attn_norm",
    "attention": "Attention",
    "proj_norm": "Proj_norm",
    "proj": "Proj",
}
CUSTOM_DOMAIN = "com.example"  # change if you have a custom domain

def index_graph(graph: GraphProto):
    """Build producer and consumer maps."""
    producer = {}  # value_name -> node
    consumers = defaultdict(list)  # value_name -> [nodes]
    name_to_node = {}

    for node in graph.node:
        name_to_node[node.name] = node
        for outp in node.output:
            if outp:
                producer[outp] = node
        for inp in node.input:
            if inp:
                consumers[inp].append(node)

    graph_inputs = set(i.name for i in graph.input)
    graph_outputs = set(o.name for o in graph.output)
    initializers = set(init.name for init in graph.initializer)

    return producer, consumers, name_to_node, graph_inputs, graph_outputs, initializers

def stable_order(items):
    # Keep deterministic order but stable: use list(items) if already OrderedSet-like; else sort
    return list(items) if isinstance(items, list) else sorted(items)

def collect_regions(graph: GraphProto):
    """Group nodes by (block_idx, kind) if name matches the prefix pattern."""
    regions = defaultdict(list)  # (block_idx, kind) -> [nodes]
    for node in graph.node:
        m = BLOCK_RE.match(node.name)
        if m:
            block_idx, kind = m.group(1), m.group(2)
            regions[(int(block_idx), kind)].append(node)
    return regions

def compute_boundary(graph, region_nodes, producer, consumers, graph_inputs, graph_outputs, initializers):
    region_set = set(region_nodes)
    region_node_names = set(n.name for n in region_nodes)

    # Values produced by nodes in region
    produced_inside = set()
    for n in region_nodes:
        for o in n.output:
            if o:
                produced_inside.add(o)

    # Inputs: any input to region nodes that is not produced by region
    boundary_inputs = OrderedDict()
    for n in region_nodes:
        for inp in n.input:
            if not inp:
                continue
            if inp not in produced_inside:
                # Either comes from outside node, initializer, or graph input
                # Keep first-seen order
                if inp not in boundary_inputs:
                    boundary_inputs[inp] = True

    # Outputs: any output from region that is used by a node outside the region or is a graph output
    boundary_outputs = OrderedDict()
    for n in region_nodes:
        for outp in n.output:
            if not outp:
                continue
            # If it's a graph output, it must be preserved
            if outp in graph_outputs:
                if outp not in boundary_outputs:
                    boundary_outputs[outp] = True
                continue
            # If any consumer is outside region, preserve it
            for c in consumers.get(outp, []):
                if c not in region_set:
                    if outp not in boundary_outputs:
                        boundary_outputs[outp] = True
                    break

    return list(boundary_inputs.keys()), list(boundary_outputs.keys())

def fuse_region(graph: GraphProto, key, nodes_in_region, optype):
    """Replace the nodes in a region by a single custom node."""
    producer, consumers, name_to_node, graph_inputs, graph_outputs, initializers = index_graph(graph)

    inputs, outputs = compute_boundary(
        graph, nodes_in_region, producer, consumers, graph_inputs, graph_outputs, initializers
    )

    # Create custom node
    block_idx, kind = key
    fused_name = f"transformer/block_list.{block_idx}/{kind}__Fused"
    fused_node = helper.make_node(
        optype,
        name=fused_name,
        inputs=inputs,
        outputs=outputs,
        domain=CUSTOM_DOMAIN
    )

    # Rebuild node list: remove region nodes, insert fused node once.
    nodes_to_remove = set(n.name for n in nodes_in_region)

    new_nodes = []
    inserted = False
    # Heuristic: insert the fused node at the position of the first region node seen.
    region_first_seen = None

    for idx, node in enumerate(graph.node):
        if node.name in nodes_to_remove:
            if region_first_seen is None:
                region_first_seen = idx
            continue
        new_nodes.append(node)

    insert_at = region_first_seen if region_first_seen is not None else len(new_nodes)
    new_nodes.insert(insert_at, fused_node)

    # Assign back
    del graph.node[:]
    graph.node.extend(new_nodes)

def prune_unused_initializers(model: ModelProto):
    """Optional: drop initializers that are no longer referenced."""
    graph = model.graph
    # Collect all used value names
    used = set()
    for n in graph.node:
        used.update([i for i in n.input if i])
        used.update([o for o in n.output if o])
    used.update(i.name for i in graph.input)
    used.update(o.name for o in graph.output)

    kept_inits = [init for init in graph.initializer if init.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept_inits)

def main():
    global CUSTOM_DOMAIN
    ap = argparse.ArgumentParser(description="Fuse LLaMA2 subgraphs into custom nodes by name prefix.")
    ap.add_argument("input", help="Path to input .onnx")
    ap.add_argument("output", help="Path to output .onnx")
    ap.add_argument("--domain", default=CUSTOM_DOMAIN, help="Custom op domain (default: com.example)")
    ap.add_argument("--keep-unused-inits", action="store_true", help="Don't prune unused initializers")
    args = ap.parse_args()

    CUSTOM_DOMAIN = args.domain

    model = onnx.load(args.input)
    load_external_data_for_model(model)  # if model uses external data

    regions = collect_regions(model.graph)
    if not regions:
        print("No regions matched the pattern. Check your node names or regex.")
    else:
        # Process each region independently
        # Sort for determinism: by block idx then kind
        for key in sorted(regions.keys(), key=lambda t: (t[0], t[1])):
            block_idx, kind = key
            if kind not in OPTYPE_MAP:
                continue
            nodes = regions[key]
            print(f"Fusing block {block_idx} kind {kind} with {len(nodes)} nodes → {OPTYPE_MAP[kind]}")
            fuse_region(model.graph, key, nodes, OPTYPE_MAP[kind])

    if not args.keep_unused_inits:
        prune_unused_initializers(model)

    # Topologically sort for cleanliness
    model = onnx.utils.polish_model(model)

    onnx.save(model, args.output)
    print(f"Saved fused model to: {args.output}")

if __name__ == "__main__":
    main()
