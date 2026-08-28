#!/usr/bin/env python3
"""Analyze and visualize cluster assignments in scheduled AccelGen MLIR."""

from __future__ import annotations

import argparse
import colorsys
import csv
import html
import math
import re
import struct
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from statistics import mean
from typing import Iterable


EXPERIMENT_RE = re.compile(
    r"^(?P<prefix>(?P<model>.+)-block(?P<block>\d+)-"
    r"(?P<layer>attention|ffn)-(?P<action>prefill|decode)-"
    r"b(?P<batch>\d+)s(?P<length>\d+))-generic-scheduled-"
    r"(?P<config>[^-]+)-(?P<max_operation>\d+)\.mlir$"
)
SSA_RE = re.compile(r"%[A-Za-z0-9_.$-]+(?:#[0-9]+)?")
RESULT_RE = re.compile(
    r"^\s*(?P<lhs>%[A-Za-z0-9_.$-]+(?::\d+)?)\s*=\s*"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_.]*)\b"
)
FUNC_RE = re.compile(r"^\s*func\.func\s+@(?P<name>[A-Za-z0-9_.$-]+)")
AFFINE_RE = re.compile(r"^\s*(#[A-Za-z0-9_.$-]+)\s*=\s*(affine_map<.*>)\s*$")


@dataclass
class GenericNode:
    index: int
    fingerprint: str
    label: str
    predecessors: set[int] = field(default_factory=set)


@dataclass
class ParsedFunction:
    name: str
    nodes: list[GenericNode]
    attributes: dict[str, float]


@dataclass
class Experiment:
    model: str
    block: int
    layer: str
    action: str
    batch: int
    length: int
    config: str
    source_path: Path
    clustered_path: Path
    source_nodes: list[GenericNode]
    edges: set[tuple[int, int]]
    memberships: dict[int, int]
    cluster_names: dict[int, str]
    cluster_metrics: dict[int, dict[str, float]]
    cluster_sizes: dict[int, int]
    ambiguous_matches: int
    excluded_source_ops: int
    unmatched_source_ops: int
    unmatched_cluster_ops: int

    @property
    def cluster_count(self) -> int:
        return len(self.cluster_names)

    @property
    def clusterable_edges(self) -> set[tuple[int, int]]:
        return {
            (src, dst)
            for src, dst in self.edges
            if src in self.memberships and dst in self.memberships
        }

    @property
    def intercluster_edges(self) -> int:
        return sum(
            self.memberships[src] != self.memberships[dst]
            for src, dst in self.clusterable_edges
        )


def brace_delta(line: str) -> int:
    return line.count("{") - line.count("}")


def affine_maps(text: str) -> dict[str, str]:
    maps: dict[str, str] = {}
    for line in text.splitlines():
        match = AFFINE_RE.match(line)
        if match:
            maps[match.group(1)] = match.group(2)
    return maps


def function_blocks(text: str) -> list[tuple[str, str, str]]:
    lines = text.splitlines()
    blocks: list[tuple[str, str, str]] = []
    index = 0
    while index < len(lines):
        match = FUNC_RE.match(lines[index])
        if not match:
            index += 1
            continue
        start = index
        depth = 0
        opened = False
        while index < len(lines):
            depth += brace_delta(lines[index])
            opened = opened or "{" in lines[index]
            index += 1
            if opened and depth == 0:
                break
        block = "\n".join(lines[start:index])
        blocks.append((match.group("name"), lines[start], block))
    return blocks


def semantic_attrs(header: str) -> tuple[str, ...]:
    attrs = re.search(r"\battrs\s*=\s*\{([^{}]*)\}", header)
    if not attrs:
        return ()
    return tuple(
        sorted(
            re.findall(
                r"accelgen\.[A-Za-z0-9_.-]+\s*=\s*[^,}]+", attrs.group(1)
            )
        )
    )


def normalize_generic(raw: str, maps: dict[str, str]) -> str:
    lines = raw.splitlines()
    header = lines[0]
    attrs = semantic_attrs(header)
    header = re.sub(r"\battrs\s*=\s*\{[^{}]*\}", "", header)
    raw = "\n".join([header, *lines[1:]])
    raw = re.sub(
        r"#[A-Za-z0-9_.$-]+",
        lambda match: maps.get(match.group(0), match.group(0)),
        raw,
    )
    raw = SSA_RE.sub("%v", raw)
    raw = re.sub(r"^\s*%v(?::\d+)?\s*=\s*", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw + "|attrs=" + ",".join(attrs)


def generic_label(raw: str) -> str:
    header = raw.splitlines()[0]
    attrs = " ".join(semantic_attrs(header))
    scalar_ops = re.findall(
        r"=\s*((?:arith|math|linalg)\.[A-Za-z0-9_]+)", "\n".join(raw.splitlines()[1:])
    )
    names = [name.split(".", 1)[1] for name in scalar_ops if name != "linalg.yield"]
    iterator_match = re.search(r'iterator_types\s*=\s*\[([^]]*)\]', header)
    has_reduction = bool(iterator_match and "reduction" in iterator_match.group(1))
    if "mulf" in names and "addf" in names and has_reduction:
        return "matmul"
    if "accelgen.transpose" in attrs:
        return "transpose"
    if "accelgen.const_broadcast" in attrs:
        return "broadcast"
    if "accelgen.expand" in attrs:
        return "expand"
    if "accelgen.memory_transformation" in attrs:
        return "layout"
    if "maximumf" in names and has_reduction:
        return "max-reduce"
    if names == ["addf"] and has_reduction:
        return "sum-reduce"
    if {"negf", "exp", "addf", "divf"}.issubset(names):
        return "sigmoid"
    aliases = {
        "mulf": "mul",
        "addf": "add",
        "subf": "sub",
        "divf": "div",
        "maximumf": "max",
        "truncf": "trunc",
        "extf": "extend",
    }
    labels = [aliases.get(name, name) for name in names]
    return "+".join(dict.fromkeys(labels)) or "generic"


def result_names(lhs: str) -> list[str]:
    if ":" not in lhs:
        return [lhs]
    base, count_text = lhs.rsplit(":", 1)
    count = int(count_text)
    return [f"{base}#{index}" for index in range(count)]


def parse_attributes(header: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in ("cycles", "externalAccess", "flops", "sramAccess"):
        match = re.search(rf"\b{key}\s*=\s*([^ ,}}]+)", header)
        if not match:
            continue
        value = match.group(1)
        if value.startswith("0x") and len(value) == 18:
            result[key] = struct.unpack(">d", bytes.fromhex(value[2:]))[0]
        else:
            result[key] = float(value)
    return result


def parse_function(name: str, header: str, block: str, maps: dict[str, str]) -> ParsedFunction:
    lines = block.splitlines()
    definitions: dict[str, set[int]] = {}
    nodes: list[GenericNode] = []
    depth = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        before = depth
        depth += brace_delta(line)
        match = RESULT_RE.match(line) if before == 1 else None
        if not match:
            index += 1
            continue

        lhs = match.group("lhs")
        op_name = match.group("name")
        raw_lines = [line]
        if op_name == "linalg.generic":
            generic_depth = depth
            index += 1
            while index < len(lines) and generic_depth > 1:
                raw_lines.append(lines[index])
                generic_depth += brace_delta(lines[index])
                depth += brace_delta(lines[index])
                index += 1
            raw = "\n".join(raw_lines)
            uses = set(SSA_RE.findall(raw_lines[0]))
            predecessors = set().union(*(definitions.get(use, set()) for use in uses))
            node_index = len(nodes)
            nodes.append(
                GenericNode(
                    index=node_index,
                    fingerprint=normalize_generic(raw, maps),
                    label=generic_label(raw),
                    predecessors=predecessors,
                )
            )
            ancestry = {node_index}
        else:
            uses = set(SSA_RE.findall(line[match.end() :]))
            ancestry = set().union(*(definitions.get(use, set()) for use in uses))
            index += 1
        for result in result_names(lhs):
            definitions[result] = set(ancestry)
        continue
    return ParsedFunction(name=name, nodes=nodes, attributes=parse_attributes(header))


def parse_mlir(path: Path) -> list[ParsedFunction]:
    text = path.read_text(encoding="utf-8")
    maps = affine_maps(text)
    return [parse_function(name, header, block, maps) for name, header, block in function_blocks(text)]


def source_edges(nodes: Iterable[GenericNode]) -> set[tuple[int, int]]:
    return {(pred, node.index) for node in nodes for pred in node.predecessors}


def match_clusters(
    source_nodes: list[GenericNode], cluster_functions: list[ParsedFunction]
) -> tuple[dict[int, int], dict[int, int], int, int, int, int]:
    source_by_fingerprint: dict[str, list[int]] = defaultdict(list)
    scheduled_by_fingerprint: dict[str, list[tuple[int, int]]] = defaultdict(list)
    eligible_source_nodes = [node for node in source_nodes if node.label != "broadcast"]
    for node in eligible_source_nodes:
        source_by_fingerprint[node.fingerprint].append(node.index)
    for cluster_index, function in enumerate(cluster_functions):
        for node in function.nodes:
            scheduled_by_fingerprint[node.fingerprint].append((cluster_index, node.index))

    source_dag_edges = source_edges(source_nodes)
    scheduled_edges = {
        ((cluster_index, pred), (cluster_index, node.index))
        for cluster_index, function in enumerate(cluster_functions)
        for node in function.nodes
        for pred in node.predecessors
    }
    assignment: dict[tuple[int, int], int] = {}
    groups: list[tuple[list[tuple[int, int]], list[int]]] = []
    unmatched_cluster = 0
    unmatched_source = 0
    all_fingerprints = set(source_by_fingerprint) | set(scheduled_by_fingerprint)
    for fingerprint in all_fingerprints:
        source_ids = source_by_fingerprint.get(fingerprint, [])
        scheduled_ids = scheduled_by_fingerprint.get(fingerprint, [])
        common = min(len(source_ids), len(scheduled_ids))
        unmatched_cluster += len(scheduled_ids) - common
        unmatched_source += len(source_ids) - common
        if common == 1:
            assignment[scheduled_ids[0]] = source_ids[0]
        elif common > 1:
            groups.append((scheduled_ids[:common], source_ids[:common]))

    def compatible(candidate_assignment: dict[tuple[int, int], int]) -> bool:
        items = list(candidate_assignment.items())
        for index, (left_ref, left_source) in enumerate(items):
            for right_ref, right_source in items[index + 1 :]:
                if left_ref[0] != right_ref[0]:
                    continue
                for scheduled_edge, source_edge in (
                    ((left_ref, right_ref), (left_source, right_source)),
                    ((right_ref, left_ref), (right_source, left_source)),
                ):
                    if (scheduled_edge in scheduled_edges) != (source_edge in source_dag_edges):
                        return False
        return True

    groups.sort(key=lambda group: math.factorial(len(group[0])))
    solutions: list[dict[tuple[int, int], int]] = []

    def solve(group_index: int, current: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= 2:
            return
        if group_index == len(groups):
            solutions.append(dict(current))
            return
        scheduled_ids, source_ids = groups[group_index]
        for ordered_sources in permutations(source_ids):
            proposed = dict(current)
            proposed.update(zip(scheduled_ids, ordered_sources))
            if compatible(proposed):
                solve(group_index + 1, proposed)

    solve(0, assignment)
    if solutions:
        assignment = solutions[0]
    else:
        # Preserve useful aggregate output if malformed input defeats graph matching.
        for scheduled_ids, source_ids in groups:
            assignment.update(zip(scheduled_ids, source_ids))
    ambiguous = sum(len(scheduled_ids) for scheduled_ids, _ in groups) if len(solutions) > 1 else 0

    memberships: dict[int, int] = {}
    cluster_sizes: dict[int, int] = defaultdict(int)
    for (cluster_index, _), source_id in assignment.items():
        memberships[source_id] = cluster_index
        cluster_sizes[cluster_index] += 1
    excluded_source = len(source_nodes) - len(eligible_source_nodes)
    return (
        memberships,
        dict(cluster_sizes),
        ambiguous,
        excluded_source,
        unmatched_source,
        unmatched_cluster,
    )


def discover_experiments(args: argparse.Namespace) -> list[Experiment]:
    experiments: list[Experiment] = []
    for clustered_path in sorted(args.clustered_dir.glob("*.mlir")):
        match = EXPERIMENT_RE.match(clustered_path.name)
        if not match or match.group("model") != args.model:
            continue
        if int(match.group("block")) != args.block:
            continue
        if int(match.group("max_operation")) != args.max_operation:
            continue
        if int(match.group("length")) not in args.lengths:
            continue
        source_path = args.generic_dir / f"{match.group('prefix')}-generic.mlir"
        if not source_path.exists():
            raise FileNotFoundError(f"missing source MLIR for {clustered_path}: {source_path}")
        source_functions = parse_mlir(source_path)
        main = next((function for function in source_functions if function.name == "main"), None)
        if main is None:
            raise ValueError(f"no @main function in {source_path}")
        cluster_functions = [
            function for function in parse_mlir(clustered_path) if function.name.startswith("Cluster_")
        ]
        (
            memberships,
            cluster_sizes,
            ambiguous,
            excluded_source,
            unmatched_source,
            unmatched_cluster,
        ) = match_clusters(main.nodes, cluster_functions)
        experiments.append(
            Experiment(
                model=match.group("model"),
                block=int(match.group("block")),
                layer=match.group("layer"),
                action=match.group("action"),
                batch=int(match.group("batch")),
                length=int(match.group("length")),
                config=match.group("config"),
                source_path=source_path,
                clustered_path=clustered_path,
                source_nodes=main.nodes,
                edges=source_edges(main.nodes),
                memberships=memberships,
                cluster_names={index: function.name for index, function in enumerate(cluster_functions)},
                cluster_metrics={index: function.attributes for index, function in enumerate(cluster_functions)},
                cluster_sizes=cluster_sizes,
                ambiguous_matches=ambiguous,
                excluded_source_ops=excluded_source,
                unmatched_source_ops=unmatched_source,
                unmatched_cluster_ops=unmatched_cluster,
            )
        )
    return experiments


def canonical_clusters(experiment: Experiment) -> tuple[dict[int, int], dict[int, int]]:
    members: dict[int, list[int]] = defaultdict(list)
    for node, cluster in experiment.memberships.items():
        members[cluster].append(node)
    ordered = sorted(members, key=lambda cluster: (min(members[cluster]), cluster))
    canonical = {cluster: index for index, cluster in enumerate(ordered)}
    anchors = {cluster: min(nodes) for cluster, nodes in members.items()}
    return canonical, anchors


def cluster_color(anchor: int, node_count: int) -> str:
    fraction = anchor / max(1, node_count - 1)
    hue = round(15 + 315 * fraction)
    red, green, blue = colorsys.hls_to_rgb(hue / 360, 0.48, 0.62)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def graph_closure(nodes: list[GenericNode]) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    successors: dict[int, set[int]] = {node.index: set() for node in nodes}
    for node in nodes:
        for predecessor in node.predecessors:
            successors[predecessor].add(node.index)
    ancestors: dict[int, set[int]] = {}
    descendants: dict[int, set[int]] = {}
    for node in nodes:
        ancestors[node.index] = set(node.predecessors)
        for predecessor in node.predecessors:
            ancestors[node.index].update(ancestors.get(predecessor, set()))
    for node in reversed(nodes):
        descendants[node.index] = set(successors[node.index])
        for successor in successors[node.index]:
            descendants[node.index].update(descendants.get(successor, set()))
    return ancestors, descendants


def semantic_stage_map(experiment: Experiment) -> dict[int, str]:
    nodes = experiment.source_nodes
    ancestors, descendants = graph_closure(nodes)
    matmuls = [node.index for node in nodes if node.label == "matmul"]
    stages: dict[int, str] = {}

    if experiment.layer == "ffn":
        if len(matmuls) != 3:
            raise ValueError(f"expected three FFN projections in {experiment.source_path}")
        gate, up, down = matmuls
        sigmoid = next(node.index for node in nodes if node.label == "sigmoid")
        products = [node.index for node in nodes if node.label == "mul"]
        combine = next(node for node in products if gate in ancestors[node] and up in ancestors[node])
        silu_product = next(node for node in products if sigmoid in ancestors[node])
        stages.update({gate: "gate_proj", up: "up_proj", down: "down_proj"})
        stages.update({sigmoid: "silu", silu_product: "silu", combine: "product"})
        for node in nodes:
            if node.index in stages or node.label == "broadcast":
                continue
            projection_inputs = {
                anchor for anchor in (gate, up) if node.index in ancestors[anchor]
            }
            if projection_inputs == {gate, up}:
                stages[node.index] = "input"
            elif projection_inputs == {gate}:
                stages[node.index] = "gate_proj"
            elif projection_inputs == {up}:
                stages[node.index] = "up_proj"
            elif combine in ancestors[node.index] and down in descendants[node.index]:
                stages[node.index] = "down_proj"
            elif node.index in ancestors[down]:
                stages[node.index] = "down_proj"
            else:
                stages[node.index] = "product"
        return stages

    if len(matmuls) != 6:
        raise ValueError(f"expected six attention matmuls in {experiment.source_path}")
    q_proj, k_proj, v_proj, qk, av, o_proj = matmuls
    stages.update(
        {
            q_proj: "q_proj",
            k_proj: "k_proj",
            v_proj: "v_proj",
            qk: "qk",
            av: "av",
            o_proj: "o_proj",
        }
    )
    k_path = descendants[k_proj] & ancestors[qk]
    k_rope_end = max(
        (node for node in k_path if nodes[node].label == "add"), default=k_proj
    )
    for node in nodes:
        index = node.index
        if index in stages or node.label == "broadcast":
            continue
        if index in descendants[q_proj] and index in ancestors[qk]:
            stages[index] = "q_rope"
        elif index in k_path:
            if index not in descendants[k_rope_end]:
                stages[index] = "k_rope"
            elif node.label == "expand":
                stages[index] = "k_repeat"
            else:
                stages[index] = "k_layout"
        elif index in descendants[v_proj] and index in ancestors[av]:
            v_path = descendants[v_proj] & ancestors[av]
            v_repeat = next(
                (candidate for candidate in v_path if nodes[candidate].label == "expand"),
                None,
            )
            if node.label == "expand":
                stages[index] = "v_repeat"
            elif v_repeat is not None and index in ancestors[v_repeat]:
                stages[index] = "v_head_layout"
            else:
                stages[index] = "v_matmul_layout"
        elif index in descendants[qk] and index in ancestors[av]:
            softmax_stages = {
                "trunc+mul": "scale",
                "add": "mask",
                "extend": "fp32_extend",
                "max-reduce": "max_reduce",
                "sub": "subtract",
                "exp": "exp",
                "sum-reduce": "sum_reduce",
                "div": "divide",
                "trunc": "bf16_trunc",
                "layout": "softmax_layout",
            }
            stages[index] = softmax_stages.get(node.label, "softmax_layout")
        elif index in descendants[av] and index in ancestors[o_proj]:
            stages[index] = "o_proj"
        else:
            projection_inputs = {
                anchor for anchor in (q_proj, k_proj, v_proj) if index in ancestors[anchor]
            }
            if projection_inputs == {q_proj, k_proj, v_proj}:
                stages[index] = "input"
            elif projection_inputs == {q_proj}:
                stages[index] = "q_proj"
            elif projection_inputs == {k_proj}:
                stages[index] = "k_proj"
            elif projection_inputs == {v_proj}:
                stages[index] = "v_proj"
            elif index in ancestors[o_proj]:
                stages[index] = "o_proj"
            else:
                stages[index] = "softmax_layout"
    return stages


SEMANTIC_LAYOUTS = {
    "attention": {
        "stages": (
            "input", "q_proj", "q_rope", "k_proj", "k_rope", "k_repeat",
            "k_layout", "v_proj", "v_head_layout", "v_repeat",
            "v_matmul_layout", "qk", "scale", "mask", "fp32_extend",
            "max_reduce", "subtract", "exp", "sum_reduce", "divide",
            "bf16_trunc", "softmax_layout", "av", "o_proj", "output",
        ),
        "labels": {
            "input": "Hidden input", "q_proj": "Q projection", "q_rope": "Q RoPE",
            "k_proj": "K projection", "k_rope": "K RoPE", "k_repeat": "K repeat",
            "k_layout": "K layout", "v_proj": "V projection",
            "v_head_layout": "V head layout", "v_repeat": "V repeat",
            "v_matmul_layout": "V matmul layout", "qk": "Q x K^T",
            "scale": "Scale", "mask": "Add mask", "fp32_extend": "FP32 extend",
            "max_reduce": "Max reduce", "subtract": "Subtract max", "exp": "Exp",
            "sum_reduce": "Sum reduce", "divide": "Divide", "bf16_trunc": "BF16 trunc",
            "softmax_layout": "Output layout", "av": "Attention x V",
            "o_proj": "O projection", "output": "Output",
        },
    },
    "ffn": {
        "stages": (
            "input", "gate_proj", "silu", "up_proj", "product",
            "down_proj", "output",
        ),
        "labels": {
            "input": "Hidden input", "gate_proj": "Gate projection", "silu": "SiLU",
            "up_proj": "Up projection", "product": "Elementwise x",
            "down_proj": "Down projection", "output": "Output",
        },
    },
}


def partition_signature(experiment: Experiment) -> tuple:
    members: dict[int, list[int]] = defaultdict(list)
    for node, cluster in experiment.memberships.items():
        members[cluster].append(node)
    partition = tuple(sorted(tuple(sorted(nodes)) for nodes in members.values()))
    labels = tuple(node.label for node in experiment.source_nodes)
    return labels, tuple(sorted(experiment.edges)), partition


def partition_groups(experiments: list[Experiment], layer: str) -> list[list[Experiment]]:
    grouped: dict[tuple, list[Experiment]] = defaultdict(list)
    for experiment in experiments:
        if experiment.layer == layer:
            grouped[partition_signature(experiment)].append(experiment)
    groups = list(grouped.values())
    groups.sort(
        key=lambda group: (
            ("prefill", "decode").index(group[0].action),
            ("edge", "server").index(group[0].config),
            min(experiment.length for experiment in group),
        )
    )
    return groups


def condition_lines(group: list[Experiment]) -> list[str]:
    conditions: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for experiment in group:
        conditions[(experiment.config, experiment.action, experiment.batch)].append(experiment.length)
    lines = []
    for (config, action, batch), lengths in sorted(conditions.items()):
        lengths = sorted(lengths)
        length_text = "L=" + ", ".join(map(str, lengths))
        lines.append(f"{config} / {action} / B={batch} / {length_text}")
    return lines


def semantic_badges(experiment: Experiment) -> dict[str, list[tuple[int, int, str]]]:
    stages = semantic_stage_map(experiment)
    canonical, anchors = canonical_clusters(experiment)
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for node, cluster in experiment.memberships.items():
        counts[(stages[node], cluster)] += 1
    result: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for (stage, cluster), count in counts.items():
        result[stage].append(
            (canonical[cluster], count, cluster_color(anchors[cluster], len(experiment.source_nodes)))
        )
    for stage in result:
        result[stage].sort()
    return result


def draw_semantic_topology(
    parts: list[str], layer: str, x: float, y: float, scale: float,
    badges: dict[str, list[tuple[int, int, str]]] | None = None,
) -> None:
    layout = SEMANTIC_LAYOUTS[layer]
    positions = layout["positions"]
    box_width = 104 * scale
    box_height = 58 * scale
    active_stages = set(positions) if badges is None else set(badges) | {"input", "output"}
    successors: dict[str, list[str]] = defaultdict(list)
    for source, target in layout["edges"]:
        successors[source].append(target)
    visible_edges: set[tuple[str, str]] = set()
    for source in active_stages:
        pending = list(successors[source])
        visited: set[str] = set()
        while pending:
            target = pending.pop()
            if target in visited:
                continue
            visited.add(target)
            if target in active_stages:
                visible_edges.add((source, target))
            else:
                pending.extend(successors[target])
    for source, target in sorted(visible_edges):
        sx, sy = positions[source]
        tx, ty = positions[target]
        start_x = x + (sx + 104) * scale
        start_y = y + (sy + 29) * scale
        end_x = x + tx * scale
        end_y = y + (ty + 29) * scale
        middle_x = (start_x + end_x) / 2
        path = f"M {start_x:.1f} {start_y:.1f} L {middle_x:.1f} {start_y:.1f} L {middle_x:.1f} {end_y:.1f} L {end_x:.1f} {end_y:.1f}"
        parts.append(f'<path class="topology-edge" d="{path}" marker-end="url(#arrow)"/>')
    for stage, (stage_x, stage_y) in positions.items():
        if stage not in active_stages:
            continue
        px = x + stage_x * scale
        py = y + stage_y * scale
        stage_badges = badges.get(stage, []) if badges else []
        fill = "#f7f9fb" if stage not in {"input", "output"} else "#eef2f5"
        parts.append(
            f'<rect class="stage" x="{px:.1f}" y="{py:.1f}" width="{box_width:.1f}" height="{box_height:.1f}" fill="{fill}"/>'
        )
        label = html.escape(layout["labels"][stage])
        parts.append(
            f'<text class="stage-label" x="{px + box_width / 2:.1f}" y="{py + 17 * scale:.1f}" text-anchor="middle">{label}</text>'
        )
        for badge_index, (cluster, count, color) in enumerate(stage_badges):
            columns = min(5, max(1, len(stage_badges)))
            badge_x = px + box_width / 2 + (badge_index % columns - (columns - 1) / 2) * 20 * scale
            badge_y = py + (38 + 18 * (badge_index // columns)) * scale
            radius = 9 * scale
            title = html.escape(f"C{cluster}: {count} generic operation{'s' if count != 1 else ''} in {layout['labels'][stage]}")
            parts.append(
                f'<circle class="badge" cx="{badge_x:.1f}" cy="{badge_y:.1f}" r="{radius:.1f}" fill="{color}"><title>{title}</title></circle>'
            )
            parts.append(
                f'<text class="badge-label" x="{badge_x:.1f}" y="{badge_y + 2.5 * scale:.1f}" text-anchor="middle">C{cluster}</text>'
            )


def svg_semantic_partitions(experiments: list[Experiment], output_path: Path) -> None:
    attention_groups = partition_groups(experiments, "attention")
    ffn_groups = partition_groups(experiments, "ffn")
    width = 2480
    margin = 36
    reference_y = 118
    attention_top = 475
    panel_width = 1180
    attention_panel_height = 278
    ffn_top = attention_top + math.ceil(len(attention_groups) / 2) * attention_panel_height + 95
    ffn_panel_height = 262
    height = ffn_top + math.ceil(len(ffn_groups) / 2) * ffn_panel_height + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#8a9199"/></marker></defs>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124;letter-spacing:0}.title{font-size:25px;font-weight:700}.subtitle{font-size:13px;fill:#5f6368}.section{font-size:19px;font-weight:700}.panel-title{font-size:13px;font-weight:700}.condition{font-size:11px;fill:#4f565d}.stage{stroke:#8a9199;stroke-width:1.2;rx:5}.stage-label{font-size:10px;font-weight:600}.topology-edge{fill:none;stroke:#8a9199;stroke-width:1.2}.badge{stroke:#fff;stroke-width:1}.badge-label{font-size:6px;font-weight:700;fill:#fff}.panel{fill:#fff;stroke:#d5d9de;stroke-width:1;rx:6}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{margin}" y="36">{html.escape(experiments[0].model)} semantic cluster partitions</text>',
        f'<text class="subtitle" x="{margin}" y="60">Each C# circle is one scheduled cluster. Repeated C# circles show a cluster spanning multiple semantic stages.</text>',
        f'<text class="subtitle" x="{margin}" y="80">Identical low-level partitions are drawn once and labeled with every configuration that produces them.</text>',
        f'<text class="section" x="{margin}" y="108">Model topology</text>',
    ]
    draw_semantic_topology(parts, "attention", margin, reference_y, 1.0)
    draw_semantic_topology(parts, "ffn", 1640, reference_y + 12, 0.9)

    attention_runs = sum(len(group) for group in attention_groups)
    ffn_runs = sum(len(group) for group in ffn_groups)
    parts.append(f'<text class="section" x="{margin}" y="{attention_top - 20}">Attention: {len(attention_groups)} unique partitions from {attention_runs} runs</text>')
    for index, group in enumerate(attention_groups):
        col = index % 2
        row = index // 2
        x = margin + col * (panel_width + 24)
        y = attention_top + row * attention_panel_height
        representative = group[0]
        lines = condition_lines(group)
        parts.append(f'<rect class="panel" x="{x}" y="{y}" width="{panel_width}" height="{attention_panel_height - 14}"/>')
        parts.append(f'<text class="panel-title" x="{x + 16}" y="{y + 23}">A{index + 1} · {representative.cluster_count} clusters · {representative.intercluster_edges}/{len(representative.clusterable_edges)} cut edges</text>')
        for line_index, line in enumerate(lines):
            parts.append(f'<text class="condition" x="{x + 16}" y="{y + 42 + line_index * 14}">{html.escape(line)}</text>')
        draw_semantic_topology(parts, "attention", x + 54, y + 56, 0.72, semantic_badges(representative))

    parts.append(f'<text class="section" x="{margin}" y="{ffn_top - 20}">FFN: {len(ffn_groups)} unique partitions from {ffn_runs} runs</text>')
    for index, group in enumerate(ffn_groups):
        col = index % 2
        row = index // 2
        x = margin + col * (panel_width + 24)
        y = ffn_top + row * ffn_panel_height
        representative = group[0]
        lines = condition_lines(group)
        parts.append(f'<rect class="panel" x="{x}" y="{y}" width="{panel_width}" height="{ffn_panel_height - 14}"/>')
        parts.append(f'<text class="panel-title" x="{x + 16}" y="{y + 23}">F{index + 1} · {representative.cluster_count} clusters · {representative.intercluster_edges}/{len(representative.clusterable_edges)} cut edges</text>')
        for line_index, line in enumerate(lines):
            parts.append(f'<text class="condition" x="{x + 16}" y="{y + 42 + line_index * 14}">{html.escape(line)}</text>')
        draw_semantic_topology(parts, "ffn", x + 238, y + 58, 0.92, semantic_badges(representative))
    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def merge_operation_sequences(left: list[str], right: list[str]) -> list[str]:
    rows, columns = len(left), len(right)
    lengths = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows - 1, -1, -1):
        for column in range(columns - 1, -1, -1):
            if left[row] == right[column]:
                lengths[row][column] = 1 + lengths[row + 1][column + 1]
            else:
                lengths[row][column] = max(
                    lengths[row + 1][column], lengths[row][column + 1]
                )
    row = column = 0
    merged: list[str] = []
    while row < rows and column < columns:
        if left[row] == right[column]:
            merged.append(left[row])
            row += 1
            column += 1
        elif lengths[row + 1][column] >= lengths[row][column + 1]:
            merged.append(left[row])
            row += 1
        else:
            merged.append(right[column])
            column += 1
    return merged + left[row:] + right[column:]


def cluster_matrix_layout(
    experiments: list[Experiment], layer: str
) -> tuple[
    list[tuple[str, str]],
    dict[str, tuple[int, int]],
    dict[Path, dict[int, int]],
]:
    selected = [experiment for experiment in experiments if experiment.layer == layer]
    nodes_by_stage: dict[Path, dict[str, list[int]]] = {}
    sequences: dict[str, list[list[str]]] = defaultdict(list)
    for experiment in selected:
        stages = semantic_stage_map(experiment)
        grouped: dict[str, list[int]] = defaultdict(list)
        for node in sorted(experiment.memberships):
            grouped[stages[node]].append(node)
        nodes_by_stage[experiment.clustered_path] = grouped
        for stage, nodes in grouped.items():
            sequence = [experiment.source_nodes[node].label for node in nodes]
            if sequence not in sequences[stage]:
                sequences[stage].append(sequence)

    columns: list[tuple[str, str]] = []
    ranges: dict[str, tuple[int, int]] = {}
    canonical_sequences: dict[str, list[str]] = {}
    stage_order = SEMANTIC_LAYOUTS[layer]["stages"]
    for stage in stage_order:
        if stage not in sequences:
            continue
        canonical: list[str] = []
        for sequence in sorted(sequences[stage], key=len, reverse=True):
            canonical = merge_operation_sequences(canonical, sequence)
        start = len(columns)
        columns.extend((stage, operation) for operation in canonical)
        ranges[stage] = (start, len(columns))
        canonical_sequences[stage] = canonical

    alignments: dict[Path, dict[int, int]] = {}
    for experiment in selected:
        alignment: dict[int, int] = {}
        for stage, nodes in nodes_by_stage[experiment.clustered_path].items():
            canonical = canonical_sequences[stage]
            start, _ = ranges[stage]
            slot = 0
            for node in nodes:
                label = experiment.source_nodes[node].label
                while slot < len(canonical) and canonical[slot] != label:
                    slot += 1
                if slot == len(canonical):
                    raise ValueError(
                        f"could not align {label} in stage {stage} for {experiment.source_path}"
                    )
                alignment[node] = start + slot
                slot += 1
        alignments[experiment.clustered_path] = alignment
    return columns, ranges, alignments


def matrix_experiment_order(experiment: Experiment) -> tuple[int, int, int]:
    return (
        ("prefill", "decode").index(experiment.action),
        ("edge", "server").index(experiment.config),
        experiment.length,
    )


def draw_cluster_matrix_section(
    parts: list[str], experiments: list[Experiment], layer: str, y: float,
    left: float, cell_width: float,
) -> float:
    selected = sorted(
        (experiment for experiment in experiments if experiment.layer == layer),
        key=matrix_experiment_order,
    )
    columns, ranges, alignments = cluster_matrix_layout(experiments, layer)
    row_height = 36
    header_top = y + 34
    header_height = 120
    rows_top = y + 170
    rows_bottom = rows_top + len(selected) * row_height
    labels = SEMANTIC_LAYOUTS[layer]["labels"]
    parts.append(
        f'<text class="section" x="24" y="{y + 20:.1f}">{layer.capitalize()} · {len(columns)} aligned generic operations</text>'
    )
    for group_index, (stage, (start, end)) in enumerate(ranges.items()):
        x = left + start * cell_width
        group_width = (end - start) * cell_width
        fill = "#f6f8fa" if group_index % 2 == 0 else "#eef2f5"
        parts.append(
            f'<rect class="operation-group" x="{x + 1:.1f}" y="{header_top:.1f}" width="{group_width - 2:.1f}" height="{header_height:.1f}" rx="9" fill="{fill}"/>'
        )
        label_x = x + group_width / 2
        if end - start >= 3:
            parts.append(
                f'<text class="operation-name" x="{label_x:.1f}" y="{header_top + 20:.1f}" text-anchor="middle">{html.escape(labels[stage])}</text>'
            )
        else:
            parts.append(
                f'<text class="operation-name" x="{label_x:.1f}" y="{header_top + 54:.1f}" text-anchor="middle" transform="rotate(-55 {label_x:.1f} {header_top + 54:.1f})">{html.escape(labels[stage])}</text>'
            )
    for column, (_, operation) in enumerate(columns):
        x = left + (column + 0.5) * cell_width
        label = html.escape(operation)
        parts.append(
            f'<text class="generic-name" x="{x:.1f}" y="{header_top + 110:.1f}" text-anchor="end" transform="rotate(-55 {x:.1f} {header_top + 110:.1f})">{label}</text>'
        )

    previous_group = None
    for row, experiment in enumerate(selected):
        row_y = rows_top + row * row_height
        group = (experiment.action, experiment.config)
        if group != previous_group and row:
            parts.append(
                f'<line x1="20" y1="{row_y:.1f}" x2="{left + len(columns) * cell_width:.1f}" y2="{row_y:.1f}" stroke="#8a9199" stroke-width="1.2"/>'
            )
        previous_group = group
        if row % 2:
            parts.append(
                f'<rect x="20" y="{row_y:.1f}" width="{left + len(columns) * cell_width - 20:.1f}" height="{row_height:.1f}" fill="#fafbfc"/>'
            )
        condition = (
            f"{experiment.config} / {experiment.action} / "
            f"B={experiment.batch} / L={experiment.length}"
        )
        parts.append(
            f'<text class="row-label" x="24" y="{row_y + 22:.1f}">{html.escape(condition)}</text>'
        )
        for column in range(len(columns)):
            x = left + column * cell_width
            parts.append(
                f'<rect class="empty-cell" x="{x + 2:.1f}" y="{row_y + 4:.1f}" width="{cell_width - 4:.1f}" height="{row_height - 8:.1f}"/>'
            )

        alignment = alignments[experiment.clustered_path]
        stages = semantic_stage_map(experiment)
        canonical, _ = canonical_clusters(experiment)
        cluster_columns: dict[int, list[int]] = defaultdict(list)
        for node, cluster in experiment.memberships.items():
            cluster_columns[cluster].append(alignment[node])
        cluster_anchors = {
            cluster: min(cluster_columns[cluster]) for cluster in cluster_columns
        }
        for node, cluster in experiment.memberships.items():
            column = alignment[node]
            x = left + column * cell_width
            color = cluster_color(cluster_anchors[cluster], len(columns))
            title = html.escape(
                f"{labels[stages[node]]}: "
                f"{experiment.source_nodes[node].label}; C{canonical[cluster]}"
            )
            parts.append(
                f'<rect class="cluster-cell" x="{x + 2:.1f}" y="{row_y + 4:.1f}" width="{cell_width - 4:.1f}" height="{row_height - 8:.1f}" fill="{color}"><title>{title}</title></rect>'
            )
            parts.append(
                f'<text class="cell-label" x="{x + cell_width / 2:.1f}" y="{row_y + 22:.1f}" text-anchor="middle">C{canonical[cluster]}</text>'
            )
    parts.append(
        f'<line x1="{left:.1f}" y1="{rows_bottom:.1f}" x2="{left + len(columns) * cell_width:.1f}" y2="{rows_bottom:.1f}" stroke="#8a9199"/>'
    )
    return rows_bottom


def svg_generic_cluster_matrix(experiments: list[Experiment], output_path: Path) -> None:
    cell_width = 43
    row_height = 28
    left = 190
    section_gap = 12
    header_top = 17
    header_height = 70
    rows_top = 96
    attention_columns, attention_ranges, attention_alignments = (
        cluster_matrix_layout(experiments, "attention")
    )
    ffn_columns, ffn_ranges, ffn_alignments = cluster_matrix_layout(
        experiments, "ffn"
    )
    attention_split = attention_ranges["qk"][1]

    def slice_ranges(
        ranges: dict[str, tuple[int, int]], start: int, end: int
    ) -> dict[str, tuple[int, int]]:
        return {
            stage: (group_start - start, group_end - start)
            for stage, (group_start, group_end) in ranges.items()
            if group_start >= start and group_end <= end
        }

    attention_first = attention_columns[:attention_split]
    attention_second = attention_columns[attention_split:]
    attention_first_ranges = slice_ranges(attention_ranges, 0, attention_split)
    attention_second_ranges = slice_ranges(
        attention_ranges, attention_split, len(attention_columns)
    )
    first_band_width = len(attention_first) * cell_width
    second_band_width = (
        len(attention_second) * cell_width
        + section_gap
        + len(ffn_columns) * cell_width
    )
    matrix_width = max(first_band_width, second_band_width)
    band_height = rows_top + 8 * row_height
    band_gap = 15
    second_band_y = band_height + band_gap
    width = int(left + matrix_width + 24)
    height = int(second_band_y + band_height + 10)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#202124;letter-spacing:0}.section-label{font-size:11px;font-weight:700}.operation-group{stroke:#89919a;stroke-width:1}.operation-name{font-size:9px;font-weight:700}.generic-name{font-family:"DejaVu Sans Mono",monospace;font-size:7px;fill:#596168}.header-divider{stroke:#c4c9ce;stroke-width:1}.row-label{font-size:10px;font-weight:600}.empty-cell{fill:#edf0f2;stroke:#fff;stroke-width:1}.cluster-cell{stroke:#fff;stroke-width:1}.cell-label{font-size:7px;font-weight:700;fill:#fff}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    def draw_headers(
        layer: str, columns: list[tuple[str, str]],
        ranges: dict[str, tuple[int, int]], section_left: float,
        y_offset: float, section_name: str,
    ) -> None:
        labels = SEMANTIC_LAYOUTS[layer]["labels"]
        section_width = len(columns) * cell_width
        parts.append(
            f'<text class="section-label" x="{section_left + section_width / 2:.1f}" y="{y_offset + 14:.1f}" text-anchor="middle">{section_name}</text>'
        )

        def wrap_label(label: str, max_characters: int) -> list[str]:
            words = label.split()
            lines: list[str] = []
            current = ""
            for word in words:
                candidate = word if not current else f"{current} {word}"
                if current and len(candidate) > max_characters:
                    lines.append(current)
                    current = word
                else:
                    current = candidate
            if current:
                lines.append(current)
            return lines

        for group_index, (stage, (start, end)) in enumerate(ranges.items()):
            x = section_left + start * cell_width
            group_width = (end - start) * cell_width
            fill = "#f6f8fa" if group_index % 2 == 0 else "#eef2f5"
            parts.append(
                f'<rect class="operation-group" x="{x + 1:.1f}" y="{y_offset + header_top:.1f}" width="{group_width - 2:.1f}" height="{header_height:.1f}" rx="7" fill="{fill}"/>'
            )
            label_x = x + group_width / 2
            max_characters = max(10, (end - start) * 7)
            operation_lines = wrap_label(labels[stage], max_characters)
            operation_y = y_offset + header_top + 12
            for line_index, line in enumerate(operation_lines):
                parts.append(
                    f'<text class="operation-name" x="{label_x:.1f}" y="{operation_y + line_index * 10:.1f}" text-anchor="middle">{html.escape(line)}</text>'
                )
            parts.append(
                f'<line class="header-divider" x1="{x + 5:.1f}" y1="{y_offset + header_top + 47:.1f}" x2="{x + group_width - 5:.1f}" y2="{y_offset + header_top + 47:.1f}"/>'
            )
        for column, (_, operation) in enumerate(columns):
            x = section_left + (column + 0.5) * cell_width
            generic_text = operation.replace("-", " ").replace("+", " ")
            generic_lines = wrap_label(generic_text, 10)
            generic_y = y_offset + header_top + 57
            for line_index, line in enumerate(generic_lines):
                parts.append(
                    f'<text class="generic-name" x="{x:.1f}" y="{generic_y + line_index * 9:.1f}" text-anchor="middle">{html.escape(line)}</text>'
                )

    def condition_key(experiment: Experiment) -> tuple[str, str, int, int]:
        return (
            experiment.action,
            experiment.config,
            experiment.batch,
            experiment.length,
        )

    attention_rows = sorted(
        (experiment for experiment in experiments if experiment.layer == "attention"),
        key=matrix_experiment_order,
    )
    ffn_by_condition = {
        condition_key(experiment): experiment
        for experiment in experiments
        if experiment.layer == "ffn"
    }
    if {condition_key(experiment) for experiment in attention_rows} != set(
        ffn_by_condition
    ):
        raise ValueError("attention and FFN matrix conditions do not match")

    def draw_cells(
        experiment: Experiment, layer: str, total_columns: int,
        alignments: dict[Path, dict[int, int]], section_left: float,
        row_y: float, column_start: int, column_end: int,
    ) -> None:
        visible_columns = column_end - column_start
        for column in range(visible_columns):
            x = section_left + column * cell_width
            parts.append(
                f'<rect class="empty-cell" x="{x + 2:.1f}" y="{row_y + 3:.1f}" width="{cell_width - 4:.1f}" height="{row_height - 6:.1f}" rx="2"/>'
            )
        alignment = alignments[experiment.clustered_path]
        stages = semantic_stage_map(experiment)
        labels = SEMANTIC_LAYOUTS[layer]["labels"]
        canonical, _ = canonical_clusters(experiment)
        cluster_columns: dict[int, list[int]] = defaultdict(list)
        for node, cluster in experiment.memberships.items():
            cluster_columns[cluster].append(alignment[node])
        anchors = {cluster: min(values) for cluster, values in cluster_columns.items()}
        for node, cluster in experiment.memberships.items():
            global_column = alignment[node]
            if not column_start <= global_column < column_end:
                continue
            column = global_column - column_start
            x = section_left + column * cell_width
            color = cluster_color(anchors[cluster], total_columns)
            title = html.escape(
                f"{labels[stages[node]]}: {experiment.source_nodes[node].label}; "
                f"C{canonical[cluster]}"
            )
            parts.append(
                f'<rect class="cluster-cell" x="{x + 2:.1f}" y="{row_y + 3:.1f}" width="{cell_width - 4:.1f}" height="{row_height - 6:.1f}" rx="2" fill="{color}"><title>{title}</title></rect>'
            )
            parts.append(
                f'<text class="cell-label" x="{x + cell_width / 2:.1f}" y="{row_y + 18:.1f}" text-anchor="middle">C{canonical[cluster]}</text>'
            )

    def draw_band(y_offset: float, second: bool) -> None:
        if second:
            attention_band_columns = attention_second
            attention_band_ranges = attention_second_ranges
            attention_column_start = attention_split
            ffn_left = (
                left + len(attention_band_columns) * cell_width + section_gap
            )
            band_right = ffn_left + len(ffn_columns) * cell_width
            draw_headers(
                "attention", attention_band_columns, attention_band_ranges,
                left, y_offset, "Attention (continued)",
            )
            draw_headers(
                "ffn", ffn_columns, ffn_ranges, ffn_left, y_offset, "FFN"
            )
        else:
            attention_band_columns = attention_first
            attention_column_start = 0
            ffn_left = None
            band_right = left + len(attention_band_columns) * cell_width
            draw_headers(
                "attention", attention_band_columns, attention_first_ranges,
                left, y_offset, "Attention",
            )

        previous_group = None
        for row, attention in enumerate(attention_rows):
            row_y = y_offset + rows_top + row * row_height
            group = (attention.action, attention.config)
            if group != previous_group and row:
                parts.append(
                    f'<line x1="20" y1="{row_y:.1f}" x2="{band_right:.1f}" y2="{row_y:.1f}" stroke="#8a9199" stroke-width="1.1"/>'
                )
            previous_group = group
            if row % 2:
                parts.append(
                    f'<rect x="16" y="{row_y:.1f}" width="{band_right - 16:.1f}" height="{row_height:.1f}" fill="#fafbfc"/>'
                )
            condition = (
                f"{attention.config} / {attention.action} / "
                f"B={attention.batch} / L={attention.length}"
            )
            parts.append(
                f'<text class="row-label" x="18" y="{row_y + 18:.1f}">{html.escape(condition)}</text>'
            )
            draw_cells(
                attention, "attention", len(attention_columns),
                attention_alignments, left, row_y, attention_column_start,
                attention_column_start + len(attention_band_columns),
            )
            if second:
                ffn = ffn_by_condition[condition_key(attention)]
                assert ffn_left is not None
                draw_cells(
                    ffn, "ffn", len(ffn_columns), ffn_alignments,
                    ffn_left, row_y, 0, len(ffn_columns),
                )
        rows_bottom = y_offset + rows_top + len(attention_rows) * row_height
        parts.append(
            f'<line x1="{left:.1f}" y1="{rows_bottom:.1f}" x2="{band_right:.1f}" y2="{rows_bottom:.1f}" stroke="#8a9199"/>'
        )

    draw_band(0, second=False)
    draw_band(second_band_y, second=True)
    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def svg_to_drawio(svg_path: Path, output_path: Path) -> None:
    svg_root = ET.parse(svg_path).getroot()
    width = float(svg_root.attrib["width"])
    height = float(svg_root.attrib["height"])
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": "2026-08-22T00:00:00.000Z",
            "agent": "AccelGen cluster analysis",
            "version": "24.7.17",
            "compressed": "false",
        },
    )
    diagram = ET.SubElement(mxfile, "diagram", {"id": "cluster-matrix", "name": "Page-1"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": str(round(width)),
            "dy": str(round(height)),
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": str(round(width)),
            "pageHeight": str(round(height)),
            "math": "0",
            "shadow": "0",
        },
    )
    graph_root = ET.SubElement(model, "root")
    ET.SubElement(graph_root, "mxCell", {"id": "0"})
    ET.SubElement(graph_root, "mxCell", {"id": "1", "parent": "0"})
    next_id = 2

    def new_id() -> str:
        nonlocal next_id
        result = str(next_id)
        next_id += 1
        return result

    def tag_name(element: ET.Element) -> str:
        return element.tag.rsplit("}", 1)[-1]

    def numeric(value: str | None, default: float = 0.0) -> float:
        return float(value) if value is not None else default

    def style(parts: list[str]) -> str:
        return ";".join(parts) + ";"

    def add_vertex(
        value: str, vertex_style: str, x: float, y: float,
        vertex_width: float, vertex_height: float, tooltip: str | None = None,
    ) -> None:
        attributes = {
            "id": new_id(),
            "value": value,
            "style": vertex_style,
            "vertex": "1",
            "parent": "1",
        }
        if tooltip:
            attributes["tooltip"] = tooltip
        cell = ET.SubElement(graph_root, "mxCell", attributes)
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": f"{x:.2f}",
                "y": f"{y:.2f}",
                "width": f"{vertex_width:.2f}",
                "height": f"{vertex_height:.2f}",
                "as": "geometry",
            },
        )

    elements = list(svg_root)
    skip_indices: set[int] = set()
    text_sizes = {
        "section-label": 11,
        "operation-name": 9,
        "generic-name": 7,
        "row-label": 10,
        "cell-label": 7,
    }
    for index, element in enumerate(elements):
        if index in skip_indices:
            continue
        element_tag = tag_name(element)
        element_class = element.attrib.get("class", "")
        if element_tag == "rect":
            if element.attrib.get("width") == "100%":
                continue
            x = numeric(element.attrib.get("x"))
            y = numeric(element.attrib.get("y"))
            rect_width = numeric(element.attrib.get("width"))
            rect_height = numeric(element.attrib.get("height"))
            fill = element.attrib.get("fill")
            value = ""
            tooltip = None
            font_parts: list[str] = []
            if element_class == "operation-group":
                fill = fill or "#f6f8fa"
                stroke = "#89919a"
                rounded = "1"
            elif element_class == "empty-cell":
                fill = "#edf0f2"
                stroke = "#ffffff"
                rounded = "1"
            elif element_class == "cluster-cell":
                fill = fill or "#ffffff"
                stroke = "#ffffff"
                rounded = "1"
                if index + 1 < len(elements):
                    label = elements[index + 1]
                    if (
                        tag_name(label) == "text"
                        and label.attrib.get("class") == "cell-label"
                    ):
                        value = label.text or ""
                        skip_indices.add(index + 1)
                title = next(
                    (child.text for child in element if tag_name(child) == "title"),
                    None,
                )
                tooltip = title
                font_parts = [
                    "fontColor=#ffffff",
                    "fontSize=7",
                    "fontStyle=1",
                    "align=center",
                    "verticalAlign=middle",
                ]
            else:
                fill = fill or "#ffffff"
                stroke = "none"
                rounded = "0"
            add_vertex(
                value,
                style(
                    [
                        f"rounded={rounded}",
                        "whiteSpace=wrap",
                        "html=1",
                        f"fillColor={fill}",
                        f"strokeColor={stroke}",
                        "strokeWidth=1",
                        *font_parts,
                    ]
                ),
                x,
                y,
                rect_width,
                rect_height,
                tooltip,
            )
        elif element_tag == "text":
            value = element.text or ""
            font_size = text_sizes.get(element_class, 10)
            if element_class == "row-label":
                text_width = 188
                text_height = 15
                font_style = "1"
                font_family = "Arial"
                font_color = "#202124"
                align = "left"
            elif element_class == "section-label":
                text_width = max(100, len(value) * 8)
                text_height = 16
                font_style = "1"
                font_family = "Arial"
                font_color = "#202124"
                align = "center"
            elif element_class == "operation-name":
                text_width = max(42, len(value) * 6)
                text_height = 13
                font_style = "1"
                font_family = "Arial"
                font_color = "#202124"
                align = "center"
            elif element_class == "generic-name":
                text_width = max(40, len(value) * 5)
                text_height = 11
                font_style = "0"
                font_family = "Courier New"
                font_color = "#596168"
                align = "center"
            else:
                text_width = max(40, len(value) * 6)
                text_height = font_size + 6
                font_style = "0"
                font_family = "Arial"
                font_color = "#202124"
                align = "center"
            text_x = numeric(element.attrib.get("x"))
            if element.attrib.get("text-anchor") == "middle":
                text_x -= text_width / 2
            text_y = max(0.0, numeric(element.attrib.get("y")) - text_height)
            add_vertex(
                value,
                style(
                    [
                        "text",
                        "html=1",
                        "strokeColor=none",
                        "fillColor=none",
                        f"align={align}",
                        "verticalAlign=middle",
                        f"fontSize={font_size}",
                        f"fontStyle={font_style}",
                        f"fontFamily={font_family}",
                        f"fontColor={font_color}",
                    ]
                ),
                text_x,
                text_y,
                text_width,
                text_height,
            )
        elif element_tag == "line":
            stroke = element.attrib.get("stroke")
            if element_class == "header-divider":
                stroke = "#c4c9ce"
            stroke = stroke or "#8a9199"
            edge = ET.SubElement(
                graph_root,
                "mxCell",
                {
                    "id": new_id(),
                    "style": style(
                        [
                            "endArrow=none",
                            "html=1",
                            f"strokeColor={stroke}",
                            f"strokeWidth={element.attrib.get('stroke-width', '1')}",
                        ]
                    ),
                    "edge": "1",
                    "parent": "1",
                },
            )
            geometry = ET.SubElement(
                edge, "mxGeometry", {"relative": "1", "as": "geometry"}
            )
            ET.SubElement(
                geometry,
                "mxPoint",
                {
                    "x": element.attrib.get("x1", "0"),
                    "y": element.attrib.get("y1", "0"),
                    "as": "sourcePoint",
                },
            )
            ET.SubElement(
                geometry,
                "mxPoint",
                {
                    "x": element.attrib.get("x2", "0"),
                    "y": element.attrib.get("y2", "0"),
                    "as": "targetPoint",
                },
            )

    ET.indent(mxfile, space="  ")
    ET.ElementTree(mxfile).write(
        output_path, encoding="utf-8", xml_declaration=True
    )


def write_csv(experiments: list[Experiment], output_dir: Path) -> None:
    run_fields = [
        "model", "block", "layer", "action", "config", "batch", "length",
        "generic_ops", "clusterable_generic_ops", "clusters", "singleton_clusters", "min_ops_per_cluster",
        "mean_ops_per_cluster", "max_ops_per_cluster", "dag_edges", "clusterable_dag_edges",
        "intercluster_edges", "cut_edge_ratio", "ambiguous_matches", "excluded_const_broadcasts",
        "unmatched_source_ops", "unmatched_cluster_ops", "cycles_sum",
        "external_access_sum", "sram_access_sum", "flops_sum", "source_mlir",
        "clustered_mlir",
    ]
    with (output_dir / "cluster_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        for experiment in sorted(experiments, key=lambda exp: (exp.layer, exp.action, exp.config, exp.length)):
            sizes = list(experiment.cluster_sizes.values())
            metrics = experiment.cluster_metrics.values()
            writer.writerow({
                "model": experiment.model,
                "block": experiment.block,
                "layer": experiment.layer,
                "action": experiment.action,
                "config": experiment.config,
                "batch": experiment.batch,
                "length": experiment.length,
                "generic_ops": len(experiment.source_nodes),
                "clusterable_generic_ops": len(experiment.source_nodes) - experiment.excluded_source_ops,
                "clusters": experiment.cluster_count,
                "singleton_clusters": sum(size == 1 for size in sizes),
                "min_ops_per_cluster": min(sizes, default=0),
                "mean_ops_per_cluster": f"{mean(sizes):.4f}" if sizes else "0",
                "max_ops_per_cluster": max(sizes, default=0),
                "dag_edges": len(experiment.edges),
                "clusterable_dag_edges": len(experiment.clusterable_edges),
                "intercluster_edges": experiment.intercluster_edges,
                "cut_edge_ratio": f"{experiment.intercluster_edges / len(experiment.clusterable_edges):.6f}" if experiment.clusterable_edges else "0",
                "ambiguous_matches": experiment.ambiguous_matches,
                "excluded_const_broadcasts": experiment.excluded_source_ops,
                "unmatched_source_ops": experiment.unmatched_source_ops,
                "unmatched_cluster_ops": experiment.unmatched_cluster_ops,
                "cycles_sum": sum(metric.get("cycles", 0.0) for metric in metrics),
                "external_access_sum": sum(metric.get("externalAccess", 0.0) for metric in metrics),
                "sram_access_sum": sum(metric.get("sramAccess", 0.0) for metric in metrics),
                "flops_sum": sum(metric.get("flops", 0.0) for metric in metrics),
                "source_mlir": experiment.source_path,
                "clustered_mlir": experiment.clustered_path,
            })

    detail_fields = [
        "model", "layer", "action", "config", "batch", "length", "canonical_cluster",
        "mlir_cluster", "generic_ops", "cycles", "external_access", "sram_access",
        "flops", "source_op_ids", "operation_kinds", "semantic_stages",
    ]
    with (output_dir / "cluster_details.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        writer.writeheader()
        for experiment in sorted(experiments, key=lambda exp: (exp.layer, exp.action, exp.config, exp.length)):
            canonical, _ = canonical_clusters(experiment)
            stages = semantic_stage_map(experiment)
            members: dict[int, list[int]] = defaultdict(list)
            for node, cluster in experiment.memberships.items():
                members[cluster].append(node)
            for cluster in sorted(members, key=lambda item: canonical[item]):
                node_ids = sorted(members[cluster])
                metric = experiment.cluster_metrics.get(cluster, {})
                writer.writerow({
                    "model": experiment.model,
                    "layer": experiment.layer,
                    "action": experiment.action,
                    "config": experiment.config,
                    "batch": experiment.batch,
                    "length": experiment.length,
                    "canonical_cluster": f"C{canonical[cluster]}",
                    "mlir_cluster": experiment.cluster_names[cluster],
                    "generic_ops": len(node_ids),
                    "cycles": metric.get("cycles", 0.0),
                    "external_access": metric.get("externalAccess", 0.0),
                    "sram_access": metric.get("sramAccess", 0.0),
                    "flops": metric.get("flops", 0.0),
                    "source_op_ids": " ".join(map(str, node_ids)),
                    "operation_kinds": " ".join(experiment.source_nodes[node].label for node in node_ids),
                    "semantic_stages": " ".join(dict.fromkeys(stages[node] for node in node_ids)),
                })

    partition_fields = [
        "partition", "layer", "clusters", "intercluster_edges", "clusterable_edges",
        "run_count", "conditions",
    ]
    with (output_dir / "partition_groups.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=partition_fields)
        writer.writeheader()
        for layer, prefix in (("attention", "A"), ("ffn", "F")):
            for index, group in enumerate(partition_groups(experiments, layer), start=1):
                representative = group[0]
                writer.writerow({
                    "partition": f"{prefix}{index}",
                    "layer": layer,
                    "clusters": representative.cluster_count,
                    "intercluster_edges": representative.intercluster_edges,
                    "clusterable_edges": len(representative.clusterable_edges),
                    "run_count": len(group),
                    "conditions": "; ".join(condition_lines(group)),
                })


def write_summary(experiments: list[Experiment], output_dir: Path) -> None:
    lines = [
        f"# {experiments[0].model} cluster analysis",
        "",
        "Cluster membership is reconstructed by matching normalized `linalg.generic` operations from scheduled cluster functions to the original generic MLIR. Cluster IDs are canonicalized by their earliest source operation for visualization only.",
        "",
        "| Layer | Mode | Target | Batch | Length | Clusterable ops | Clusters | Cut edges | Ambiguous | Unmatched |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in sorted(experiments, key=lambda exp: (exp.layer, exp.action, exp.config, exp.length)):
        lines.append(
            f"| {experiment.layer} | {experiment.action} | {experiment.config} | "
            f"{experiment.batch} | {experiment.length} | {len(experiment.source_nodes) - experiment.excluded_source_ops} | "
            f"{experiment.cluster_count} | {experiment.intercluster_edges}/{len(experiment.clusterable_edges)} | "
            f"{experiment.ambiguous_matches} | "
            f"{experiment.unmatched_source_ops + experiment.unmatched_cluster_ops} |"
        )
    def cluster_range(layer: str, action: str, config: str) -> str:
        selected = sorted(
            (
                experiment
                for experiment in experiments
                if (experiment.layer, experiment.action, experiment.config)
                == (layer, action, config)
            ),
            key=lambda experiment: experiment.length,
        )
        if not selected:
            return "unavailable"
        if selected[0].cluster_count == selected[-1].cluster_count:
            return (
                f"{selected[0].cluster_count} across "
                f"L={selected[0].length}-{selected[-1].length}"
            )
        return (
            f"{selected[0].cluster_count} at L={selected[0].length} to "
            f"{selected[-1].cluster_count} at L={selected[-1].length}"
        )

    lines.extend([
        "",
        "## Key results",
        "",
        f"- Attention prefill is the most length-sensitive case: edge grows from {cluster_range('attention', 'prefill', 'edge')}, while server grows from {cluster_range('attention', 'prefill', 'server')}.",
        f"- FFN prefill grows from {cluster_range('ffn', 'prefill', 'edge')} on edge and remains {cluster_range('ffn', 'prefill', 'server')} on server.",
        f"- Decode is largely length-insensitive: attention is {cluster_range('attention', 'decode', 'edge')} on edge and {cluster_range('attention', 'decode', 'server')} on server; FFN is {cluster_range('ffn', 'decode', 'edge')} and {cluster_range('ffn', 'decode', 'server')}, respectively.",
        "- Edge uses batch 1 while server uses batch 8 in these files. Their comparison represents the complete deployment configurations, not an architecture-only controlled experiment.",
        "- The edge and server generic DAGs also contain different operation counts. Use within-configuration length trends for causal claims; absolute edge/server cluster counts are descriptive rather than a controlled fusion comparison.",
        "",
        "## Interpretation",
        "",
        "- Fewer clusters indicate that the scheduler can keep larger operation groups together on the selected architecture.",
        "- The cut-edge ratio is a topology-oriented proxy for materialized communication; use `external_access_sum` from `cluster_runs.csv` for the scheduler's volume-aware metric.",
        "- `ambiguous_matches` counts structurally identical operations that lack persistent source IDs. They are mapped deterministically, but symmetric branches may be exchanged.",
        "- Constant-broadcast generics are shown as gray infrastructure nodes because the scheduler passes their scalar constants as cluster arguments instead of cloning them.",
        "- Any nonzero unmatched count is a validation failure and should be investigated before using that row in a paper.",
        "",
        "## Figures",
        "",
        "The matrix aligns generic operations across configurations. Colors and `C#` labels are local to one row; repeated color within that row denotes operations assigned to the same cluster. Rounded header outlines identify the named model operation containing adjacent generics.",
        "",
        "![Generic-operation cluster matrix](generic_cluster_matrix.svg)",
        "",
        "Editable source: [`generic_cluster_matrix.drawio`](generic_cluster_matrix.drawio)",
        "",
    ])
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generic-dir", type=Path, default=Path("eval/generic"))
    parser.add_argument("--clustered-dir", type=Path, default=Path("eval/model"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/llama3-8b-clusters"))
    parser.add_argument("--model", default="llama3-8b")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--max-operation", type=int, default=6)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 4096])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiments = discover_experiments(args)
    if not experiments:
        raise SystemExit("no matching clustered MLIR files found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(experiments, args.output_dir)
    write_summary(experiments, args.output_dir)
    svg_path = args.output_dir / "generic_cluster_matrix.svg"
    svg_generic_cluster_matrix(experiments, svg_path)
    svg_to_drawio(svg_path, args.output_dir / "generic_cluster_matrix.drawio")
    unmatched = sum(exp.unmatched_source_ops + exp.unmatched_cluster_ops for exp in experiments)
    print(f"Analyzed {len(experiments)} experiments into {args.output_dir}")
    print(f"Unmatched generic operations: {unmatched}")
    return 1 if unmatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
