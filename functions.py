# pip install --upgrade pip
# pip install rdflib pykeen pandas owlready2 torch owlrl matplotlib numpy seaborn
import dataclasses
import json
import math
import os
import pickle
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Dict, List, Tuple, Set, Optional, Callable
import itertools
import copy
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import rdflib
import torch
import torch.nn.functional as F
from rdflib.collection import Collection
from rdflib.namespace import RDF, RDFS, OWL
import seaborn as sns
# OWL loading helpers + load_owl

torch.set_default_dtype(torch.float32)


def _uri_str(node: rdflib.term.Node) -> Optional[str]:
    return str(node) if isinstance(node, rdflib.term.URIRef) else None


def _ordered_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _local_name(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rsplit("/", 1)[-1]


def load_owl(
        owl_path: str,
        allowed_namespaces: Optional[Set[str]] = None,
) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Extract named classes, subclass relations, and disjoint pairs
    from an OWL ontology file.
    """
    g = rdflib.Graph()
    g.parse(owl_path, format="xml")

    def _keep(uri: str) -> bool:
        if allowed_namespaces is None:
            return True
        return any(uri.startswith(ns) for ns in allowed_namespaces)

    classes: Set[str] = set()

    for c in g.subjects(RDF.type, OWL.Class):
        uri = _uri_str(c)
        if uri and _keep(uri):
            classes.add(uri)

    for child, _, parent in g.triples((None, RDFS.subClassOf, None)):
        child_uri = _uri_str(child)
        parent_uri = _uri_str(parent)
        if child_uri and _keep(child_uri):  classes.add(child_uri)
        if parent_uri and _keep(parent_uri): classes.add(parent_uri)

    subclass_of: Set[Tuple[str, str]] = set()

    for child, _, parent in g.triples((None, RDFS.subClassOf, None)):
        child_uri, parent_uri = _uri_str(child), _uri_str(parent)
        if child_uri and parent_uri and _keep(child_uri) and _keep(parent_uri):
            subclass_of.add((child_uri, parent_uri))

    for a, _, b in g.triples((None, OWL.equivalentClass, None)):
        a_uri, b_uri = _uri_str(a), _uri_str(b)
        if a_uri and b_uri and _keep(a_uri) and _keep(b_uri):
            subclass_of.add((a_uri, b_uri))
            subclass_of.add((b_uri, a_uri))

    disjoint_pairs: Set[Tuple[str, str]] = set()

    for a, _, b in g.triples((None, OWL.disjointWith, None)):
        a_uri, b_uri = _uri_str(a), _uri_str(b)
        if a_uri and b_uri and _keep(a_uri) and _keep(b_uri):
            disjoint_pairs.add(_ordered_pair(a_uri, b_uri))

    for adc in g.subjects(RDF.type, OWL.AllDisjointClasses):
        for _, _, members in g.triples((adc, OWL.members, None)):
            try:
                collection = Collection(g, members)
            except Exception:
                continue
            member_uris = [
                uri for m in collection
                if (uri := _uri_str(m)) and _keep(uri)
            ]
            for i in range(len(member_uris)):
                for j in range(i + 1, len(member_uris)):
                    disjoint_pairs.add(_ordered_pair(member_uris[i], member_uris[j]))

    classes_list = sorted(classes)
    subclass_list = sorted(subclass_of)
    disjoint_list = sorted(disjoint_pairs)

    print(f"Loaded {owl_path}: {len(classes_list)} classes, "
          f"{len(subclass_list)} subclass axioms, {len(disjoint_list)} disjoint pairs")

    return classes_list, subclass_list, disjoint_list


# adding noise to ontology

def load_owl_with_errors(
        owl_path: str,
        allowed_namespaces: Optional[Set[str]] = None,
        subclass_drop_rate: float = 0.1,
        subclass_flip_rate: float = 0.0,
        disjoint_inject_rate: float = 0.05,
        class_drop_rate: float = 0.0,
        seed: int = 40,
) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Load ontology and inject controlled noise."""

    rng = random.Random(seed)
    classes, subclass_of, disjoint_pairs = load_owl(owl_path, allowed_namespaces)

    classes = set(classes)
    subclass_of = set(subclass_of)
    disjoint_pairs = set(disjoint_pairs)

    if class_drop_rate > 0:
        drop_classes = set(rng.sample(list(classes), int(len(classes) * class_drop_rate)))
        classes -= drop_classes
        subclass_of = {(c, p) for c, p in subclass_of if c not in drop_classes and p not in drop_classes}
        disjoint_pairs = {(a, b) for a, b in disjoint_pairs if a not in drop_classes and b not in drop_classes}

    if subclass_drop_rate > 0:
        subclass_of = {e for e in subclass_of if rng.random() > subclass_drop_rate}

    if subclass_flip_rate > 0:
        subclass_of = {(p, c) if rng.random() < subclass_flip_rate else (c, p)
                       for c, p in subclass_of}

    if disjoint_inject_rate > 0:
        class_list = list(classes)
        num_inject = int(disjoint_inject_rate * max(1, len(disjoint_pairs) + 1))
        injected = set()
        max_attempts = num_inject * 10
        attempts = 0
        while len(injected) < num_inject and attempts < max_attempts:
            pair = _ordered_pair(*rng.sample(class_list, 2))
            if pair not in disjoint_pairs:
                injected.add(pair)
            attempts += 1
        disjoint_pairs |= injected

    classes_list = sorted(classes)
    subclass_list = sorted(subclass_of)
    disjoint_list = sorted(disjoint_pairs)

    print(f"[NOISY] {owl_path}: {len(classes_list)} classes | "
          f"{len(subclass_list)} subclass | {len(disjoint_list)} disjoint")

    return classes_list, subclass_list, disjoint_list


class OntologyEdges:
    """
    Stores all subclass and disjoint edges for a given ontology,
    including asserted and entailed closures, depths, and entailment weights.
    Provides loss and violation helpers used by the training loops.
    """

    def __init__(self,
                 classes: List[str],
                 subclass_of: List[Tuple[str, str]],
                 disjoint_pairs: List[Tuple[str, str]],
                 device: Optional[str] = None):

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.cls2id = {uri: i for i, uri in enumerate(classes)}
        self.id2cls = {i: uri for i, uri in enumerate(classes)}
        self.num_classes = len(classes)

        # Subclass edges
        self.asserted_sub_edges = [(self.cls2id[c], self.cls2id[p]) for c, p in subclass_of]

        self.closure_sub_edges_with_depth = self._compute_closure_with_depth(self.asserted_sub_edges)
        self.closure_sub_edges = torch.tensor(
            [(c, a) for c, a, _ in self.closure_sub_edges_with_depth],
            dtype=torch.long, device=self.device
        )
        self.closure_depths = torch.tensor(
            [d for _, _, d in self.closure_sub_edges_with_depth],
            dtype=torch.float, device=self.device
        )

        # Disjoint edges
        self.asserted_disjoint_edges = [(self.cls2id[a], self.cls2id[b]) for a, b in disjoint_pairs]

        entailed_disjoint = self._entail_disjointness(
            self.asserted_disjoint_edges, self.closure_sub_edges_with_depth
        )
        self.entailed_disjoint_edges = torch.tensor(
            list(entailed_disjoint), dtype=torch.long, device=self.device
        )

        # Per-edge entailment weight
        self.entail_count = self._compute_entail_count()

        # Per-class depth from root (used in oversize_loss)
        depths = torch.full((self.num_classes,), float("inf"), device=self.device)
        all_children = {c for c, _ in self.asserted_sub_edges}
        all_parents = {p for _, p in self.asserted_sub_edges}
        for r in all_parents - all_children:
            depths[r] = 0.0

        changed = True
        while changed:
            changed = False
            for c, p in self.asserted_sub_edges:
                if depths[p] + 1 < depths[c]:
                    depths[c] = depths[p] + 1
                    changed = True

        depths[torch.isinf(depths)] = depths[~torch.isinf(depths)].max()
        self.class_depths = depths

        # Sibling edges
        sibling_pairs = self._compute_sibling_pairs()
        self.sibling_edges = (
            torch.tensor(sibling_pairs, dtype=torch.long, device=self.device)
            if sibling_pairs else None
        )

        # Violation tracking
        self.last_subclass_violations: Optional[int] = None
        self.last_disjoint_violations: Optional[int] = None
        self.last_sibling_distance: Optional[float] = None

    # Internal helpers
    def _compute_closure_with_depth(self, edges):
        parents = {i: {} for i in range(self.num_classes)}
        for c, p in edges:
            parents[c][p] = 1

        changed = True
        while changed:
            changed = False
            for c in range(self.num_classes):
                for p, d in list(parents[c].items()):
                    for gp, gd in parents[p].items():
                        new_d = d + gd
                        if gp not in parents[c] or new_d < parents[c][gp]:
                            parents[c][gp] = new_d
                            changed = True

        return [(c, a, d) for c, anc in parents.items() for a, d in anc.items()]

    def _entail_disjointness(self, disjoint_edges, subclass_closure) -> set:
        descendants = {}
        for c, a, _ in subclass_closure:
            descendants.setdefault(a, set()).add(c)

        entailed = set(disjoint_edges)
        for a, b in disjoint_edges:
            for da in descendants.get(a, []):
                entailed.add(tuple(sorted((da, b))))
            for db in descendants.get(b, []):
                entailed.add(tuple(sorted((a, db))))
        return entailed

    def _compute_entail_count(self):
        descendants = defaultdict(set)
        ancestors = defaultdict(set)
        for c, p, _ in self.closure_sub_edges_with_depth:
            descendants[p].add(c)
            ancestors[c].add(p)
        for i in range(self.num_classes):
            descendants[i].add(i)
            ancestors[i].add(i)

        counts = [
            len(descendants[c]) * len(ancestors[p])
            for c, p, _ in self.closure_sub_edges_with_depth
        ]
        return torch.log1p(torch.tensor(counts, dtype=torch.float, device=self.device))

    def _compute_sibling_pairs(self, max_pairs_per_parent: Optional[int] = 200,
                               seed: int = 0) -> List[Tuple[int, int]]:
        """
        max_pairs_per_parent caps combinatorial blowup at high-fan-out hubs
        (e.g. GO's near-root classes with hundreds of direct children, which
        would otherwise contribute C(n,2) pairs each and dominate/destabilize
        distance_loss once it activates). Pass None to disable capping.
        """
        rng = random.Random(seed)
        parent_to_children: dict = defaultdict(set)
        for child, parent in self.asserted_sub_edges:
            parent_to_children[parent].add(child)

        pairs: set = set()
        for children in parent_to_children.values():
            children = list(children)
            n = len(children)
            all_pairs_count = n * (n - 1) // 2
            if max_pairs_per_parent is None or all_pairs_count <= max_pairs_per_parent:
                for i in range(n):
                    for j in range(i + 1, n):
                        pairs.add(tuple(sorted((children[i], children[j]))))
            else:
                # Random subsample instead of full O(n^2) enumeration.
                sampled = 0
                attempts = 0
                max_attempts = max_pairs_per_parent * 20
                while sampled < max_pairs_per_parent and attempts < max_attempts:
                    a, b = rng.sample(children, 2)
                    p = tuple(sorted((a, b)))
                    if p not in pairs:
                        pairs.add(p)
                        sampled += 1
                    attempts += 1
        return list(pairs)

    def check_entailed_and_closure_edges(
            self,
            mn: torch.Tensor,
            mx: torch.Tensor,
            disjoint_margin: float = 0.02,
    ) -> dict:
        """
        Check how many entailed disjoint and transitive closure subclass edges
        are correctly satisfied (found/predicted) by the current box embedding.

        An edge is considered 'found' when the geometric constraint holds:
          - Subclass closure edge (c ⊆ p): mn[p] <= mn[c] AND mx[c] <= mx[p]
                                            in ALL dimensions.
          - Entailed disjoint edge (a ⊓ b = ⊥): boxes are separated by at
                                                  least `disjoint_margin` in
                                                  at least one dimension.

        """

        result = {}

        # Closure subclass edges
        if self.closure_sub_edges.numel() > 0:
            child = self.closure_sub_edges[:, 0]  # (E,)
            parent = self.closure_sub_edges[:, 1]

            # Containment holds iff mn[parent] <= mn[child] AND mx[child] <= mx[parent] for all dims
            lower_ok = (mn[parent] <= mn[child])
            upper_ok = (mx[child] <= mx[parent])

            edge_ok = (lower_ok & upper_ok).all(dim=1)
            total = edge_ok.numel()
            found = edge_ok.sum().item()
        else:
            total = found = 0

        result["closure_sub_total"] = total
        result["closure_sub_found"] = int(found)
        result["closure_sub_rate"] = found / total if total > 0 else 1.0

        # Entailed disjoint edges
        if self.entailed_disjoint_edges.numel() > 0:
            a = self.entailed_disjoint_edges[:, 0]
            b = self.entailed_disjoint_edges[:, 1]

            # Separation in each dim (positive = gap, negative = overlap)
            sep_ab = mn[b] - mx[a]
            sep_ba = mn[a] - mx[b]
            sep = torch.maximum(sep_ab, sep_ba)  # best-case separation per dim

            # A pair is 'satisfied' if at least one dimension achieves >= margin
            pair_ok = (sep.max(dim=1).values >= disjoint_margin)
            dis_total = pair_ok.numel()
            dis_found = pair_ok.sum().item()
        else:
            dis_total = dis_found = 0

        result["entailed_dis_total"] = dis_total
        result["entailed_dis_found"] = int(dis_found)
        result["entailed_dis_rate"] = dis_found / dis_total if dis_total > 0 else 1.0

        return result

    # Violation counters
    def count_subclass_violations(self, mn: torch.Tensor, mx: torch.Tensor) -> int | None:
        if self.closure_sub_edges.numel() == 0:
            self.last_subclass_violations = 0
            return 0
        child, parent = self.closure_sub_edges[:, 0], self.closure_sub_edges[:, 1]
        lower = (mn[child] < mn[parent]).sum().item()
        upper = (mx[child] > mx[parent]).sum().item()
        self.last_subclass_violations = int(lower) + int(upper)
        return self.last_subclass_violations

    def count_disjoint_violations(self, mn: torch.Tensor, mx: torch.Tensor, margin: float = 0.02) -> int | None:
        if len(self.entailed_disjoint_edges) == 0:
            self.last_disjoint_violations = 0
            return 0
        a, b = self.entailed_disjoint_edges[:, 0], self.entailed_disjoint_edges[:, 1]
        sep_ab = mn[b] - mx[a]
        sep_ba = mn[a] - mx[b]
        sep = torch.maximum(sep_ab, sep_ba)
        violated = (sep.max(dim=1).values < margin).sum().item()
        self.last_disjoint_violations = int(violated)
        return self.last_disjoint_violations

    # Loss functions
    def subclass_loss(self, mn: torch.Tensor, mx: torch.Tensor) -> torch.Tensor:
        """Containment loss weighted by entailment count per edge."""
        child, parent = self.closure_sub_edges[:, 0], self.closure_sub_edges[:, 1]
        lower = F.softplus(mn[parent] - mn[child])
        upper = F.softplus(mx[child] - mx[parent])
        per_edge_loss = (lower + upper).sum(dim=1)
        return (self.entail_count * per_edge_loss).mean()

    def disjoint_loss(self, mn: torch.Tensor, mx: torch.Tensor,
                      margin: float = 0.02, use_entailed: bool = True) -> torch.Tensor:
        dis_edges = (self.entailed_disjoint_edges if use_entailed
                     else torch.tensor(self.asserted_disjoint_edges,
                                       dtype=torch.long, device=self.device))
        if len(dis_edges) == 0:
            return torch.tensor(0.0, device=self.device)

        a, b = dis_edges[:, 0], dis_edges[:, 1]
        sep_ab = mn[b] - mx[a]
        sep_ba = mn[a] - mx[b]
        sep = torch.maximum(sep_ab, sep_ba)
        return F.softplus(margin - sep.max(dim=1).values).mean()

    def oversize_loss(self, log_vols: torch.Tensor, cfg) -> torch.Tensor:
        target = cfg.base_log_volume - cfg.depth_scale * torch.sqrt(self.class_depths)
        target = torch.clamp(target, min=cfg.min_log_volume)
        return (
                F.softplus(log_vols - target) +
                F.softplus(cfg.min_log_volume - log_vols)
        ).mean()

    def distance_loss(self, mn: torch.Tensor, mx: torch.Tensor) -> torch.Tensor:
        """
        Penalises siblings that have empty space between them.
        A gap of zero means boxes touch or overlap — no penalty.
        """
        if self.sibling_edges is None or len(self.sibling_edges) == 0:
            return torch.tensor(0.0, device=self.device)

        a, b = self.sibling_edges[:, 0], self.sibling_edges[:, 1]

        # Per-dim gap: positive = actual separation, negative = overlap
        gap = torch.maximum(mn[b] - mx[a], mn[a] - mx[b])  # (n_pairs, dim)

        # Only penalise positive gaps (separation); overlapping pairs cost nothing
        return F.softplus(gap).mean()

    def avg_sibling_distance(self, mn: torch.Tensor, mx: torch.Tensor) -> float:
        """
        Average gap between sibling boxes across all pairs and dimensions.
        Positive values = separation (boxes apart), negative = overlap.
        """
        if self.sibling_edges is None or len(self.sibling_edges) == 0:
            self.last_sibling_distance = 0.0
            return 0.0
        a, b = self.sibling_edges[:, 0], self.sibling_edges[:, 1]
        gap = torch.maximum(mn[b] - mx[a], mn[a] - mx[b])  # (n_pairs, dim)
        self.last_sibling_distance = gap.mean().item()
        return self.last_sibling_distance


@dataclass
class BoxConfig:
    # geometry
    dim: int = 6

    # optimisation
    steps: int = 3000
    lr: float = 1.0 / math.sqrt(dim)
    seed: int = 42

    # regularisation
    min_box_size: float = 0.05
    size_weight: float = 0.1

    # disjoint margin
    disjoint_margin: float = 0.02

    # constraint weights
    subclass_weight: float = 10.0
    disjoint_weight: float = 1.0

    # oversized-box control
    big_box_weight: float = 0.1
    base_log_volume: float = 2.0
    depth_scale: float = 0.5
    min_log_volume: float = -4.0

    # sibling proximity
    distance_weight: float = 0.1


class BoxEmbedding(torch.nn.Module):
    """
    Each class is represented by an axis-aligned box in R^d.
      center ∈ R^d
      half_size ∈ R_+^d (via softplus)
      min = center - half_size
      max = center + half_size
    """

    def __init__(self, n_classes: int, dim: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.center = torch.nn.Parameter(torch.randn(n_classes, dim, generator=g) * 0.1)
        self.raw_half_size = torch.nn.Parameter(torch.randn(n_classes, dim, generator=g) * 0.1)
        self._eps = 1e-6

    def half_size(self) -> torch.Tensor:
        return F.softplus(self.raw_half_size, beta=1.0) + self._eps

    def get_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:
        hs = self.half_size()
        return self.center - hs, self.center + hs

    def side_lengths(self) -> torch.Tensor:
        return 2.0 * self.half_size()

    def volumes(self) -> torch.Tensor:
        """
        Log-volume per box: sum of log(side_length) across dims.
        """
        return torch.log(self.side_lengths().clamp(min=1e-8)).sum(dim=-1)


@dataclass
class CurriculumSchedule:
    subclass_start: float = 0.0  # structural foundation first
    disjoint_start: float = 0.4  # separation only after containment exists
    sibling_start: float = 0.5  # siblings should be close
    big_box_start: float = 0.7  # size control
    ramp: bool = False


def _build_df(model, edges, cfg):
    with torch.no_grad():
        mn_np, mx_np = (t.cpu().numpy() for t in model.get_min_max())
    classes = [edges.id2cls[i] for i in range(edges.num_classes)]
    return pd.DataFrame({
        "class_uri": classes,
        "class_name": [_local_name(u) for u in classes],
        **{f"min_{d}": mn_np[:, d] for d in range(cfg.dim)},
        **{f"max_{d}": mx_np[:, d] for d in range(cfg.dim)},
    }).sort_values("class_name").reset_index(drop=True)


def _load_ontology(owl_path, noise, _preloaded):
    if noise:
        return load_owl_with_errors(owl_path)
    return _preloaded or load_owl(owl_path)


def learn_boxes_from_owl(owl_path, cfg, device=None, _preloaded=None, noise=False, steps=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    if steps is None:
        steps = cfg.steps

    classes, subclass_of, disjoint_pairs = _load_ontology(owl_path, noise, _preloaded)
    edges = OntologyEdges(classes, subclass_of, disjoint_pairs, device=device)

    if noise:
        clean_classes, clean_sub, clean_dis = _preloaded or load_owl(owl_path)
        eval_edges = OntologyEdges(clean_classes, clean_sub, clean_dis, device=device)
    else:
        eval_edges = edges

    model = torch.compile(BoxEmbedding(len(classes), cfg.dim, cfg.seed).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    final_loss = None

    for step in range(1, steps + 1):
        opt.zero_grad()
        mn, mx = model.get_min_max()
        mn, mx = mn.clamp(-1e6, 1e6), mx.clamp(-1e6, 1e6)

        loss = cfg.size_weight * F.softplus(cfg.min_box_size - (mx - mn).clamp(min=1e-6)).mean()

        if edges.closure_sub_edges.numel() > 0:
            loss += cfg.subclass_weight * edges.subclass_loss(mn, mx)

        if edges.sibling_edges is not None and len(edges.sibling_edges) > 0:
            loss += cfg.distance_weight * edges.distance_loss(mn, mx)

        if len(edges.asserted_disjoint_edges) > 0:
            loss += cfg.disjoint_weight * edges.disjoint_loss(mn, mx, margin=cfg.disjoint_margin, use_entailed=True)

        loss += cfg.big_box_weight * edges.oversize_loss(model.volumes(), cfg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        final_loss = loss.item()

        if step % max(1, steps // 10) == 0:
            with torch.no_grad():
                mn_eval, mx_eval = model.get_min_max()
                mn_eval = mn_eval.clamp(-1e6, 1e6)
                mx_eval = mx_eval.clamp(-1e6, 1e6)
                sub_viol = edges.count_subclass_violations(mn_eval, mx_eval)
                dis_viol = edges.count_disjoint_violations(mn_eval, mx_eval, cfg.disjoint_margin)
                sib_dist = edges.avg_sibling_distance(mn_eval, mx_eval)

                edges.last_subclass_violations = sub_viol
                edges.last_disjoint_violations = dis_viol

            print(f"step {step:>5}/{steps} | loss={loss.item():.4f} "
                  f"| sub_viol={sub_viol} | dis_viol={dis_viol} | avg_sib_dist={sib_dist:.4f}")

    with torch.no_grad():
        mn, mx = model.get_min_max()
        mn, mx = mn.clamp(-1e6, 1e6), mx.clamp(-1e6, 1e6)

        edges.last_subclass_violations = eval_edges.count_subclass_violations(mn, mx)
        edges.last_disjoint_violations = eval_edges.count_disjoint_violations(
            mn, mx, cfg.disjoint_margin
        )
        edges.last_sibling_distance = edges.avg_sibling_distance(mn, mx)

    return model, _build_df(model, edges, cfg), edges, final_loss


def learn_boxes_with_curriculum(owl_path, cfg, device=None, _preloaded=None,
                                schedule=None, noise=False, steps=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    if steps is None:
        steps = cfg.steps

    classes, subclass_of, disjoint_pairs = _load_ontology(owl_path, noise, _preloaded)
    edges = OntologyEdges(classes, subclass_of, disjoint_pairs, device=device)

    if noise:
        clean_classes, clean_sub, clean_dis = _preloaded or load_owl(owl_path)
        eval_edges = OntologyEdges(clean_classes, clean_sub, clean_dis, device=device)
    else:
        eval_edges = edges

    model = torch.compile(BoxEmbedding(len(classes), cfg.dim, cfg.seed).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    def scheduled_weight(base_weight, start_frac, step):
        if schedule is None:
            return base_weight
        start_step = int(start_frac * steps)
        if step < start_step:
            return 0.0
        if not schedule.ramp or start_frac == 0.0:
            return base_weight
        progress = (step - start_step) / max(1, steps - start_step)
        return base_weight * min(max(progress, 0.0), 1.0)

    final_loss = None

    for step in range(1, steps + 1):
        opt.zero_grad()
        mn, mx = model.get_min_max()
        mn, mx = mn.clamp(-1e6, 1e6), mx.clamp(-1e6, 1e6)

        loss = cfg.size_weight * F.softplus(cfg.min_box_size - (mx - mn).clamp(min=1e-6)).mean()

        subclass_w = scheduled_weight(cfg.subclass_weight, schedule.subclass_start if schedule else 0.0, step)
        sibling_w = scheduled_weight(cfg.distance_weight, schedule.sibling_start if schedule else 0.0, step)
        disjoint_w = scheduled_weight(cfg.disjoint_weight, schedule.disjoint_start if schedule else 0.0, step)
        bigbox_w = scheduled_weight(cfg.big_box_weight, schedule.big_box_start if schedule else 0.0, step)

        if subclass_w > 0 and edges.closure_sub_edges.numel() > 0:
            loss += subclass_w * edges.subclass_loss(mn, mx)

        if sibling_w > 0:
            loss += sibling_w * edges.distance_loss(mn, mx)

        if disjoint_w > 0:
            loss += disjoint_w * edges.disjoint_loss(mn, mx, margin=cfg.disjoint_margin, use_entailed=True)

        if bigbox_w > 0:
            loss += bigbox_w * edges.oversize_loss(model.volumes(), cfg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        final_loss = loss.item()

        if step % max(1, steps // 10) == 0:
            mn_eval, mx_eval = model.get_min_max()
            mn_eval = mn_eval.clamp(-1e6, 1e6)
            mx_eval = mx_eval.clamp(-1e6, 1e6)

            sub_viol = edges.count_subclass_violations(mn_eval, mx_eval)
            dis_viol = edges.count_disjoint_violations(mn_eval, mx_eval, cfg.disjoint_margin)
            sib_dist = edges.avg_sibling_distance(mn_eval, mx_eval)

            edges.last_subclass_violations = sub_viol
            edges.last_disjoint_violations = dis_viol

            print(f"step {step:>5}/{steps} | loss={loss.item():.4f} "
                  f"| sub_viol={sub_viol} | dis_viol={dis_viol} | avg_sib_dist={sib_dist:.4f}")

    with torch.no_grad():
        mn, mx = model.get_min_max()
        mn, mx = mn.clamp(-1e6, 1e6), mx.clamp(-1e6, 1e6)

        edges.last_subclass_violations = eval_edges.count_subclass_violations(mn, mx)
        edges.last_disjoint_violations = eval_edges.count_disjoint_violations(
            mn, mx, cfg.disjoint_margin
        )
        edges.last_sibling_distance = edges.avg_sibling_distance(mn, mx)

    return model, _build_df(model, edges, cfg), edges, final_loss


def _train_one_dim(args: tuple) -> tuple[int, dict]:
    d, owl_path, device, learn_fn, classes, subclass_of, disjoint_pairs, schedule, noise, cfg, steps = args

    if cfg is None:
        cfg = BoxConfig(dim=d, steps=10000, size_weight=0.1)

    else:
        cfg = dataclasses.replace(cfg, dim=d)

    kwargs = dict(
        owl_path=owl_path,
        cfg=cfg,
        device=device,
        _preloaded=(classes, subclass_of, disjoint_pairs),
        noise=noise,
        steps=steps,
    )

    if schedule is not None:
        kwargs["schedule"] = schedule

    result = learn_fn(**kwargs)

    if len(result) == 4:
        model, df, edges, loss = result
    else:
        model, df, edges = result
        loss = None

    return d, {
        "model": model,
        "df": df,
        "edges": edges,
        "sub_viol": edges.last_subclass_violations,
        "dis_viol": edges.last_disjoint_violations,
        "avg_sibling_dist": edges.last_sibling_distance,
        "loss": loss,
    }


def sweep_dimensions(
        owl_path: str,
        learn_fn: Callable,
        dims=range(2, 11),
        device: Optional[str] = None,
        max_workers: Optional[int] = None,
        noise: bool = False,
        schedule=None,
        cfg=None,
        path: Optional[str] = None,
        steps: Optional[int] = None,
) -> Dict[int, Dict]:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    dims = list(dims)

    print(f"Loading OWL: {owl_path}")

    # ------------------ timing ------------------
    load_start = time.time()
    classes, subclass_of, disjoint_pairs = (
        load_owl_with_errors(owl_path) if noise else load_owl(owl_path)
    )
    load_time = time.time() - load_start
    print(
        f"OWL loaded in {timedelta(seconds=int(load_time))} ({load_time:.2f}s)"
    )
    # --------------------------------------------

    print(
        f"Sweeping dims={dims} | device={device} | "
        f"noise={noise} | schedule={'yes' if schedule else 'no'}"
    )

    if path is not None:
        os.makedirs(path, exist_ok=True)

    results: Dict[int, Dict] = {}
    dim_times = []

    def _save_single(dim: int, info: dict):
        """Save one dimension in both old (pkl) and new (json) formats."""
        if path is None:
            return

        dim_path = os.path.join(path, f"dim_{dim}")
        os.makedirs(dim_path, exist_ok=True)

        raw_model = getattr(info["model"], "_orig_mod", info["model"])
        state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in raw_model.state_dict().items()
        }

        tmp_path = os.path.join(dim_path, "model.pt.tmp")
        final_path = os.path.join(dim_path, "model.pt")

        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, final_path)

        with open(os.path.join(dim_path, "data.pkl"), "wb") as f:
            pickle.dump(
                {
                    "df": info["df"],
                    "sub_viol": info["sub_viol"],
                    "dis_viol": info["dis_viol"],
                    "avg_sibling_dist": info.get("avg_sibling_dist"),
                    "loss": info.get("loss"),
                },
                f,
            )

        df = info["df"]
        num_classes = len(df)

        model = info["model"]
        edges = info.get("edges")

        with torch.no_grad():
            mn, mx = model.get_min_max()
            mn = mn.clamp(-1e6, 1e6)
            mx = mx.clamp(-1e6, 1e6)

            box_sizes = (mx - mn).mean(dim=1)
            volumes = model.volumes()

        closure_check = {}
        if edges is not None:
            closure_check = edges.check_entailed_and_closure_edges(mn, mx, 0.02)

        metrics = {
            "ontology": os.path.basename(path).replace("_noise", "").split("/")[-1],
            "dimension": dim,
            "variant": "curr" if schedule is not None else "plain",
            "steps": steps if steps is not None else (cfg.steps if cfg else 10000),
            "noise": noise,
            "curriculum": schedule is not None,
            "num_classes": num_classes,
            "num_subclass_axioms": len(info.get("subclass_of", [])),
            "num_disjoint_pairs": len(info.get("disjoint_pairs", [])),
            "final_loss": float(info.get("loss", 0))
            if info.get("loss")
            else None,
            "subclass_violations": int(info["sub_viol"]),
            "disjoint_violations": int(info["dis_viol"]),
            "clean_subclass_violations": int(info["sub_viol"]),
            "clean_disjoint_violations": int(info["dis_viol"]),
            "avg_sibling_distance": float(info.get("avg_sibling_dist", 0))
            if info.get("avg_sibling_dist")
            else None,
            "avg_box_size": float(box_sizes.mean().item()),
            "avg_log_volume": float(volumes.mean().item()),
            "min_box_size": float(box_sizes.min().item()),
            "max_box_size": float(box_sizes.max().item()),
            "closure_subclass_total": closure_check.get("closure_sub_total"),
            "closure_subclass_found": closure_check.get("closure_sub_found"),
            "closure_subclass_rate": float(
                closure_check.get("closure_sub_rate", 0)
            )
            if closure_check.get("closure_sub_rate")
            else None,
            "entailed_disjoint_total": closure_check.get(
                "entailed_dis_total"
            ),
            "entailed_disjoint_found": closure_check.get(
                "entailed_dis_found"
            ),
            "entailed_disjoint_rate": float(
                closure_check.get("entailed_dis_rate", 0)
            )
            if closure_check.get("entailed_dis_rate")
            else None,
        }

        with open(os.path.join(dim_path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    if device == "mps":
        for d in dims:
            print("=" * 60)
            print(f"Training dim={d} with {learn_fn.__name__}")

            start_dim = time.time()

            _, result = _train_one_dim(
                (
                    d,
                    owl_path,
                    device,
                    learn_fn,
                    classes,
                    subclass_of,
                    disjoint_pairs,
                    schedule,
                    noise,
                    cfg,
                    steps,
                )
            )

            elapsed_dim = time.time() - start_dim
            dim_times.append(elapsed_dim)

            results[d] = result
            _save_single(d, result)

            print(
                f"dim={d} done | "
                f"sub_viol={result['sub_viol']} | "
                f"dis_viol={result['dis_viol']} | "
                f"time={timedelta(seconds=int(elapsed_dim))} "
                f"({elapsed_dim:.1f}s)"
            )

    else:
        work_items = [
            (
                d,
                owl_path,
                device,
                learn_fn,
                classes,
                subclass_of,
                disjoint_pairs,
                schedule,
                noise,
                cfg,
                steps,
            )
            for d in dims
        ]

        future_start = {}

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {}

            for item in work_items:
                fut = pool.submit(_train_one_dim, item)
                futures[fut] = item[0]
                future_start[fut] = time.time()

            for fut in as_completed(futures):
                d = futures[fut]

                try:
                    d, result = fut.result()

                    elapsed_dim = time.time() - future_start[fut]
                    dim_times.append(elapsed_dim)

                    results[d] = result

                    try:
                        _save_single(d, result)
                    except Exception as e:
                        print(f"[SAVE ERROR] dim={d}: {e}")
                        continue

                    print(
                        f"dim={d} done | "
                        f"sub_viol={result['sub_viol']} | "
                        f"dis_viol={result['dis_viol']} | "
                        f"time={timedelta(seconds=int(elapsed_dim))} "
                        f"({elapsed_dim:.1f}s)"
                    )

                except Exception as e:
                    print(f"[ERROR] dim={d} failed: {e}")

    avg_dim_time = (
        sum(dim_times) / len(dim_times) if dim_times else 0.0
    )

    print("\n" + "=" * 60)
    print(
        f"OWL loading time : "
        f"{timedelta(seconds=int(load_time))} ({load_time:.2f}s)"
    )
    print(
        f"Average per dim  : "
        f"{timedelta(seconds=int(avg_dim_time))} ({avg_dim_time:.2f}s)"
    )
    print(f"Finished sweep. Stored {len(results)} dimensions.")
    print("=" * 60)

    return dict(sorted(results.items()))


def fast_load_sweep_results(path):
    """
    Fast-load sweep results by extracting only metrics.json files.

    Skips model loading entirely — returns just the metrics dictionaries.
    Much faster than load_sweep_results() when you only need metrics for analysis.

    Args:
        path: Base directory containing dim_* subdirectories with metrics.json

    Returns:
        Dict mapping dimension (int) -> metrics dict from metrics.json
    """
    import json

    results = {}
    dim_dirs = sorted([d for d in os.listdir(path) if d.startswith("dim_")])

    for d in dim_dirs:
        dim_path = os.path.join(path, d)
        metrics_path = os.path.join(dim_path, "metrics.json")

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            # Extract dimension from dirname or metrics
            dim = int(d.replace("dim_", ""))
            results[dim] = metrics
            print(f"  ✓ Loaded metrics (dim={dim})")
        else:
            print(f"  ⚠ No metrics.json in {d}")

    print(f"Loaded metrics for {len(results)} dimensions from {path}")
    return results


def load_sweep_results(path, classes, subclass_of, disjoint_pairs, device=None):
    """
    Load sweep results from disk, supporting multiple formats:
    - Old format: data.pkl + model.pt
    - New format: metrics.json + model.pt

    Returns dict mapping dimension -> {model, edges, df/metrics, sub_viol, dis_viol, ...}
    """
    import json

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    results = {}
    dim_dirs = sorted([d for d in os.listdir(path) if d.startswith("dim_")])

    for d in dim_dirs:
        print(f"Loading dim={d}...")
        dim_path = os.path.join(path, d)

        # Try to load metrics (new format) or data.pkl (old format)
        metrics_path = os.path.join(dim_path, "metrics.json")
        data_pkl_path = os.path.join(dim_path, "data.pkl")
        model_path = os.path.join(dim_path, "model.pt")

        # Load model
        try:
            state_dict = torch.load(model_path, map_location=device)
        except Exception as e:
            print(f"[SKIP] {model_path}: {e}")
            continue

        # Remove compile prefix if present
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # Infer actual dimension from model
        dim = state_dict["center"].shape[1]

        # Reconstruct model and load weights
        raw_model = BoxEmbedding(len(classes), dim, seed=0).to(device)
        raw_model.load_state_dict(state_dict)

        # Skip torch.compile() - models are already trained, no speed benefit for loading
        model = raw_model

        # Create edges object
        edges = OntologyEdges(classes, subclass_of, disjoint_pairs, device=device)

        # Load metrics/data
        data = {"model": model, "edges": edges}

        if os.path.exists(metrics_path):
            # New format: metrics.json
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            # Map JSON fields to expected result format
            data.update({
                "sub_viol": metrics.get("subclass_violations"),
                "dis_viol": metrics.get("disjoint_violations"),
                "avg_sibling_dist": metrics.get("avg_sibling_distance"),
                "loss": metrics.get("final_loss"),
                "df": None,  # Not available in JSON format
                "metrics": metrics,  # Full metrics dict
            })
            print(f"  ✓ Loaded from metrics.json (dim={dim})")

        elif os.path.exists(data_pkl_path):
            # Old format: data.pkl
            with open(data_pkl_path, "rb") as f:
                pkl_data = pickle.load(f)

            data.update(pkl_data)
            print(f"  ✓ Loaded from data.pkl (dim={dim})")
        else:
            print(f"  ⚠ No metrics found for dim={dim}")

        results[dim] = data

    print(f"Loaded {len(results)} dimensions from {path}")
    return results


def create_summary_from_saved(output_base, ontology_name, variants):
    """
    Create summary JSON from saved results (supports both pkl and json formats).

    Args:
        output_base: Base directory (e.g., 'saved/go')
        ontology_name: Name of ontology (e.g., 'go')
        variants: List of variant names (e.g., ['plain', 'curr'])

    Returns:
        Summary dict with all results and best per variant
    """
    import json
    from pathlib import Path

    summary = {
        "ontology": ontology_name,
        "variants": {},
        "best_results": {},
    }

    for variant in variants:
        variant_dir = Path(output_base) / variant
        if not variant_dir.exists():
            print(f"  Skipping {variant} (directory not found)")
            continue

        variant_results = []

        for dim_dir in sorted(variant_dir.iterdir()):
            if not dim_dir.name.startswith("dim_"):
                continue

            # Try metrics.json first, then data.pkl
            metrics_path = dim_dir / "metrics.json"
            data_pkl_path = dim_dir / "data.pkl"

            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                variant_results.append(metrics)
            elif data_pkl_path.exists():
                with open(data_pkl_path, "rb") as f:
                    pkl_data = pickle.load(f)

                # Convert pkl format to metrics-like dict
                metrics = {
                    "dimension": int(dim_dir.name.replace("dim_", "")),
                    "variant": variant,
                    "subclass_violations": pkl_data.get("sub_viol"),
                    "disjoint_violations": pkl_data.get("dis_viol"),
                    "avg_sibling_distance": pkl_data.get("avg_sibling_dist"),
                    "final_loss": pkl_data.get("loss"),
                }
                variant_results.append(metrics)

        if variant_results:
            summary["variants"][variant] = variant_results

            # Find best by total violations
            best = min(variant_results,
                       key=lambda m: (m.get("subclass_violations", 0) or 0) + (m.get("disjoint_violations", 0) or 0))
            summary["best_results"][variant] = best

    # Save summary
    summary_path = Path(output_base) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved: {summary_path}")
    print(f"\nBest results for {ontology_name.upper()}:")
    for variant, best in summary["best_results"].items():
        dim = best.get("dimension", "?")
        sub = best.get("subclass_violations", "?")
        dis = best.get("disjoint_violations", "?")
        print(f"  {variant:15} | dim={dim} | sub={sub} | dis={dis}")

    return summary


def _extract_metrics_from_result(result, dim=None):
    """
    Extract metrics from either fast_load or normal load result format.

    Supports:
    - Fast format (dict from metrics.json): has keys like 'subclass_violations', 'avg_box_size'
    - Normal format (dict with 'model', 'edges', 'metrics'): may have nested 'metrics' or computed values

    Returns normalized dict with: sub_viol, dis_viol, avg_box_size, avg_sibling_dist, loss
    """
    # Fast format: direct metrics from JSON
    if "subclass_violations" in result or "avg_box_size" in result:
        return {
            "sub_viol": result.get("subclass_violations") or 0,
            "dis_viol": result.get("disjoint_violations") or 0,
            "avg_box_size": result.get("avg_box_size") or 0.0,
            "avg_sibling_dist": result.get("avg_sibling_distance") or 0.0,
            "loss": result.get("final_loss") or 0.0,
        }

    # Normal format: has 'model' and 'edges', compute on the fly
    if "model" in result and "edges" in result:
        model = result["model"]
        edges = result["edges"]
        with torch.no_grad():
            mn, mx = model.get_min_max()
        return {
            "sub_viol": edges.count_subclass_violations(mn, mx),
            "dis_viol": edges.count_disjoint_violations(mn, mx),
            "avg_box_size": (mx - mn).mean().item(),
            "avg_sibling_dist": edges.avg_sibling_distance(mn, mx),
            "loss": result.get("loss") or result.get("metrics", {}).get("final_loss") or 0.0,
        }

    # Fallback: try to use whatever is available
    return {
        "sub_viol": result.get("sub_viol") or result.get("subclass_violations") or 0,
        "dis_viol": result.get("dis_viol") or result.get("disjoint_violations") or 0,
        "avg_box_size": result.get("avg_box_size") or 0.0,
        "avg_sibling_dist": result.get("avg_sibling_dist") or result.get("avg_sibling_distance") or 0.0,
        "loss": result.get("loss") or result.get("final_loss") or 0.0,
    }


def plot_curriculum_improvement(
        ontology_dirs=None,
        dim_range=(2, 10),
        save_path=None,
        fmt="pdf",
        exclude_outliers=False,
):
    """
    Plot mean % improvement of Curriculum over Plain across all ontologies.

    Single plot with:
    - Lines: Per-ontology % improvement (2 lines per ontology: sub + dis)
    - Bars: Average across all ontologies

    Positive % = Curriculum has fewer violations (improvement).
    Negative % = Curriculum has more violations (worse than Plain).

    Args:
        exclude_outliers: If True, exclude extreme outliers (e.g., GO dim 2) from mean calculation
    """
    import os

    if ontology_dirs is None:
        ontology_dirs = ['saved/cim', 'saved/doid', 'saved/go', 'saved/oeo']

    # Load results for all ontologies
    onto_data = {}
    for onto_dir in ontology_dirs:
        plain_path = f'{onto_dir}/plain'
        curr_path = f'{onto_dir}/curr'

        if not os.path.exists(plain_path) or not os.path.exists(curr_path):
            print(f"  Skipping {onto_dir} (missing plain or curr)")
            continue

        plain_results = fast_load_sweep_results(plain_path)
        curr_results = fast_load_sweep_results(curr_path)

        onto_name = os.path.basename(onto_dir).upper()
        onto_data[onto_name] = {'plain': plain_results, 'curr': curr_results}

    if not onto_data:
        raise ValueError("No ontology data found")

    # Collect ALL available dimensions from the data (not a continuous range)
    all_dims = set()
    for onto_name, data in onto_data.items():
        plain = data['plain']
        curr = data['curr']
        for d in plain.keys():
            if d in curr:
                all_dims.add(d)

    # Sort dimensions and filter by requested range
    min_dim, max_dim = dim_range
    dims = sorted([d for d in all_dims if min_dim <= d <= max_dim])

    if not dims:
        raise ValueError(f"No dimensions found in range {dim_range}")

    sub_by_onto = {onto: [] for onto in onto_data}
    dis_by_onto = {onto: [] for onto in onto_data}

    for onto_name, data in onto_data.items():
        plain = data['plain']
        curr = data['curr']

        for d in dims:
            if d not in plain or d not in curr:
                sub_by_onto[onto_name].append(None)
                dis_by_onto[onto_name].append(None)
                continue

            p_sub = _extract_metrics_from_result(plain[d], d)['sub_viol']
            c_sub = _extract_metrics_from_result(curr[d], d)['sub_viol']
            p_dis = _extract_metrics_from_result(plain[d], d)['dis_viol']
            c_dis = _extract_metrics_from_result(curr[d], d)['dis_viol']

            sub_imp = ((p_sub - c_sub) / p_sub * 100) if p_sub > 0 else (0.0 if c_sub == 0 else -100.0)
            dis_imp = ((p_dis - c_dis) / p_dis * 100) if p_dis > 0 else (0.0 if c_dis == 0 else -100.0)

            sub_by_onto[onto_name].append(sub_imp)
            dis_by_onto[onto_name].append(dis_imp)

    # Compute means (ignoring None and optionally excluding extreme outliers)
    sub_means = []
    dis_means = []
    for i in range(len(dims)):
        d = dims[i]
        sub_vals = [v for v in [sub_by_onto[o][i] for o in onto_data] if v is not None]
        dis_vals = [v for v in [dis_by_onto[o][i] for o in onto_data] if v is not None]

        # Exclude extreme outliers if requested (e.g., GO at dim 2)
        if exclude_outliers and d == 2:
            sub_vals = [v for v in sub_vals if v > -500]  # Exclude extreme negative
            dis_vals = [v for v in dis_vals if v > -500]

        sub_means.append(np.mean(sub_vals) if sub_vals else 0)
        dis_means.append(np.mean(dis_vals) if dis_vals else 0)

    x = np.arange(len(dims))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))

    # Color palette for ontologies
    onto_colors = {'CIM': '#1f77b4', 'DOID': '#ff7f0e', 'GO': '#2ca02c', 'OEO': '#d62728'}

    # Plot lines for each ontology
    for onto_name in sorted(onto_data.keys()):
        color = onto_colors.get(onto_name, '#999999')
        sub_vals = sub_by_onto[onto_name]
        dis_vals = dis_by_onto[onto_name]

        # Subclass line (solid)
        valid_sub = [(i, v) for i, v in enumerate(sub_vals) if v is not None]
        if valid_sub:
            idx, vals = zip(*valid_sub)
            ax.plot(list(idx), vals, marker='o', linestyle='-', color=color, linewidth=2, markersize=6,
                    label=f'{onto_name} Sub', alpha=0.7)

        # Disjoint line (dashed)
        valid_dis = [(i, v) for i, v in enumerate(dis_vals) if v is not None]
        if valid_dis:
            idx, vals = zip(*valid_dis)
            ax.plot(list(idx), vals, marker='s', linestyle='--', color=color, linewidth=2, markersize=6,
                    label=f'{onto_name} Dis', alpha=0.7)

    # Plot averages as bars
    ax.bar(x - bar_width / 2, sub_means, bar_width, label='Avg Sub', color='#E07B54', alpha=0.5, edgecolor='black',
           linewidth=1.2)
    ax.bar(x + bar_width / 2, dis_means, bar_width, label='Avg Dis', color='#5B8DB8', alpha=0.5, edgecolor='black',
           linewidth=1.2)

    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

    # Y-axis scaling: focus on positive range (improvements) while keeping negatives visible
    all_sub = [v for vals in sub_by_onto.values() for v in vals if v is not None]
    all_dis = [v for vals in dis_by_onto.values() for v in vals if v is not None]
    all_vals = all_sub + all_dis

    min_val = min(all_vals) if all_vals else -100
    max_val = max(all_vals) if all_vals else 100

    # Focus on positive range: show full positive spectrum, compress negative
    # Set upper limit slightly above max positive value
    ylim_high = max(max_val * 1.15, 50)

    # For lower limit: show enough to see negative values, but don't let extreme negatives dominate
    # Use 25% of the positive range for negative space, or show down to -200 if needed
    negative_buffer = max(abs(min_val) * 0.3, 150)
    ylim_low = -negative_buffer

    ax.set_ylim(ylim_low, ylim_high)

    # Collect and display off-scale values (very negative improvements)
    off_scale = []
    for onto_name in sorted(onto_data.keys()):
        sub_vals = sub_by_onto[onto_name]
        dis_vals = dis_by_onto[onto_name]
        for i, d in enumerate(dims):
            if sub_vals[i] is not None and sub_vals[i] < -200:
                off_scale.append(f'{onto_name} dim {d} (Sub): {sub_vals[i]:.0f}%')
            if dis_vals[i] is not None and dis_vals[i] < -200:
                off_scale.append(f'{onto_name} dim {d} (Dis): {dis_vals[i]:.0f}%')

    if off_scale:
        # Display off-scale values in a compact box
        textstr = 'Off-scale values:\n' + '\n'.join(off_scale[:8])  # Limit to 8 lines
        if len(off_scale) > 8:
            textstr += f'\n... and {len(off_scale) - 8} more'
        ax.text(1.02, 0.98, textstr, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Labels
    ax.set_xlabel('Embedding Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('% Improvement (Curriculum vs Plain)', fontsize=12, fontweight='bold')
    title_suffix = ' (GO dim 2 excluded from average)' if exclude_outliers else ''
    ax.set_title(f'Curriculum Learning: % Violation Reduction Across Ontologies{title_suffix}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims])
    ax.legend(loc='lower right', framealpha=0.9, fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.spines[['top', 'right']].set_visible(False)

    # Add value labels on bars
    for i, (sv, dv) in enumerate(zip(sub_means, dis_means)):
        ax.text(i - bar_width / 2, sv + 3, f'{sv:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(i + bar_width / 2, dv + 3, f'{dv:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    fig.tight_layout()

    if save_path:
        out_path = f'{save_path}.{fmt}'
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {out_path}")

    plt.show()
    return fig


def plot_dimension_degeneration(
        ontology_dirs=None,
        dim_range=(2, 30),
        save_path=None,
        fmt="pdf",
        y_max=None,
):
    """
    Plot normalized aggregated violation degeneration as dimensions increase.

    Shows how total violations (subclass + disjoint) grow relative to each 
    ontology's best dimension for both curriculum and plain learning.
    All ontologies start at 0% at their best dimension, then lines show
    relative degradation as dimensions increase. This allows comparing
    degeneration patterns across ontologies of different scales.

    Curriculum learning: solid line
    Plain learning: dashed line (----)

    Args:
        ontology_dirs: List of paths to ontology directories
        dim_range: Tuple (min_dim, max_dim) to display
        save_path: Path to save figure (without extension)
        fmt: Output format ('pdf', 'png', 'eps')
        y_max: Maximum Y-axis value (default: auto-scale to show GO clearly)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    if ontology_dirs is None:
        ontology_dirs = ['saved/cim', 'saved/doid', 'saved/go', 'saved/oeo']

    # Load results for all ontologies
    onto_data = {}
    for onto_dir in ontology_dirs:
        plain_path = f'{onto_dir}/plain'
        curr_path = f'{onto_dir}/curr'

        if not os.path.exists(plain_path) or not os.path.exists(curr_path):
            print(f"  Skipping {onto_dir} (missing plain or curr)")
            continue

        plain_results = fast_load_sweep_results(plain_path)
        curr_results = fast_load_sweep_results(curr_path)

        onto_name = os.path.basename(onto_dir).upper()
        onto_data[onto_name] = {'plain': plain_results, 'curr': curr_results}

    if not onto_data:
        raise ValueError("No ontology data found")

    # Collect ALL available dimensions from the data
    all_dims = set()
    for onto_name, data in onto_data.items():
        plain = data['plain']
        curr = data['curr']
        for d in plain.keys():
            if d in curr:
                all_dims.add(d)

    # Sort dimensions and filter by requested range
    min_dim, max_dim = dim_range
    dims = sorted([d for d in all_dims if min_dim <= d <= max_dim])

    if not dims:
        raise ValueError(f"No dimensions found in range {dim_range}")

    # For each ontology, find best dimension (minimum total violations for curriculum)
    # Then compute normalized scores relative to that best
    norm_curr_by_onto = {onto: [] for onto in onto_data}
    norm_plain_by_onto = {onto: [] for onto in onto_data}
    best_dims = {}  # Track which dim was best for each ontology
    best_totals = {}  # Track actual best total for each ontology

    for onto_name, data in onto_data.items():
        plain = data['plain']
        curr = data['curr']

        # Find best dimension (minimum total violations for curriculum)
        best_dim = None
        best_total = float('inf')
        for d in dims:
            if d not in curr:
                continue
            metrics = _extract_metrics_from_result(curr[d], d)
            total = metrics['sub_viol'] + metrics['dis_viol']
            if total < best_total:
                best_total = total
                best_dim = d
        best_dims[onto_name] = best_dim
        best_totals[onto_name] = best_total

        # Compute normalized values relative to best dimension
        for d in dims:
            if d not in plain or d not in curr:
                norm_curr_by_onto[onto_name].append(None)
                norm_plain_by_onto[onto_name].append(None)
                continue

            # Aggregate violations (sub + dis) for both methods
            p_total = (_extract_metrics_from_result(plain[d], d)['sub_viol'] + 
                       _extract_metrics_from_result(plain[d], d)['dis_viol'])
            c_total = (_extract_metrics_from_result(curr[d], d)['sub_viol'] + 
                       _extract_metrics_from_result(curr[d], d)['dis_viol'])

            # Get best dimension value for normalization (curriculum's best)
            best_total = best_totals[onto_name]

            # Normalize: 0% = best, >0% = worse (degeneration)
            # Use additive smoothing for near-zero baselines to avoid explosion
            # When best_total < 10, use (value - best) / (best + 10) to keep scale reasonable
            if best_total < 10:
                # For ontologies with very few violations, use absolute scale
                # This prevents tiny baselines from exploding the plot
                norm_curr = c_total - best_total  # absolute difference
                norm_plain = p_total - best_total
            else:
                # For ontologies with meaningful violations, use percentage
                eps = 1e-9
                norm_curr = 100 * (c_total - best_total) / max(best_total, eps)
                norm_plain = 100 * (p_total - best_total) / max(best_total, eps)

            norm_curr_by_onto[onto_name].append(norm_curr)
            norm_plain_by_onto[onto_name].append(norm_plain)

    # Create plot
    x = np.arange(len(dims))

    fig, ax = plt.subplots(figsize=(14, 7))

    # Color palette for ontologies
    onto_colors = {'CIM': '#1f77b4', 'DOID': '#ff7f0e', 'GO': '#2ca02c', 'OEO': '#d62728'}

    # Plot lines for each ontology
    for onto_name in sorted(onto_data.keys()):
        color = onto_colors.get(onto_name, '#999999')
        curr_vals = norm_curr_by_onto[onto_name]
        plain_vals = norm_plain_by_onto[onto_name]

        # Curriculum line (solid)
        valid_curr = [(i, v) for i, v in enumerate(curr_vals) if v is not None]
        if valid_curr:
            idx, vals = zip(*valid_curr)
            ax.plot(list(idx), vals, marker='o', linestyle='-', color=color, linewidth=2, markersize=6,
                    label=f'{onto_name} (curr)', alpha=0.8)

        # Plain line (dashed)
        valid_plain = [(i, v) for i, v in enumerate(plain_vals) if v is not None]
        if valid_plain:
            idx, vals = zip(*valid_plain)
            ax.plot(list(idx), vals, marker='s', linestyle='--', color=color, linewidth=2, markersize=6,
                    label=f'{onto_name} (plain)', alpha=0.6)

    # Horizontal line at 0 (optimal/best dimension)
    ax.axhline(y=0, color="black", linewidth=2, label="Best (curr baseline)")

    # Y-axis: linear scale focused on showing GO's range clearly
    # GO ranges from 0 to ~200% (curr) and ~20000% (plain)
    # But we want to see the curriculum comparison, so cap at reasonable level
    if y_max is None:
        # Auto-scale: find max non-extreme value
        all_vals = []
        for onto_name in onto_data:
            for v in norm_curr_by_onto[onto_name]:
                if v is not None and v < 5000:  # Ignore extreme outliers
                    all_vals.append(v)
            for v in norm_plain_by_onto[onto_name]:
                if v is not None and v < 5000:
                    all_vals.append(v)
        if all_vals:
            y_max = max(max(all_vals) * 1.2, 500)  # At least 500, or 1.2x max reasonable value
        else:
            y_max = 1000
    
    ax.set_ylim(-10, y_max)
    
    # Add grid lines
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    # Labels
    ax.set_xlabel('Embedding Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel("Relative Increase in Aggregated Violations (%)")
    ax.set_title(
        "Dimension-Induced Degeneration: Aggregated Violations (Subclass + Disjoint)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims])

    # Legend in upper left to avoid overlapping with degeneration curves
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9, ncol=2)
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()

    if save_path:
        out_path = f'{save_path}.{fmt}'
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {out_path}")

    plt.show()
    return fig


def plot_sweep_comparison(
        results_plain: dict,
        results_curriculum: dict,
        onto: str,
        figsize: tuple = (16, 5),
):
    dims = sorted(set(results_plain.keys()) & set(results_curriculum.keys()))
    x = np.arange(len(dims))
    bar_w = 0.35

    METRICS = [
        ("sub_viol", "Final Subclass Violations"),
        ("dis_viol", "Final Disjoint Violations"),
        ("avg_box_size", "Avg Box Side Length"),
        ("avg_sibling_dist", "Avg Sibling Distance"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(figsize[0] + 5, figsize[1]))
    for ax, (key, title) in zip(axes, METRICS):
        plain_vals = [_extract_metrics_from_result(results_plain[d], d)[key] for d in dims]
        curr_vals = [_extract_metrics_from_result(results_curriculum[d], d)[key] for d in dims]

        ax.bar(x - bar_w / 2, plain_vals, bar_w, label="Plain", color="#5B8DB8", alpha=0.85)
        ax.bar(x + bar_w / 2, curr_vals, bar_w, label="Curriculum", color="#E07B54", alpha=0.85)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Embedding Dimension")
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_ylabel(title)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Plain vs Curriculum {onto} — final metrics across embedding dimensions",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"sweep_comparison_{onto}.png", dpi=150, bbox_inches="tight")
    plt.show()


def table_sweep_comparison(
        results_plain: dict,
        results_curriculum: dict,
        onto: str,
):
    dims = sorted(set(results_plain.keys()) & set(results_curriculum.keys()))

    METRICS = [
        ("sub_viol", "Sub Violations"),
        ("dis_viol", "Dis Violations"),
        ("avg_box_size", "Avg Box Size"),
        ("avg_sibling_dist", "Avg Sibling Dist"),
        ("loss", "Loss"),
    ]
    metric_keys = [k for k, _ in METRICS]
    metric_labels = [l for _, l in METRICS]

    # --- collect per-dim pct changes ---
    LOWER_IS_BETTER = {"sub_viol", "dis_viol", "avg_box_size", "loss"}

    rows = {}
    for d in dims:
        plain = _extract_metrics_from_result(results_plain[d], d)
        curr = _extract_metrics_from_result(results_curriculum[d], d)
        pct = {}
        for key in metric_keys:
            p = plain[key]
            c = curr[key]
            if p == 0 and c == 0:
                pct[key] = 0.0
            elif p == 0:
                pct[key] = float("nan")
            else:
                pct[key] = (c - p) / p * 100

        # compute improvement scores (positive = curriculum is better)
        improvements = []
        for key in metric_keys:
            raw = pct[key]
            if np.isnan(raw):
                continue
            if key == "avg_sibling_dist":
                p, c = plain[key], curr[key]
                if p == 0:
                    imp = 0.0
                else:
                    imp = (abs(p) - abs(c)) / abs(p) * 100  # reduction toward 0
            elif key in LOWER_IS_BETTER:
                imp = -raw  # negative % change = improvement
            else:
                imp = raw
            improvements.append(imp)

        pct["overall"] = np.mean(improvements) if improvements else float("nan")
        rows[d] = pct

    # --- build DataFrame ---
    all_keys = metric_keys + ["overall"]
    all_labels = metric_labels + ["OVERALL Δ"]

    df = pd.DataFrame(
        {d: [rows[d][k] for k in all_keys] for d in dims},
        index=all_labels,
    ).T  # dims as rows, metrics as columns
    df.index.name = "Dim"

    # append a summary row: average pct change across dims for each metric
    df.loc["AVG across dims"] = df.mean()

    # --- pretty-print ---
    def fmt(v):
        if np.isnan(v):
            return "  n/a  "
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.1f}%"

    def fmt(v):
        try:
            if np.isnan(v):
                return "  n/a  "
        except (TypeError, ValueError):
            return str(v)
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.1f}%"

    apply_fn = df.map if hasattr(df, "map") and callable(df.map) else df.applymap
    styled = apply_fn(fmt)
    print(f"\nPlain → Curriculum  |  {onto}  |  % change = (curr − plain) / plain × 100\n")
    print(styled.to_string())
    print()

    return df


def plot_aggregated_sweep(ontology_dirs=None, dim_range=(2, 10), save_path=None, fmt="pdf",
                          results_plain=None, results_curr=None, results_plain_noise=None, results_curr_noise=None,
                          mode="compact"):
    """
    Aggregate and plot sweep results across multiple ontologies for 4 variants.

    TWO MODES:
    1. Disk mode (default): Pass ontology_dirs=['saved/cim', ...] → loads metrics.json from disk
    2. Memory mode: Pass pre-loaded results_* dicts directly (faster, no I/O)

    PLOTTING MODES:
    - "full": 2x2 grid with all 5 metrics per variant (original, space-heavy)
    - "compact": 2 plots showing ONLY violations (sub + dis) — the key metric
    - "ratio": Single plot showing curriculum/plain violation ratio (improvement factor)
    - "table": Returns LaTeX table string instead of plot

    Args:
        ontology_dirs: List of ontology base paths (disk mode), e.g. ['saved/cim', 'saved/doid', ...]
        dim_range: Tuple (min_dim, max_dim) to include, default (2, 10)
        save_path: Optional path to save figure (e.g., 'aggregated_sweep.pdf')
        fmt: Save format ('pdf', 'png', etc.)
        results_plain: Pre-loaded results dict {onto_name: {dim: result}} (memory mode)
        results_curr: Pre-loaded curriculum results
        results_plain_noise: Pre-loaded plain+noise results
        results_curr_noise: Pre-loaded curriculum+noise results
        mode: "full" | "compact" | "ratio" | "table"

    Example (memory mode):
        results_plain = {'cim': fast_load_sweep_results('saved/cim/plain'), ...}
        plot_aggregated_sweep(results_plain=results_plain, results_curr=results_curr, mode="compact")
    """
    # Detect mode
    memory_mode = results_plain is not None

    if memory_mode:
        print(f"\nUsing pre-loaded results (memory mode, {mode} output)...")
        aggregated = {
            'plain': _aggregate_results_dict(results_plain, dim_range),
            'curriculum': _aggregate_results_dict(results_curr, dim_range) if results_curr else {},
            'plain_noise': _aggregate_results_dict(results_plain_noise, dim_range) if results_plain_noise else {},
            'curriculum_noise': _aggregate_results_dict(results_curr_noise, dim_range) if results_curr_noise else {},
        }
    else:
        if not ontology_dirs:
            raise ValueError("Either ontology_dirs (disk mode) or results_* dicts (memory mode) must be provided")

        VARIANTS = {'plain': 'plain', 'curriculum': 'curr', 'plain_noise': 'plain_noise',
                    'curriculum_noise': 'curr_noise'}
        aggregated = {vname: {} for vname in VARIANTS.keys()}

        for var_name, var_folder in VARIANTS.items():
            print(f"\nLoading variant: {var_name}...")
            dim_metrics = {}

            for onto_dir in ontology_dirs:
                variant_path = os.path.join(onto_dir, var_folder)
                if not os.path.exists(variant_path):
                    print(f"  ⊘ Skipping {variant_path} (not found)")
                    continue

                for item in sorted(os.listdir(variant_path)):
                    if not item.startswith('dim_'):
                        continue
                    dim = int(item.replace('dim_', ''))
                    if dim < dim_range[0] or dim > dim_range[1]:
                        continue
                    metrics_file = os.path.join(variant_path, item, 'metrics.json')
                    if not os.path.exists(metrics_file):
                        continue
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    if dim not in dim_metrics:
                        dim_metrics[dim] = []
                    dim_metrics[dim].append({
                        'sub_viol': metrics.get('subclass_violations', 0) or 0,
                        'dis_viol': metrics.get('disjoint_violations', 0) or 0,
                        'avg_box_size': metrics.get('avg_box_size', 0.0) or 0.0,
                        'avg_sibling_dist': metrics.get('avg_sibling_distance', 0.0) or 0.0,
                        'loss': metrics.get('final_loss', 0.0) or 0.0,
                    })

            for dim in sorted(dim_metrics.keys()):
                lists = dim_metrics[dim]
                if not lists:
                    continue
                avg = {
                    'sub_viol': np.mean([x['sub_viol'] for x in lists]),
                    'dis_viol': np.mean([x['dis_viol'] for x in lists]),
                    'avg_box_size': np.mean([x['avg_box_size'] for x in lists]),
                    'avg_sibling_dist': np.mean([x['avg_sibling_dist'] for x in lists]),
                    'loss': np.mean([x['loss'] for x in lists]),
                }
                aggregated[var_name][dim] = avg

            print(f"  ✓ Loaded {len(dim_metrics)} dimensions")

    # === OUTPUT MODES ===

    if mode == "table":
        return _make_latex_table(aggregated, dim_range)

    if mode == "ratio":
        return _plot_violation_ratio(aggregated, dim_range, save_path, fmt)

    if mode == "compact":
        return _plot_compact_violations(aggregated, dim_range, save_path, fmt)

    # Default: full 2x2 grid (original behavior)
    return _plot_full_grid(aggregated, dim_range, save_path, fmt)


def _plot_compact_violations(aggregated, dim_range, save_path=None, fmt="pdf"):
    """
    Compact 2-plot design: violations only (the key metric).
    Plot 1: Plain vs Curriculum (no noise)
    Plot 2: Plain+Noise vs Curriculum+Noise
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pairs = [
        ('plain', 'curriculum', 'Without Noise'),
        ('plain_noise', 'curriculum_noise', 'With Noise'),
    ]

    for ax, (var1, var2, title_suffix) in zip(axes, pairs):
        data1 = aggregated.get(var1, {})
        data2 = aggregated.get(var2, {})

        if not data1 and not data2:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title_suffix)
            continue

        dims = sorted(set(data1.keys()) | set(data2.keys()))

        total_viol1 = [data1[d]['sub_viol'] + data1[d]['dis_viol'] for d in dims] if data1 else [0] * len(dims)
        total_viol2 = [data2[d]['sub_viol'] + data2[d]['dis_viol'] for d in dims] if data2 else [0] * len(dims)

        ax.plot(dims, total_viol1, marker='o', linewidth=2, color='#5B8DB8', label='Plain')
        ax.plot(dims, total_viol2, marker='s', linewidth=2, color='#E07B54', label='Curriculum')

        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Total Violations (Subclass + Disjoint)')
        ax.set_title(f'{title_suffix}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xticks(dims[::max(1, len(dims) // 5)])

    fig.suptitle('Curriculum Learning Effect on Violations (Averaged Across Ontologies)',
                 fontsize=12, fontweight='bold', y=1.05)
    fig.patch.set_facecolor('white')
    for ax in axes:
        ax.set_facecolor('white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format=fmt, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved: {save_path}")

    plt.show()


def _plot_violation_ratio(aggregated, dim_range, save_path=None, fmt="pdf"):
    """
    Single plot: shows curriculum/plain violation ratio.
    Ratio < 1 means curriculum improves over plain.
    Very compact — tells the whole story in one line.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 5))

    pairs = [
        ('plain', 'curriculum', 'Curriculum/Plain (no noise)'),
        ('plain_noise', 'curriculum_noise', 'Curriculum/Plain (with noise)'),
    ]

    for var1, var2, label in pairs:
        data1 = aggregated.get(var1, {})
        data2 = aggregated.get(var2, {})

        if not data1 or not data2:
            continue

        dims = sorted(set(data1.keys()) & set(data2.keys()))
        if not dims:
            continue

        ratios = []
        for d in dims:
            viol1 = data1[d]['sub_viol'] + data1[d]['dis_viol']
            viol2 = data2[d]['sub_viol'] + data2[d]['dis_viol']
            ratio = viol2 / viol1 if viol1 > 0 else 1.0
            ratios.append(ratio)

        ax.plot(dims, ratios, marker='o', linewidth=2, label=label)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='No improvement')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Violation Ratio (Curriculum / Plain)')
    ax.set_title('Curriculum Learning Improvement Factor\n(Ratio < 1 = curriculum reduces violations)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(0, 1.5)

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format=fmt, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved: {save_path}")

    plt.show()


def _make_latex_table(aggregated, dim_range):
    """
    Generate LaTeX table comparing variants at key dimensions.
    Compact, publication-ready.
    """
    dims_to_show = [d for d in [2, 5, 10] if dim_range[0] <= d <= dim_range[1]]
    if not dims_to_show:
        dims_to_show = sorted(list(aggregated['plain'].keys()))[:3]

    variants = ['plain', 'curriculum', 'plain_noise', 'curriculum_noise']
    variant_labels = ['Plain', 'Curriculum', 'Plain + Noise', 'Curriculum + Noise']

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Aggregated violation metrics across ontologies (averaged)}")
    lines.append("\\begin{tabular}{l|ccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Variant} & \\textbf{Dim 2} & \\textbf{Dim 5} & \\textbf{Dim 10} \\\\")
    lines.append("\\midrule")

    for var, label in zip(variants, variant_labels):
        data = aggregated.get(var, {})
        if not data:
            continue

        row_vals = []
        for d in dims_to_show:
            if d in data:
                total = data[d]['sub_viol'] + data[d]['dis_viol']
                row_vals.append(f"{total:.1f}")
            else:
                row_vals.append("-")

        lines.append(f"{label} & {' & '.join(row_vals)} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_code = "\n".join(lines)
    print("\n" + "=" * 60)
    print("LATEX TABLE GENERATED")
    print("=" * 60)
    print(latex_code)

    return latex_code


def _plot_full_grid(aggregated, dim_range, save_path=None, fmt="pdf"):
    """Original 2x2 grid with all 5 metrics per variant."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    variant_order = ['plain', 'curriculum', 'plain_noise', 'curriculum_noise']
    titles = ['Plain', 'Curriculum', 'Plain + Noise', 'Curriculum + Noise']

    for ax, var_name, title in zip(axes, variant_order, titles):
        data = aggregated[var_name]
        if not data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        dims = sorted(data.keys())
        sub_viol = [data[d]['sub_viol'] for d in dims]
        dis_viol = [data[d]['dis_viol'] for d in dims]
        avg_box_size = [data[d]['avg_box_size'] for d in dims]
        avg_sibling_dist = [data[d]['avg_sibling_dist'] for d in dims]
        loss = [data[d]['loss'] for d in dims]

        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("Violations", color="tab:red")
        l1, = ax.plot(dims, sub_viol, marker="o", color="tab:red", label="Subclass viol.", linewidth=2)
        l2, = ax.plot(dims, dis_viol, marker="s", color="tab:orange", label="Disjoint viol.", linewidth=2)
        ax.tick_params(axis="y", labelcolor="tab:red")
        ax.grid(True, alpha=0.3, linestyle=":")

        ax2 = ax.twinx()
        ax2.set_ylabel("Box Size / Distance / Loss", color="tab:blue")
        l3, = ax2.plot(dims, avg_box_size, marker="^", color="tab:blue", label="Avg box size", linewidth=2)
        l4, = ax2.plot(dims, avg_sibling_dist, marker="D", color="tab:purple", linestyle="--", label="Avg sibling dist",
                       linewidth=2)
        l5, = ax2.plot(dims, loss, marker="x", color="tab:green", linestyle=":", label="Loss", linewidth=2)
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        handles = [l1, l2, l3, l4, l5]
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.9)
        ax.set_title(title, fontweight="bold", fontsize=12)

    fig.suptitle("Aggregated Sweep Results Across Ontologies (Averaged per Dimension)",
                 fontsize=14, fontweight="bold", y=1.02)

    fig.patch.set_facecolor('white')
    for ax in axes:
        ax.set_facecolor('white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format=fmt, bbox_inches="tight", facecolor='white')
        print(f"\n✓ Saved: {save_path}")

    plt.show()


def _aggregate_results_dict(results_by_onto, dim_range):
    """
    Helper: aggregate pre-loaded results dict into averaged metrics per dimension.

    Args:
        results_by_onto: Dict {ontology_name: {dim: result_dict}}
                         where result_dict is from fast_load_sweep_results or load_sweep_results
        dim_range: Tuple (min_dim, max_dim)

    Returns:
        Dict {dim: averaged_metrics_dict}
    """
    import numpy as np

    dim_metrics = {}

    for onto_name, results in results_by_onto.items():
        for dim, result in results.items():
            if dim < dim_range[0] or dim > dim_range[1]:
                continue

            metrics = _extract_metrics_from_result(result, dim)

            if dim not in dim_metrics:
                dim_metrics[dim] = []
            dim_metrics[dim].append(metrics)

    # Average across ontologies
    aggregated = {}
    for dim in sorted(dim_metrics.keys()):
        lists = dim_metrics[dim]
        if not lists:
            continue
        aggregated[dim] = {
            'sub_viol': np.mean([x['sub_viol'] for x in lists]),
            'dis_viol': np.mean([x['dis_viol'] for x in lists]),
            'avg_box_size': np.mean([x['avg_box_size'] for x in lists]),
            'avg_sibling_dist': np.mean([x['avg_sibling_dist'] for x in lists]),
            'loss': np.mean([x['loss'] for x in lists]),
        }

    return aggregated


def evaluate_models(results_dict) -> pd.DataFrame:
    records = []
    for dim in sorted(results_dict.keys()):
        model = results_dict[dim]["model"]
        edges = results_dict[dim]["edges"]

        with torch.no_grad():
            mn, mx = model.get_min_max()

        records.append({
            "dim": dim,
            "sub_viol": edges.count_subclass_violations(mn, mx),
            "dis_viol": edges.count_disjoint_violations(mn, mx),
            "avg_box_size": (mx - mn).mean().item(),
            "avg_sibling_dist": edges.avg_sibling_distance(mn, mx),
            "loss": results_dict[dim].get("loss"),
        })

    return pd.DataFrame(records)


def plot_evaluation(eval_df, title: str = "Ontology Box Evaluation",
                    save_path: str = None, fmt: str = "pdf", legend: bool = True, ):
    dims = eval_df["dim"]

    fig, ax1 = plt.subplots(figsize=(7, 5))

    # --- Axis 1: Violations ---
    ax1.set_xlabel("Embedding dimension")
    ax1.set_xticks(dims[::2])
    ax1.set_ylabel("Violations", color="tab:red")
    l1, = ax1.plot(dims, eval_df["sub_viol"], marker="o",
                   color="tab:red", label="Subclass violations")
    l2, = ax1.plot(dims, eval_df["dis_viol"], marker="s",
                   color="tab:orange", label="Disjoint violations")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # --- Axis 2: Averages + Loss ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Average sizes / Loss", color="tab:blue")

    l3, = ax2.plot(dims, eval_df["avg_box_size"], marker="^",
                   color="tab:blue", label="Avg box size")

    handles = [l1, l2, l3]

    l6 = ax2.axhline(y=0, color="tab:blue", linestyle="--", linewidth=1, label="Zero")
    handles.append(l6)

    if "avg_sibling_dist" in eval_df.columns:
        l4, = ax2.plot(dims, eval_df["avg_sibling_dist"], marker="D",
                       color="tab:purple", linestyle="--",
                       label="Avg sibling distance")
        handles.append(l4)

    if "loss" in eval_df.columns:
        l5, = ax2.plot(dims, eval_df["loss"], marker="x",
                       color="tab:green", linestyle=":",
                       label="Loss")
        handles.append(l5)

    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # ax1.set_ylim(0, 60)
    # ax2.set_ylim(-15, 15)

    # --- Combined legend outside ---
    if legend:
        labels = [h.get_label() for h in handles]
        fig.legend(handles, labels,
                   loc="center left",
                   bbox_to_anchor=(0.97, 0.5),
                   borderaxespad=0.)

    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Make space for external legend
    plt.tight_layout(rect=[0, 0, 0.94, 1])

    if save_path:
        plt.savefig(save_path, format=fmt, bbox_inches="tight")

    plt.show()


def evaluate_concluded_relationships(results_dict, disjoint_margin: float):
    """
    Compute violations for concluded (entailed, non-asserted) relationships.

    Args:
        results_dict: Dict[int, dict] with 'model' and 'edges' keys (legacy)
                     OR dict[str, dict] loaded from metrics.json (new mode)
        disjoint_margin: Margin threshold for disjoint violations

    Returns:
        DataFrame with concluded subclass/disjoint violation rates
    """
    import torch

    # Detect input mode: metrics dict (from JSON) or legacy (model + edges)
    sample_key = list(results_dict.keys())[0]
    sample_val = results_dict[sample_key]

    if isinstance(sample_val, dict) and 'model' not in sample_val and 'subclass_violations' in sample_val:
        # NEW MODE: metrics.json dictionaries
        return _evaluate_concluded_from_metrics(results_dict, disjoint_margin)
    else:
        # LEGACY MODE: model + edges objects
        return _evaluate_concluded_legacy(results_dict, disjoint_margin)


def _evaluate_concluded_from_metrics(metrics_by_dim: dict, disjoint_margin: float) -> pd.DataFrame:
    """
    Evaluate concluded relationships using only metrics.json data.

    This version does NOT require live model or edges objects.
    It computes violation rates from pre-computed closure/entailed counts.

    Args:
        metrics_by_dim: Dict mapping dimension (int or str) to metrics dict
                       (as loaded from metrics.json)
        disjoint_margin: Margin for disjoint violations (not used in metrics-only mode)

    Returns:
        DataFrame with concluded violation statistics
    """
    records = []

    for dim_key, metrics in metrics_by_dim.items():
        dim = int(dim_key) if isinstance(dim_key, str) else dim_key

        # Concluded subclass: total entailed - asserted
        closure_total = metrics.get('closure_subclass_total') or 0
        closure_found = metrics.get('closure_subclass_found') or 0
        closure_rate = metrics.get('closure_subclass_rate') or 1.0

        # Violation rate = 1 - found_rate (approximation when exact viol count unavailable)
        # If we have the actual violation count, use it
        conc_sub_viol = metrics.get('concluded_subclass_violations', 0)
        conc_sub_total = closure_total

        # If no explicit violation count, estimate from rate
        if conc_sub_viol == 0 and closure_total > 0:
            # Estimate: violations ≈ total * (1 - rate)
            estimated_viol = int(closure_total * (1.0 - closure_rate))
            conc_sub_viol = estimated_viol

        # Concluded disjoint: entailed disjoint pairs
        entailed_dis_total = metrics.get('entailed_disjoint_total') or 0
        entailed_dis_found = metrics.get('entailed_disjoint_found') or 0
        entailed_dis_rate = metrics.get('entailed_disjoint_rate') or 1.0

        conc_dis_viol = metrics.get('concluded_disjoint_violations', 0)
        conc_dis_total = entailed_dis_total

        if conc_dis_viol == 0 and entailed_dis_total > 0:
            estimated_viol = int(entailed_dis_total * (1.0 - entailed_dis_rate))
            conc_dis_viol = estimated_viol

        # Embedding dimension for rate normalization
        embedding_dim = metrics.get('dimension', dim)

        records.append({
            'dim': dim,
            'conc_sub_total': conc_sub_total,
            'conc_sub_viol': conc_sub_viol,
            'conc_sub_rate': conc_sub_viol / max(conc_sub_total * embedding_dim, 1),
            'conc_dis_total': conc_dis_total,
            'conc_dis_viol': conc_dis_viol,
            'conc_dis_rate': conc_dis_viol / max(conc_dis_total, 1),
            'closure_rate': closure_rate,
            'entailed_dis_rate': entailed_dis_rate,
        })

    return pd.DataFrame(records).sort_values('dim').reset_index(drop=True)


def _evaluate_concluded_legacy(results_dict, disjoint_margin: float) -> pd.DataFrame:
    """
    Legacy implementation requiring live model and edges objects.
    Kept for backward compatibility.
    """
    records = []

    for dim, info in results_dict.items():
        model = info['model']
        edges = info['edges']

        with torch.no_grad():
            mn, mx = model.get_min_max()

        embedding_dim = mn.shape[1]

        # concluded subclass edges
        asserted_sub_set = {(int(c), int(p)) for c, p in edges.asserted_sub_edges}

        closure_list = edges.closure_sub_edges.tolist()  # [[c, p], ...]
        concluded_sub_idx = [
            i for i, cp in enumerate(closure_list)
            if (int(cp[0]), int(cp[1])) not in asserted_sub_set
        ]

        if concluded_sub_idx:
            idx_t = torch.tensor(concluded_sub_idx, dtype=torch.long,
                                 device=edges.device)
            c_edges = edges.closure_sub_edges[idx_t]
            c_ids, p_ids = c_edges[:, 0], c_edges[:, 1]

            # A subclass edge violates when child is NOT contained in parent.
            lower_viol = (mn[c_ids] < mn[p_ids]).sum().item()
            upper_viol = (mx[c_ids] > mx[p_ids]).sum().item()
            conc_sub_viol = int(lower_viol) + int(upper_viol)
            conc_sub_total = len(concluded_sub_idx)
        else:
            conc_sub_viol = 0
            conc_sub_total = 0

        # concluded disjoint pairs
        asserted_dis_set = {frozenset(p) for p in edges.asserted_disjoint_edges}

        if len(edges.entailed_disjoint_edges) > 0:
            entailed_list = edges.entailed_disjoint_edges.tolist()
            concluded_dis_idx = [
                i for i, pair in enumerate(entailed_list)
                if frozenset(pair) not in asserted_dis_set
            ]
        else:
            concluded_dis_idx = []

        if concluded_dis_idx:
            idx_t = torch.tensor(concluded_dis_idx, dtype=torch.long,
                                 device=edges.device)
            c_dis = edges.entailed_disjoint_edges[idx_t]
            a_ids, b_ids = c_dis[:, 0], c_dis[:, 1]

            sep_ab = mn[b_ids] - mx[a_ids]
            sep_ba = mn[a_ids] - mx[b_ids]
            sep = torch.maximum(sep_ab, sep_ba)

            # A disjoint pair is violated when boxes still overlap (max-sep < margin).
            violated = (sep.max(dim=1).values < disjoint_margin).sum().item()
            conc_dis_viol = int(violated)
            conc_dis_total = len(concluded_dis_idx)
        else:
            conc_dis_viol = 0
            conc_dis_total = 0

        records.append({
            'dim': dim,
            'conc_sub_total': conc_sub_total,
            'conc_sub_viol': conc_sub_viol,
            'conc_sub_rate': conc_sub_viol / max(conc_sub_total * embedding_dim, 1),
            'conc_dis_total': conc_dis_total,
            'conc_dis_viol': conc_dis_viol,
            'conc_dis_rate': conc_dis_viol / max(conc_dis_total, 1),
        })

    return pd.DataFrame(records).sort_values('dim').reset_index(drop=True)


def plot_concluded_evaluation(eval_df, title="Concluded Relationship Violations"):
    dims = eval_df["dim"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (y_plain, y_rate, ylabel, t) in zip(axes, [
        (eval_df["conc_sub_viol"], eval_df["conc_dis_viol"], "Violation count", "Raw violation counts"),
        (eval_df["conc_sub_rate"], eval_df["conc_dis_rate"], "Violation rate",
         "Violation rates (fraction of possible)"),
    ]):
        ax.plot(dims, y_plain, marker="o", color="tab:red", label="Concluded subclass")
        ax.plot(dims, y_rate, marker="s", color="tab:orange", label="Concluded disjoint")
        ax.set_xlabel("Embedding dimension")
        ax.set_ylabel(ylabel)
        ax.set_title(t, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def sweep_schedule_combinations(
        owl_path,
        params,
        dim,
        steps,
        base_schedule,
        _preloaded=None,
        device=None,
):
    # Load once before the loop
    if _preloaded is None:
        _preloaded = load_owl(owl_path)  # whatever your loader is

    # Build all combinations
    keys = list(params.keys())
    values = list(params.values())
    combos = list(itertools.product(*values))

    results = []

    for combo in combos:
        # Build schedule
        sched = copy.deepcopy(base_schedule)
        for k, v in zip(keys, combo):
            setattr(sched, k, v)

        # Build config
        cfg = BoxConfig(
            dim=dim,
            steps=steps,
        )

        # Train
        model, df, edges, final_loss = learn_boxes_with_curriculum(
            owl_path=owl_path,
            cfg=cfg,
            device=device,
            _preloaded=_preloaded,
            schedule=sched,
        )

        # Final metrics
        mn, mx = model.get_min_max()

        sub_viol = edges.count_subclass_violations(mn, mx)
        dis_viol = edges.count_disjoint_violations(mn, mx, cfg.disjoint_margin)

        result_row = {
            **{k: v for k, v in zip(keys, combo)},
            "sub_viol": sub_viol,
            "dis_viol": dis_viol,
            "loss_final": final_loss,
        }

        results.append(result_row)

        print(f"Done: {result_row}")

    return pd.DataFrame(results)


def plot_combo_heatmap_unified(
        df: pd.DataFrame,
        title: str = "Schedule Parameter Sweep",
        metrics: list[str] | None = None,
        figsize: tuple[float, float] | None = None,
):
    param_cols = [
        c for c in df.columns
        if c not in ("sub_viol", "dis_viol", "sub_viol_norm", "dis_viol_norm", "loss_final")
    ]

    if metrics is None:
        metrics = [m for m in ["sub_viol", "dis_viol", "loss_final"] if m in df.columns]
    else:
        metrics = [m for m in metrics if m in df.columns]

    df = df.copy()

    def fmt_val(v):
        try:
            f = float(v)
            return f"{f:.0%}" if 0.0 <= f <= 1.0 else str(v)
        except (TypeError, ValueError):
            return str(v)

    # Build param columns as formatted strings
    param_df = df[param_cols].map(fmt_val)

    # Build metric columns
    metric_df = df[metrics].copy().astype(object)
    for m in metrics:
        is_loss = m.startswith("loss")
        metric_df[m] = df[m].apply(lambda v: f"{v:.3f}" if is_loss else f"{int(round(v))}")

    # Combine: param cols first, then metrics
    combined = pd.concat([param_df, metric_df], axis=1)

    n_rows, n_cols = combined.shape
    fig_w = max(6, n_cols * 1.8)
    fig_h = max(4, n_rows * 0.55)

    fig, ax = plt.subplots(figsize=figsize or (fig_w, fig_h))

    # Blank numeric data (all zeros) — we only want annotations, no color
    blank = pd.DataFrame(0, index=combined.index, columns=combined.columns)

    sns.heatmap(
        blank,
        ax=ax,
        annot=combined,
        fmt="",
        cmap=["white"],  # flat white, no color variation
        cbar=False,
        linewidths=0.4,
        linecolor="lightgrey",
        annot_kws={"fontsize": 9},
        vmin=0, vmax=1,
    )

    # Shade param columns lightly to distinguish them
    for i in range(len(param_cols)):
        ax.add_patch(plt.Rectangle((i, 0), 1, n_rows, fill=True,
                                   color="#f0f0f0", zorder=0))

    all_cols = list(param_cols) + metrics
    ax.set_xticklabels(all_cols, fontsize=10, rotation=20, ha="right")
    ax.set_yticklabels(range(n_rows), fontsize=8, rotation=0)
    ax.set_ylabel("combination index", fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)

    plt.tight_layout()
    plt.show()


def create_comparison_table(eval_plain, eval_curr, onto_name):
    """
    Create LaTeX-ready comparison table for concluded relationships.

    Args:
        eval_plain: DataFrame from evaluate_concluded_relationships (plain learning)
        eval_curr: DataFrame from evaluate_concluded_relationships (curriculum learning)
        onto_name: Name of the ontology (e.g., 'CIM', 'DOID')

    Prints LaTeX table code to stdout.
    """
    dims = sorted(set(eval_plain['dim'].tolist()) & set(eval_curr['dim'].tolist()))

    print(f"\n{'=' * 70}")
    print(f"LATEX TABLE: {onto_name} Concluded Relationships")
    print(f"{'=' * 70}")

    print("\\begin{table}[htbp]")
    print("\\centering")
    print(f"\\caption{{Concluded relationship evaluation for {onto_name}}}")
    print("\\begin{tabular}{c|cc|cc}")
    print("\\toprule")
    print("\\textbf{Dim} & \\multicolumn{2}{c|}{\\textbf{Plain}} & \\multicolumn{2}{c}{\\textbf{Curriculum}} \\\\")
    print("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    print("& Sub Viol & Dis Viol & Sub Viol & Dis Viol \\\\")
    print("\\midrule")

    for dim in dims:
        plain_row = eval_plain[eval_plain['dim'] == dim].iloc[0]
        curr_row = eval_curr[eval_curr['dim'] == dim].iloc[0]

        dim_int = int(dim) if isinstance(dim, float) else dim

        print(f"{dim_int:2d} & "
              f"{int(plain_row['conc_sub_viol']):4d} & {int(plain_row['conc_dis_viol']):3d} & "
              f"{int(curr_row['conc_sub_viol']):4d} & {int(curr_row['conc_dis_viol']):3d} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print(f"{'=' * 70}\n")


def create_clean_vs_noisy_table(onto_name, clean_results, noisy_results, dims):
    """
    Create LaTeX table comparing clean vs noisy evaluation results.

    Args:
        onto_name: Name of the ontology
        clean_results: Dict with 'eval_plain' and 'eval_curr' DataFrames
        noisy_results: Dict with 'eval_plain_noisy' and 'eval_curr_noisy' DataFrames
        dims: List of dimensions to include

    Prints LaTeX table code to stdout.
    """
    print(f"\n{'=' * 70}")
    print(f"LATEX TABLE: {onto_name} Clean vs Noisy Comparison")
    print(f"{'=' * 70}")

    print("\\begin{table}[htbp]")
    print("\\centering")
    print(f"\\caption{{Link prediction performance: Clean vs Noisy {onto_name}}}")
    print("\\begin{tabular}{c|cc|cc}")
    print("\\toprule")
    print(
        "\\multirow{2}{*}{\\textbf{Dim}} & \\multicolumn{2}{c|}{\\textbf{Clean}} & \\multicolumn{2}{c}{\\textbf{Noisy}} \\\\")
    print("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    print("& Plain Viol & Curr Viol & Plain Viol & Curr Viol \\\\")
    print("\\midrule")

    for dim in dims:
        # Clean results
        clean_plain_viol = None
        clean_curr_viol = None

        if 'eval_plain' in clean_results:
            row = clean_results['eval_plain'][clean_results['eval_plain']['dim'] == dim]
            if len(row) > 0:
                clean_plain_viol = int(row.iloc[0]['conc_sub_viol']) + int(row.iloc[0]['conc_dis_viol'])

        if 'eval_curr' in clean_results:
            row = clean_results['eval_curr'][clean_results['eval_curr']['dim'] == dim]
            if len(row) > 0:
                clean_curr_viol = int(row.iloc[0]['conc_sub_viol']) + int(row.iloc[0]['conc_dis_viol'])

        # Noisy results
        noisy_plain_viol = None
        noisy_curr_viol = None

        if 'eval_plain_noisy' in noisy_results:
            row = noisy_results['eval_plain_noisy'][noisy_results['eval_plain_noisy']['dim'] == dim]
            if len(row) > 0:
                noisy_plain_viol = int(row.iloc[0]['conc_sub_viol']) + int(row.iloc[0]['conc_dis_viol'])

        if 'eval_curr_noisy' in noisy_results:
            row = noisy_results['eval_curr_noisy'][noisy_results['eval_curr_noisy']['dim'] == dim]
            if len(row) > 0:
                noisy_curr_viol = int(row.iloc[0]['conc_sub_viol']) + int(row.iloc[0]['conc_dis_viol'])

        plain_clean_str = str(clean_plain_viol) if clean_plain_viol is not None else "--"
        plain_noisy_str = str(noisy_plain_viol) if noisy_plain_viol is not None else "--"
        curr_clean_str = str(clean_curr_viol) if clean_curr_viol is not None else "--"
        curr_noisy_str = str(noisy_curr_viol) if noisy_curr_viol is not None else "--"

        print(
            f"{dim:2d} & {plain_clean_str:>8s} & {curr_clean_str:>8s} & {plain_noisy_str:>8s} & {curr_noisy_str:>8s} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print(f"{'=' * 70}\n")