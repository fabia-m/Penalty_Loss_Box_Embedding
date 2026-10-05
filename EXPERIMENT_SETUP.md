# Experiment Setup: Box Embedding Evaluation of Ontologies

This document describes the complete experimental configuration for box embedding evaluations across multiple ontologies, including plain learning and curriculum learning.

---

## Table of Contents

1. [Overview](#overview)
2. [Ontologies Evaluated](#ontologies-evaluated)
3. [Box Configuration](#box-configuration)
4. [Curriculum Schedules](#curriculum-schedules)
5. [Training Protocol](#training-protocol)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Experimental Variants](#experimental-variants)
8. [Computational Resources](#computational-resources)
9. [Reproducibility](#reproducibility)

---

## Overview

All experiments evaluate **box embeddings** for ontology representation learning, where each class is represented as an axis-aligned hyperrectangle in ℝᵈ. The method learns geometric constraints that encode:

- **Subclass relationships**: Child boxes must be contained within parent boxes
- **Disjointness constraints**: Disjoint classes must have separated boxes
- **Size regularization**: Boxes should maintain reasonable volumes
- **Sibling separation**: Sibling classes should be dispersed in the embedding space

---

## Ontologies Evaluated

| Ontology | Domain                         | Classes | SubClass Axioms | Disjoint Pairs | File Size |
|----------|--------------------------------|---------|-----------------|----------------|-----------|
| **CIM**  | Clinical Information Modeling  | 1,933   | 1931            | 0              | ~8 MB     |
| **OEO**  | Open Energy Ontology           | 1,565   | 1568            | 60             | ~4 MB     |
| **DOID** | Human Disease Ontology         | 14,735  | 17,274          | 1,428          | ~30 MB    |
| **GO**   | Gene Ontology                  | 51,937  | 58,529          | 29             | ~130 MB   |

### Ontology Selection Rationale

- **CIM**: Tree-like hierarchy with zero disjoint axioms — tests baseline containment learning
- **OEO**: Smallest ontology (1.5K classes) with moderate disjoint constraints — tests curriculum effect on trivially embeddable ontologies
- **DOID**: Medium-scale biomedical ontology with high disjoint density (1,428 pairs) — tests constraint handling under pressure; critical threshold case for curriculum learning
- **GO**: Largest ontology tested (6× DOID) — tests scalability

## Box Configuration (BoxConfig)

Standard configuration applied across all ontologies unless noted:

```python
@dataclass
class BoxConfig:
    # Geometry
    dim: int = 6                      # Embedding dimension (swept: 2-30)
    
    # Optimisation
    steps: int = 10_000               # Training iterations
    lr: float = 1.0 / sqrt(dim)       # Dimension-scaled learning rate
    seed: int = 42                    # Random seed for reproducibility
    
    # Regularization
    min_box_size: float = 0.05        # Minimum box extent per dimension
    size_weight: float = 0.1          # Weight for size regularization
    
    # Disjoint margin
    disjoint_margin: float = 0.02     # Minimum separation for disjoint boxes
    
    # Constraint weights
    subclass_weight: float = 1.0      # Structural foundation (second priority)
    disjoint_weight: float = 2.0      # Highest priority (critical constraint)
    big_box_weight: float = 0.1       # Low priority (volume control)
    distance_weight: float = 0.1      # Low priority (sibling separation)
    
    # Volume control
    base_log_volume: float = 2.0      # Target log-volume at root
    depth_scale: float = 0.5          # Depth-based volume scaling
    min_log_volume: float = -4.0      # Minimum allowed log-volume
```

### Weight Rationale

| Weight             | Value | Reasoning                                     |
|--------------------|-------|-----------------------------------------------|
| `subclass_weight`  | 1.0   | Hierarchy is foundational — high, but secondary to disjointness |
| `disjoint_weight`  | 2.0   | Highest priority — the critical constraint (mirrors the S1 scheduling finding) |
| `size_weight`      | 0.1   | Gentle regularization, not dominant           |
| `big_box_weight`   | 0.1   | Prevents degenerate large boxes               |
| `distance_weight`  | 0.1   | Encourages sibling dispersion without forcing |

---

## Curriculum Schedules

Curriculum learning phases in different loss components at different training stages.

### Standard Final Schedule

The final schedule used for all reported results (identical to **S1: Standard** below):

```python
CurriculumSchedule(
    subclass_start = 0.0,   # Active from first step
    disjoint_start = 0.01,  # Active from ~step 100 (1% of 10k)
    sibling_start = 0.5,    # Activated after 50% of training
    big_box_start = 0.7,    # Activated after 70% of training
    ramp = False            # Step function activation (on/off)
)
```

## Alternative Schedules Tested

### Schedule S0: Plain Learning (Baseline)

```python
# No curriculum — all loss terms active from step 0
# Used as baseline for comparison
```

**Tested on:** CIM, OEO, DOID, GO  
**Result:** Baseline performance; DOID: 779 violations (dim 6), 717 (dim 10); GO: 5,280 violations at dim 10

---

### Schedule S1: Standard (Recommended)

```python
CurriculumSchedule(
    subclass_start = 0.0,   # Active from first step
    disjoint_start = 0.01,  # Active from ~step 100 (1% of 10k)
    sibling_start = 0.5,    # Activated after 50% of training
    big_box_start = 0.7,    # Activated after 70% of training
    ramp = False            # Step function activation
)
```

**Tested on:** CIM, OEO, DOID, GO  
**Result:** **Optimal overall** — DOID: 0 violations at dim 10, 1 at dim 15 (vs. 717 plain at dim 10); GO: 30 violations at dim 10 vs. 5,280 for plain (99.4% reduction). CIM: 0 at every dimension. OEO: best is 2 violations at dim 2 for both methods.

---

### Schedule S2: Late Activation

```python
CurriculumSchedule(
    subclass_start = 0.0,
    disjoint_start = 0.5,   # Delayed to 50%
    sibling_start = 0.5,
    big_box_start = 0.5,
    ramp = False
)
```

**Tested on:** CIM, OEO, DOID, GO  
**Result:** **Catastrophic failure** — DOID: 3,515 violations at dim 10 (4.9× worse than plain); GO: 18,649 violations (3.5× worse than plain). CIM unaffected (0 violations — no disjoint axioms). OEO: 1,414 violations (6.2× worse than plain). Confirms that delaying disjoint constraints allows irreversible box overlap.

---

### Schedule S3: GO-Style Ramp

```python
CurriculumSchedule(
    subclass_start = 0.1,   # Subclass delayed slightly
    disjoint_start = 0.0,   # Disjoint from step 0
    sibling_start = 0.7,    # Late sibling separation
    big_box_start = 0.2,    # Early size control
    ramp = True             # Gradual linear interpolation
)
```

**Tested on:** CIM, OEO, DOID, GO  
**Result:** Near-optimal on DOID (1 violation at dims 10 and 15), excellent on GO (107 violations at dim 10 vs. 5,280 plain). Ramp activation helps large ontologies, but the step-activation standard schedule still wins (GO: 30 vs. 107 at dim 10).

---

### Schedule S4: Disjoint-Late

```python
CurriculumSchedule(
    subclass_start = 0.0,
    disjoint_start = 0.7,   # Delayed to 70%
    sibling_start = 0.3,
    big_box_start = 0.3,
    ramp = False
)
```

**Tested on:** CIM, OEO, DOID, GO  
**Result:** **Worst performance** — DOID: 3,160 violations at dim 10; GO: 19,926 violations.

---

### Schedule S5: Very Late (All Components)

```python
CurriculumSchedule(
    subclass_start = 0.0,
    disjoint_start = 0.8,   # Delayed to 80%
    sibling_start = 0.8,
    big_box_start = 0.8,
    ramp = False
)
```

**Tested on:** CIM, OEO, DOID, GO  
**Result:** Catastrophic — DOID: 3,447 violations at dim 10; GO: 18,222 violations. All late-activation schedules degrade severely; there is no fully consistent ordering among the late variants.

---

## Schedule Comparison Table

| Schedule             | Ontologies | Sub Start | Disj Start | Sib Start | Box Start | Ramp | Improvement                            |
|----------------------|------------|-----------|------------|-----------|-----------|------|----------------------------------------|
| **S0_plain**         | All        | 0.0       | 0.0        | 0.0       | 0.0       | No   | Baseline                               |
| **S1_standard**      | All        | 0.0       | 0.01       | 0.5       | 0.7       | No   | DOID 0 (d10) / 1 (d15); GO 30 at d10 (−99.4% vs plain); CIM 0 everywhere |
| **S2_late**          | All        | 0.0       | 0.5        | 0.5       | 0.5       | No   | ≤ 30.9× worse (DOID dim 2)             |
| **S3_go**            | All        | 0.1       | 0.0        | 0.7       | 0.2       | Yes  | Best ramp; behind S1 (107 vs 30, GO dim 10) |
| **S4_disjoint_late** | All        | 0.0       | 0.7        | 0.3       | 0.3       | No   | ≤ 13.6× worse (DOID dim 2)             |
| **S5_very_late**     | All        | 0.0       | 0.8        | 0.8       | 0.8       | No   | ≤ 35.1× worse (DOID dim 2)             |

**Key finding:** Disjoint constraints must activate early in training: only S1 (disjoint at 1%) reaches zero violations at higher dimensions, while every schedule delaying disjointness past 50% degrades severely (worst cells 31–35× worse than plain at DOID dim 2), with no consistent ordering among the late variants.

---

## Testing Protocol

Each schedule was evaluated across:

- **Dimensions:** 2–10, 15, 20, 25, 30 for all ontologies (DOID additionally 11–13, GO additionally 11–12)
- **Ontologies:** CIM, DOID, GO, OEO
- **Metrics:**
  - Subclass violations (absolute count + rate)
  - Disjoint violations (absolute count + rate)
  - Concluded relationship violations (entailed but non-asserted)
  - Average sibling distance
  - Box size distribution

**Selection criterion:** Final schedule chosen based on lowest total violations at optimal dimension, prioritizing subclass constraint satisfaction.

---
## Training Protocol

1. **Plain Learning** (`learn_boxes_from_owl`)
   - All loss components active throughout training
   - Used for: All ontologies
   - 10k steps
   
2. **Curriculum Learning** (`learn_boxes_with_curriculum`)
   - Loss components activated per schedule
   - Used for: All ontologies
   - 10k steps


## Evaluation Metrics

### Primary Metrics

1. **Subclass Violations**
   - Count of edges where `box_child ⊄ box_parent`
   - Computed as: `(min_parent > min_child) OR (max_child > max_parent)` in any dimension
   - Reported as: absolute count + rate (%)

2. **Disjoint Violations**
   - **Definition**: for each entailed disjoint pair $(a,b)$ — the asserted axioms plus all pairs inferred through the subclass hierarchy — let $s_d(a,b)$ be the separation between the two boxes in dimension $d$:
     $$s_d(a,b) = \max\left(\min_{b,d} - \max_{a,d},\ \min_{a,d} - \max_{b,d}\right)$$
     with $\min_{\cdot,d}$, $\max_{\cdot,d}$ the box lower/upper bounds in dimension $d$, and let $\sigma_{ab} = \max_d s_d$ be the best separation over all dimensions.
   - **Violated** iff $\sigma_{ab} < \texttt{disjoint\_margin}$ (0.02): the boxes are separated by at least the margin in *any one* dimension; overlap in the remaining dimensions is allowed.
   - **Metrics reported**:
     - *Violation count*: number of entailed pairs with $\sigma_{ab} < 0.02$ (absolute count + rate)
     - *Per-pair severity*: $\mathrm{softplus}(0.02 - \sigma_{ab})$ — continuous (0 for perfectly separated pairs), stored per axiom in `per_axiom_violations.json`
     - *Overlap volume*: $J_{ab} = \prod_d \max\left(0,\ \min(\max_{a,d}, \max_{b,d}) - \max(\min_{a,d}, \min_{b,d})\right)$ — positive iff the boxes intersect; margin-free stricter severity reading
   - **Entailed set**: evaluation deliberately covers **entailed** disjointness, not only asserted axioms — e.g. DOID's 1,428 asserted axioms expand to 4,629 entailed pairs — so the metric validates more than the surface axioms.

3. **Average Sibling Distance**
   - Mean gap between sibling boxes across all dimensions
   - Positive values = separation; negative values = overlap
   - Diagnostic metric (not optimized directly)

4. **Box Size**
   - Mean box size across all dimensions
   - Smaller values = smaller penalty

### Link Prediction (Concluded Relationships)

Evaluates generalization to **entailed but non-asserted** relationships:

- **Concluded Subclass**: Transitive closure edges not in training set
- **Concluded Disjoint**: Entailed disjoint pairs not explicitly asserted
- **Metric**: Violation count on concluded edges (proxy for link prediction)

---

## Experimental Variants
### Plain vs Curriculum Comparison

**Goal**: Measure impact of phased loss activation across ontology sizes

**Configuration**:
- Same BoxConfig for both methods
- Identical random seeds (42) with additional test on seeds 4, 666, 1010.
- Same dimension sweep per ontology (2–10/12/13, 15, 20, 25, 30)

**Ontologies**: CIM, OEO, DOID, GO

---

## Computational Resources
### Hardware

- **Platform**: Apple Silicon (M-series)
- **Acceleration**: MPS (Metal Performance Shaders)
- **Memory**: Up to 8GB peak (GO at dim 20)

### Loading Time and Training Time

| Ontology | Classes | Loading Time | Dim 2 (Plain) | Dim 2 (Curr) | Dim 30 (Plain) | Dim 30 (Curr) |
|----------|---------|--------------|---------------|--------------|----------------|---------------|
| CIM      | 1,933   | 2 second     | ~20 sec       | ~20 sec      | ~60 sec        | ~30 sec       |
| OEO      | 1,565   | <1 second    | ~30 sec       | ~20 sec      | ~25 sec        | ~20 sec       |
| DOID     | 14,735  | ~5 seconds   | ~45 sec       | ~40 sec      | ~70 sec        | ~60 sec       |
| GO       | 51,937  | ~25 seconds  | ~90 sec       | ~85 sec      | ~5 min         | ~3.5 min      |

**Notes**:
- Higher dimensions require more memory and longer computation. Also more steps are recommended.
- GO dominates total experiment time due to class count (52K).

### Software Stack

- **Python**: 3.11
- **PyTorch**: 2.11.0 (MPS backend)
- **OWL Loading**: rdflib

---

## Reproducibility

### Random Seeds

All experiments use `seed=42` for:
- Box initialization (Gaussian: μ=0, σ=0.1)
- Training randomness

Additional experiments (OEO and CIM) use `seed=4, 666, 1010` for:
- Box initialization (Gaussian: μ=0, σ=0.1)
- Training randomness
