#!/usr/bin/env python3
"""
Schedule Ablation Study — All 4 Ontologies

Tests 5 schedule variants across CIM, OEO, DOID, GO at dimensions 2, 5, 10, 15.
Total: 4 ontologies × 5 schedules × 4 dims = 80 runs

Usage:
    cd /Users/mo9728/dev/loss_function_penalty
    .venv/bin/python3 run_schedule_ablation.py
"""
import sys
import os
from datetime import timedelta
import time
import json

sys.path.insert(0, '.')

from functions import (
    sweep_dimensions,
    learn_boxes_from_owl,
    learn_boxes_with_curriculum,
    BoxConfig,
    CurriculumSchedule,
)

# =============================================================================
# Configuration
# =============================================================================

ONTOLOGIES = {
    'cim': 'data/TheCimOntology.owl',
    'oeo': 'data/oeo-full.owl',
    'doid': 'data/doid.owl',
    'go': 'data/go.owl',
}

OUTPUT_BASE = "saved/schedule_ablation_all"

# Dimensions to test
DIMS = [2, 5, 10, 15]

# Schedules to evaluate
SCHEDULES = {
    # BASELINE: No curriculum at all (plain learning)
    'S0_plain': None,
    
    # All constraints from step 1 (curriculum but no phasing)
    'S1_standard': CurriculumSchedule(
        subclass_start=0.0,
        disjoint_start=0.01,
        sibling_start=0.5,
        big_box_start=0.7,
        ramp=False,
    ),
    
    # Standard: disjoint + regularization delayed to 50%
    'S2_late': CurriculumSchedule(
        subclass_start=0.0,
        disjoint_start=0.5,
        sibling_start=0.5,
        big_box_start=0.5,
        ramp=False,
    ),
    
    # GO-style: disjoint starts at 30%, with gradual ramp
    'S3_go': CurriculumSchedule(
        subclass_start=0.1,
        disjoint_start=0.0,
        big_box_start=0.2,
        sibling_start=0.7,
        ramp=True,
    ),
    
    # Aggressive delay: disjoint waits until 70%
    'S4_disjoint_late': CurriculumSchedule(
        subclass_start=0.0,
        disjoint_start=0.7,
        sibling_start=0.3,
        big_box_start=0.3,
        ramp=False,
    ),
    
    # Very late: everything but subclass delayed to 80%
    'S5_very_late': CurriculumSchedule(
        subclass_start=0.0,
        disjoint_start=0.8,
        sibling_start=0.8,
        big_box_start=0.8,
        ramp=False,
    ),
}

# =============================================================================
# Main execution
# =============================================================================

def main():
    print("=" * 100)
    print("SCHEDULE ABLATION STUDY — ALL ONTOLOGIES")
    print("=" * 100)
    print(f"\nOntologies: {list(ONTOLOGIES.keys())}")
    print(f"Dimensions: {DIMS}")
    print(f"Schedules: {list(SCHEDULES.keys())}")
    print(f"Total runs: {len(ONTOLOGIES) * len(SCHEDULES) * len(DIMS)}")
    print(f"Output: {OUTPUT_BASE}")
    print("\n" + "=" * 100)
    
    all_results = {}
    
    for onto_name, owl_path in ONTOLOGIES.items():
        print(f"\n{'='*100}")
        print(f"ONTOLOGY: {onto_name.upper()} ({owl_path})")
        print(f"{'='*100}")
        
        onto_results = {}
        
        for sched_name, schedule in SCHEDULES.items():
            print(f"\n{'-'*80}")
            print(f"SCHEDULE: {sched_name}")
            print(f"{'-'*80}")
            
            if schedule is None:
                print("  → Plain learning (NO curriculum)")
                learn_fn = learn_boxes_from_owl
            else:
                print(f"  → Curriculum: subclass@{schedule.subclass_start:.0%}, disjoint@{schedule.disjoint_start:.0%}, "
                      f"sibling@{schedule.sibling_start:.0%}, big_box@{schedule.big_box_start:.0%}, ramp={schedule.ramp}")
                learn_fn = learn_boxes_with_curriculum
            
            # Create output directory for this ontology + schedule
            sched_output_dir = os.path.join(OUTPUT_BASE, onto_name, sched_name)
            os.makedirs(sched_output_dir, exist_ok=True)
            
            # Run sweep across dimensions
            start_time = time.time()
            
            try:
                results = sweep_dimensions(
                    owl_path=owl_path,
                    learn_fn=learn_fn,
                    dims=tuple(DIMS),
                    cfg=BoxConfig(
                        steps=10000,
                        seed=42,
                        size_weight=0.1,
                        subclass_weight=1.0,
                        disjoint_weight=2.0,
                        big_box_weight=0.1,
                        depth_scale=0.5,
                        distance_weight=0.1,
                    ),
                    schedule=schedule,
                    path=sched_output_dir,
                )
                
                elapsed = time.time() - start_time
                
                # Store results
                onto_results[sched_name] = results
                
                # Print summary
                print(f"\n  ✓ Completed in {timedelta(seconds=int(elapsed))} ({elapsed:.1f}s)")
                print(f"\n  Results by dimension:")
                for dim in DIMS:
                    if dim in results:
                        r = results[dim]
                        print(f"    Dim {dim:2d}: sub_viol={r.get('sub_viol', '?'):>6}, "
                              f"dis_viol={r.get('dis_viol', '?'):>6}, "
                              f"loss={r.get('loss', float('inf')):.4f}")
                
            except Exception as e:
                print(f"\n  ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                onto_results[sched_name] = None
        
        all_results[onto_name] = onto_results
    
    # =============================================================================
    # Summary Tables
    # =============================================================================
    
    print("\n" + "=" * 100)
    print("SUMMARY TABLES")
    print("=" * 100)
    
    for onto_name, onto_results in all_results.items():
        print(f"\n{'='*80}")
        print(f"{onto_name.upper()}")
        print(f"{'='*80}")
        
        print(f"\n{'Schedule':<20} | {'Dim 2':<12} | {'Dim 5':<12} | {'Dim 10':<12} | {'Dim 15':<12}")
        print(f"{'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
        
        for sched_name in SCHEDULES.keys():
            results = onto_results.get(sched_name)
            if results is None:
                print(f"{sched_name:<20} | {'ERROR':<12} | {'ERROR':<12} | {'ERROR':<12} | {'ERROR':<12}")
                continue
            
            def fmt_dim(dim):
                if dim not in results:
                    return "N/A"
                r = results[dim]
                total_viol = r.get('sub_viol', 0) + r.get('dis_viol', 0)
                return f"Total: {total_viol:>6}"
            
            row = f"{sched_name:<20} | {fmt_dim(2):<12} | {fmt_dim(5):<12} | {fmt_dim(10):<12} | {fmt_dim(15):<12}"
            print(row)
    
    # =============================================================================
    # Save JSON Summary
    # =============================================================================
    
    summary_path = os.path.join(OUTPUT_BASE, "summary.json")
    json_summary = {}
    
    for onto_name, onto_results in all_results.items():
        json_summary[onto_name] = {}
        for sched_name, results in onto_results.items():
            if results is None:
                json_summary[onto_name][sched_name] = {"error": "failed"}
                continue
            
            json_summary[onto_name][sched_name] = {}
            for dim, r in results.items():
                json_summary[onto_name][sched_name][f"dim_{dim}"] = {
                    "subclass_violations": r.get('sub_viol', 0),
                    "disjoint_violations": r.get('dis_viol', 0),
                    "total_violations": r.get('sub_viol', 0) + r.get('dis_viol', 0),
                    "loss": r.get('loss'),
                }
    
    with open(summary_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {OUTPUT_BASE}/")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
