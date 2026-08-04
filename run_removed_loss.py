import sys
import os
from datetime import timedelta
import time

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

OWL_PATH = "data/doid.owl"
OUTPUT_BASE = "saved/doid/removed_loss"

# Dimensions to test — focus where we saw effects before
DIMS = [6]

# Schedules to evaluate — REAL VARIANCE in timing
SCHEDULES = {
    # BASELINE: No curriculum at all
    
    # All constraints from step 1 (curriculum but no phasing)
    
    # Standard: disjoint + regularization delayed to 50%
    'S1_no_sub': CurriculumSchedule(
        subclass_start=1.0,
        disjoint_start=0.0,
        sibling_start=0.0,
        big_box_start=0.0,
        ramp=False,
    ),

    # GO-style: disjoint starts at 30%, with gradual ramp (EXACT notebook match)
    'S2_no_disj': CurriculumSchedule(
        subclass_start=0.0,
        disjoint_start=1.0,
        big_box_start=0.0,
        sibling_start=0.0,
        ramp=False,
    ),
    
   # Aggressive delay: disjoint waits until 70%
    'S3_no_sibling': CurriculumSchedule(
        subclass_start=0.0,
        disjoint_start=0.0,
        sibling_start=1.0,
        big_box_start=0.0,
        ramp=False,
    ),
    
    # Very late: everything but subclass delayed to 80%
    'S4_no_big_box': CurriculumSchedule(
        subclass_start=0.0,
        disjoint_start=0.0,
        sibling_start=0.0,
        big_box_start=1.0,
        ramp=False,
    ),
}

# =============================================================================
# Main execution
# =============================================================================

def main():
    print("=" * 80)
    print("SCHEDULE ABLATION STUDY — OEO")
    print("=" * 80)
    print(f"\nOntology: {OWL_PATH}")
    print(f"Dimensions: {DIMS}")
    print(f"Schedules: {list(SCHEDULES.keys())}")
    print(f"Total runs: {len(SCHEDULES) * len(DIMS)}")
    print(f"Output: {OUTPUT_BASE}")
    print("\n" + "=" * 80)
    
    all_results = {}
    
    for sched_name, schedule in SCHEDULES.items():
        print(f"\n{'='*80}")
        print(f"SCHEDULE: {sched_name}")
        print(f"{'='*80}")
        
        if schedule is None:
            print("  → Plain learning (NO curriculum)")
            learn_fn = learn_boxes_from_owl
        else:
            print(f"  → Curriculum: subclass@{schedule.subclass_start:.0%}, disjoint@{schedule.disjoint_start:.0%}, "
                  f"sibling@{schedule.sibling_start:.0%}, "
                  f"big_box@{schedule.big_box_start:.0%}, "
                  f"ramp={schedule.ramp}")
            learn_fn = learn_boxes_with_curriculum
        
        # Create output directory for this schedule
        sched_output_dir = os.path.join(OUTPUT_BASE, sched_name)
        os.makedirs(sched_output_dir, exist_ok=True)
        
        # Run sweep across dimensions
        start_time = time.time()
        
        try:
            results = sweep_dimensions(
                owl_path=OWL_PATH,
                learn_fn=learn_fn,
                dims=tuple(DIMS),  # type: ignore
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
            all_results[sched_name] = results
            
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
            all_results[sched_name] = None
    
    # =============================================================================
    # Summary Table
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    print(f"\n{'Schedule':<20} | {'Dim 6':<20} | {'Dim 10':<20} | {'Dim 15':<20}")
    print(f"{'-'*20}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}")
    
    for sched_name in SCHEDULES.keys():
        results = all_results.get(sched_name)
        if results is None:
            print(f"{sched_name:<20} | {'ERROR':<20} | {'ERROR':<20} | {'ERROR':<20}")
            continue
        
        def fmt_dim(dim):
            if dim not in results:
                return "N/A"
            r = results[dim]
            total_viol = r.get('sub_viol', 0) + r.get('dis_viol', 0)
            return f"Total: {total_viol:>6}"
        
        row = f"{sched_name:<20} | {fmt_dim(6):<20} | {fmt_dim(10):<20} | {fmt_dim(15):<20}"
        print(row)
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_BASE}/")


if __name__ == "__main__":
    main()
