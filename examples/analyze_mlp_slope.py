import pandas as pd
import numpy as np
from scipy import stats

def analyze_slope():
    csv_path = "save_mlp/loss_history_full.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: CSV not found.")
        return

    # Analyze last 5000 steps (approx last 25% of training)
    # Analyze last 20000 steps (80k to 100k)
    tail_df = df[df['iteration'] > 80000]
    
    if len(tail_df) < 10:
        print("Not enough data points.")
        return

    y = tail_df['loss'].values
    x = tail_df['iteration'].values
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    print(f"Analysis of last {len(tail_df)} steps (Iter 15000-20000):")
    print(f"Mean Loss: {np.mean(y):.6f}")
    print(f"Slope: {slope:.8f} (Change in loss per step)")
    print(f"R-squared: {r_value**2:.4f}")
    
    # Projected improvement
    current_loss = np.mean(y)
    target_loss = 0.0006
    
    if slope >= 0:
        print("\n--- Projection ---")
        print("Slope is positive/zero. It will NEVER reach the target.")
    else:
        steps_needed = (target_loss - current_loss) / slope
        total_steps = 50000 + steps_needed
        
        # Speed: 50k steps took ~8 minutes (480 seconds) -> ~104 steps/sec
        steps_per_sec = 104
        seconds_needed = steps_needed / steps_per_sec
        hours_needed = seconds_needed / 3600
        days_needed = hours_needed / 24
        
        # DeepSet Time
        deepset_time_hours = 3.5
        
        print(f"\n--- Comparison Projection ---")
        print(f"Target Loss: {target_loss}")
        print(f"Current MLP Loss: {current_loss:.6f}")
        print(f"MLP Improvement Rate: {slope:.10f} / step")
        print(f"Steps needed to reach target: {int(steps_needed):,}")
        print(f"Total MLP Time Needed: {hours_needed:.2f} hours")
        print(f"DeepSet Time: {deepset_time_hours} hours")
        
        if hours_needed < deepset_time_hours:
            print("Result: MLP would be FASTER if it maintained this rate.")
        else:
            print(f"Result: MLP would be SLOWER ({hours_needed/deepset_time_hours:.1f}x slower).")

if __name__ == "__main__":
    analyze_slope()
