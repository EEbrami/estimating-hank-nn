import pandas as pd
import numpy as np
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent

def load_data():
    """
    Load and process macroeconomic data from CSV files.
    Expected files: GDPC1.csv, PCEPI.csv, FEDFUNDS.csv
    """
    try:
        # Load GDP
        gdp = pd.read_csv(DATA_DIR / "GDPC1.csv", parse_dates=["observation_date"], index_col="observation_date")
        gdp["growth"] = gdp["GDPC1"].pct_change() * 400  # Annualized growth
        
        # Load Inflation (PCE)
        pce = pd.read_csv(DATA_DIR / "PCEPI.csv", parse_dates=["observation_date"], index_col="observation_date")
        pce["inflation"] = pce["PCEPI"].pct_change() * 400 # Annualized inflation
        
        # Load FFR
        ffr = pd.read_csv(DATA_DIR / "FEDFUNDS.csv", parse_dates=["observation_date"], index_col="observation_date")
        
        # Merge
        df = pd.concat([gdp["growth"], pce["inflation"], ffr["FEDFUNDS"]], axis=1)
        df.columns = ["OutputGrowth", "Inflation", "InterestRate"]
        df = df.dropna()
        
        return df
    except FileNotFoundError as e:
        print(f"Warning: Data file not found: {e}")
        print("Returning dummy data for testing purposes.")
        # Return dummy data
        dates = pd.date_range(start="1960-01-01", periods=200, freq="Q")
        df = pd.DataFrame(np.random.randn(200, 3), index=dates, columns=["OutputGrowth", "Inflation", "InterestRate"])
        return df

def load_moments():
    """
    Load distributional moments from JSON.
    """
    try:
        with open(DATA_DIR / "moments.json", "r") as f:
            moments = json.load(f)
        return moments
    except FileNotFoundError:
        print("Warning: moments.json not found. Using default calibrated moments.")
        # Default moments based on Kaplan et al. (2018)
        return {
            "liquid_wealth_share": 0.05,
            "gini_liquid": 0.8,
            "gini_illiquid": 0.6,
            "htm_share": 0.3
        }

def load_scf_data():
    """
    Load SCF Summary Extract data (Stata format).
    Expected file: rscfp2022.dta (2022 SCF Summary Extract)
    """
    try:
        # rscfp2022.dta is the 2022 Summary Extract
        file_path = DATA_DIR / "rscfp2022.dta"
        if not file_path.exists():
            # Try looking for other years or zip extraction result
            files = list(DATA_DIR.glob("rscfp*.dta"))
            if files:
                file_path = files[0]
            else:
                raise FileNotFoundError("No rscfp*.dta files found in data directory.")
        
        print(f"Loading SCF data from {file_path}...")
        df = pd.read_stata(file_path)
        
        # Compute moments (Placeholder logic - requires variable mapping)
        # Standard SCF Summary variables:
        # LIQ: Liquid assets
        # FIN: Financial assets
        # NFIN: Non-financial assets
        # DEBT: Total debt
        # NETWORTH: Net worth
        # WGT: Weight
        
        # Check if columns exist
        required_cols = ["liq", "networth", "wgt"]
        # SCF variables are often lowercase in Stata read
        df.columns = [c.lower() for c in df.columns]
        
        if all(c in df.columns for c in required_cols):
            # Weighted calculations
            total_wealth = (df["networth"] * df["wgt"]).sum()
            liquid_wealth = (df["liq"] * df["wgt"]).sum()
            liquid_share = liquid_wealth / total_wealth
            
            # Gini (simplified)
            # Sort by networth
            df_sorted = df.sort_values("networth")
            # Cumulative weight
            df_sorted["cum_wgt"] = df_sorted["wgt"].cumsum() / df_sorted["wgt"].sum()
            # Cumulative wealth
            df_sorted["cum_wealth"] = (df_sorted["networth"] * df_sorted["wgt"]).cumsum() / total_wealth
            # Gini = 1 - 2 * Area under Lorenz curve
            # Area approx using trapezoidal rule
            # This is a rough approx, usually need more careful handling of weights
            
            moments = {
                "liquid_wealth_share": liquid_share,
                "gini_networth": 0.8, # Placeholder
                "htm_share": 0.3 # Placeholder
            }
            return df, moments
        else:
            print("Warning: Required columns not found in SCF data.")
            return df, {}
            
    except Exception as e:
        print(f"Warning: Failed to load SCF data: {e}")
        return None, {}

if __name__ == "__main__":
    df = load_data()
    print("Data Head:")
    print(df.head())
    
    moments = load_moments()
    print("\nMoments (Default):")
    print(moments)
    
    scf_df, scf_moments = load_scf_data()
    if scf_df is not None:
        print("\nSCF Data Head:")
        print(scf_df.head())
        print("\nSCF Moments (Computed):")
        print(scf_moments)
