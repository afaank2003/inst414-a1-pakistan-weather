#!/usr/bin/env python3
"""
Pakistan Weather EDA (INST414 A1)
- Parses Date -> year, month, year-month
- Annual aggregates: mean temps/sunshine/wind; sum precipitation; count heat-wave days (Temp_Max >= threshold)
- Monthly climatology: average temperature and precipitation by calendar month across years
- Exploratory relationship: daily Sunshine_Duration vs Temp_Mean scatter
- Saves CSV summaries + PNG figures

Usage:
  python eda_pakistan_weather.py --input Pakistan_weather_data.csv --outdir outputs/

Dependencies:
  pandas, numpy, matplotlib
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["Date", "Temp_Max", "Temp_Min", "Temp_Mean",
                "Precipitation_Sum", "Sunshine_Duration",
                "Windspeed_Max", "Windgusts_Max"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].copy()
    df.sort_values("Date", inplace=True)
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["ym"] = df["Date"].dt.to_period("M").astype(str)
    return df


def compute_annual(df: pd.DataFrame, heat_threshold_c: float) -> pd.DataFrame:
    annual = df.groupby("year").agg({
        "Temp_Mean": "mean",
        "Temp_Max": "mean",
        "Temp_Min": "mean",
        "Precipitation_Sum": "sum",
        "Sunshine_Duration": "mean",
        "Windspeed_Max": "mean",
        "Windgusts_Max": "mean",
    }).reset_index()

    df["heatwave_day"] = df["Temp_Max"] >= float(heat_threshold_c)
    heatwaves = df.groupby("year")["heatwave_day"].sum().reset_index()
    heatwaves.rename(columns={"heatwave_day": f"heatwave_days_ge_{int(heat_threshold_c)}C"}, inplace=True)
    annual = annual.merge(heatwaves, on="year", how="left")
    return annual


def compute_monthly_climatology(df: pd.DataFrame) -> pd.DataFrame:
    monthly = df.groupby("month").agg({
        "Temp_Mean": "mean",
        "Temp_Max": "mean",
        "Temp_Min": "mean",
        "Precipitation_Sum": "mean",
        "Sunshine_Duration": "mean",
    }).reset_index()
    return monthly


def save_figures(annual: pd.DataFrame, monthly: pd.DataFrame, df: pd.DataFrame, outdir: Path, heat_threshold_c: float):
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(annual["year"], annual["Temp_Mean"], marker="o")
    plt.title("Pakistan — Annual Mean Temperature (°C)")
    plt.xlabel("Year")
    plt.ylabel("Temp_Mean (°C)")
    plt.savefig(outdir / "annual_mean_temp.png", bbox_inches="tight")
    plt.close()

    heat_col = f"heatwave_days_ge_{int(heat_threshold_c)}C"
    if heat_col in annual.columns:
        plt.figure()
        plt.bar(annual["year"], annual[heat_col])
        plt.title(f"Pakistan — Heatwave Days (Temp_Max ≥ {int(heat_threshold_c)}°C) per Year")
        plt.xlabel("Year")
        plt.ylabel("Days")
        plt.savefig(outdir / "annual_heatwave_days.png", bbox_inches="tight")
        plt.close()

    plt.figure()
    plt.plot(monthly["month"], monthly["Temp_Mean"], marker="o")
    plt.title("Pakistan — Seasonal Cycle of Mean Temperature (°C)")
    plt.xlabel("Month")
    plt.ylabel("Temp_Mean (°C)")
    plt.savefig(outdir / "seasonal_temp_mean.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.bar(monthly["month"], monthly["Precipitation_Sum"])
    plt.title("Pakistan — Seasonal Cycle of Precipitation (avg per day)")
    plt.xlabel("Month")
    plt.ylabel("Precipitation_Sum")
    plt.savefig(outdir / "seasonal_precip.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.scatter(df["Sunshine_Duration"], df["Temp_Mean"])
    plt.title("Sunshine Duration vs. Mean Temperature (Daily)")
    plt.xlabel("Sunshine_Duration")
    plt.ylabel("Temp_Mean (°C)")
    plt.savefig(outdir / "scatter_sunshine_vs_temp.png", bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="EDA for Pakistan Weather dataset (INST414 A1).")
    p.add_argument("--input", required=True, help="Path to Pakistan_weather_data.csv")
    p.add_argument("--outdir", default="eda_outputs", help="Directory to save figures and tables")
    p.add_argument("--heat-threshold-c", type=float, default=40.0, help="Heat-wave threshold on Temp_Max in °C")
    args = p.parse_args()

    inpath = Path(args.input)
    outdir = Path(args.outdir)
    thresh = float(args.heat_threshold_c)

    df = load_and_prepare(inpath)
    annual = compute_annual(df, thresh)
    monthly = compute_monthly_climatology(df)

    outdir.mkdir(parents=True, exist_ok=True)
    annual.to_csv(outdir / "annual_summary.csv", index=False)
    monthly.to_csv(outdir / "monthly_climatology.csv", index=False)

    save_figures(annual, monthly, df, outdir, thresh)

    hot_year = annual.loc[annual["Temp_Mean"].idxmax(), ["year", "Temp_Mean"]].to_dict()
    cool_year = annual.loc[annual["Temp_Mean"].idxmin(), ["year", "Temp_Mean"]].to_dict()
    wet_year = annual.loc[annual["Precipitation_Sum"].idxmax(), ["year", "Precipitation_Sum"]].to_dict()
    heat_col = f"heatwave_days_ge_{int(thresh)}C"
    hw_year = annual.loc[annual[heat_col].idxmax(), ["year", heat_col]].to_dict() if heat_col in annual.columns else {}

    print("=== SUMMARY ===")
    print(f"Hottest year (Temp_Mean): {int(hot_year['year'])} ~ {hot_year['Temp_Mean']:.2f} °C")
    print(f"Coolest year (Temp_Mean): {int(cool_year['year'])} ~ {cool_year['Temp_Mean']:.2f} °C")
    print(f"Wettest year (Precip sum): {int(wet_year['year'])} ~ {wet_year['Precipitation_Sum']:.1f}")
    if hw_year:
        k = list(hw_year.keys())
        print(f"Most heat-wave days (≥ {int(thresh)} °C): {int(hw_year['year'])} ~ {int(hw_year[heat_col])} days")
    print(f"Outputs saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
