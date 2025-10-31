import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


def main():
    """
    Main function to run the Dask analysis on the New York cars dataset.
    """

    #
    # 1️⃣ Load dataset using Dask
    try:
        # Specify dtypes explicitly to avoid inference errors for problematic columns
        df = dd.read_csv("New_York_cars.csv", dtype={'1-owner vehicle': 'object',
                                                              'Accidents or damage': 'object',
                                                              'Clean title': 'object',
                                                              'Mileage': 'float64',
                                                              'Personal use only': 'object',
                                                              'money': 'object',  # Read money as object initially
                                                              'Year': 'object',  # Read Year as object initially
                                                              'new&used': 'object'
                                                              })
    except FileNotFoundError:
        print("Error: 'New_York_cars.csv' not found. Please ensure the dataset is in the same directory as the script.")
        return


    # 2️ Basic Cleaning

    # Drop rows with missing essential data
    df = df.dropna(subset=["brand", "money", "Year", "new&used"])

    # Convert 'money' and 'Year' to numeric after dropping NaNs, coercing errors
    df["money"] = df["money"].astype(str).str.replace(r'[$,]', '', regex=True)  # Remove currency symbols if present
    df["money"] = df["money"].map_partitions(pd.to_numeric,
                                             errors='coerce')  # Use map_partitions with pd.to_numeric for robustness
    df["Year"] = df["Year"].map_partitions(pd.to_numeric, errors='coerce').astype(
        'Int64')  # Use map_partitions with pd.to_numeric and then astype for Int64

    # Drop any rows that resulted in NaN after coercion for 'money' and 'Year'
    df = df.dropna(subset=["money", "Year"])

    print("Data loaded and cleaned successfully.")


    #  Analysis 1: Average Car Price by Brand

    print("Performing Analysis 1: Average Car Price by Brand...")
    # Group by brand, calculate mean price, compute the result, and sort
    avg_price_by_brand = df.groupby("brand")["money"].mean().compute().sort_values(ascending=False)

    # Visualization - Limit to top 10 brands and increase figure size
    plt.figure(figsize=(16, 9))
    sns.barplot(x=avg_price_by_brand.index[:10], y=avg_price_by_brand.values[:10], palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 10 Most Expensive Car Brands (Average Price)")
    plt.ylabel("Average Price ($)")
    plt.xlabel("Brand")
    plt.tight_layout()

    plt.show()
    # print(f"Plot saved to {plot_path}")

    # Insight
    print("\nTop 5 Most Expensive Brands:\n", avg_price_by_brand.head(5))


    #  Analysis 2: Average Price by Year

    print("\nPerforming Analysis 2: Average Price by Year...")
    avg_price_by_year = df.groupby("Year")["money"].mean().compute().sort_index()

    # Visualization - Increase figure size
    plt.figure(figsize=(14, 8))
    sns.lineplot(x=avg_price_by_year.index, y=avg_price_by_year.values, marker="o", color="royalblue")
    plt.title("Average Car Price by Year")
    plt.ylabel("Average Price ($)")
    plt.xlabel("Year")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.show()

    # Insight
    print("\nAverage Price by Latest Years:\n", avg_price_by_year.tail(5))


    # 5️⃣ Analysis 3: Count of New vs Used Cars

    print("\nPerforming Analysis 3: Count of New vs Used Cars...")
    # Convert to lowercase before value_counts
    car_condition = df["new&used"].str.lower().value_counts().compute()

    # Filter for only 'new' and 'used' categories
    car_condition_filtered = car_condition[car_condition.index.isin(['new', 'used'])]

    # Visualization - Increase figure size
    plt.figure(figsize=(9, 6))
    sns.barplot(x=car_condition_filtered.index, y=car_condition_filtered.values, palette="pastel")
    plt.title("New vs Used Cars Count")
    plt.ylabel("Count")
    plt.xlabel("Condition")
    plt.tight_layout()

    plt.show()

    # Insight
    print("\nNew vs Used Cars Count:\n", car_condition_filtered)


    # 6️⃣ Analysis 4: Top 10 Most Expensive Models

    print("\nPerforming Analysis 4: Top 10 Most Expensive Models...")
    # Use .nlargest() to find top 10 rows by 'money', select columns, and compute
    top_expensive_models = df.nlargest(10, "money")[["brand", "Model", "money"]].compute()

    # Visualization - Increase figure size
    plt.figure(figsize=(16, 9))
    sns.barplot(x="Model", y="money", data=top_expensive_models, hue="brand", dodge=False, palette="magma")
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 10 Most Expensive Car Models")
    plt.ylabel("Price ($)")
    plt.xlabel("Model")
    plt.tight_layout()

    plt.show()

    # Insight
    print("\nTop 10 Most Expensive Car Models:\n", top_expensive_models)

    # Insights Summary

    print("Top Insights Summary")

    print("1. The most expensive car brands on average are dominated by luxury and high-performance manufacturers.")
    print("2. There is a clear trend of increasing average car prices for newer models, especially from 2020 onwards.")
    print("3. The market contains a substantial number of used vehicles, indicating a healthy resale market.")
    print("4. The most expensive individual car listings are from premium brands like Tesla and Mercedes-Benz.")



if __name__ == "__main__":
    main()