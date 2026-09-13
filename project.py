import pandas as pd

df = pd.read_csv("seasonal_agriculture_performance_dataset (3).csv")

print(df.head())


print(df.shape)
print(df.columns)

print(df.isnull().sum())

print(df.duplicated().sum())

print(df["Rainfall_mm"].mean())
print(df["Soil_Moisture_pct"].mean())
print(df["Yield_Tonnes_Ha"].mean())

df["Rainfall_mm"] = df["Rainfall_mm"].fillna(df["Rainfall_mm"].mean())

df["Soil_Moisture_pct"] = df["Soil_Moisture_pct"].fillna(df["Soil_Moisture_pct"].mean())

df["Yield_Tonnes_Ha"] = df["Yield_Tonnes_Ha"].fillna(df["Yield_Tonnes_Ha"].mean())

print(df.isnull().sum())

df.to_csv("cleaned_agriculture_data.csv", index=False)


season_profit = df.groupby("Season")["Profit_INR"].mean()

print(season_profit)

season_yield = df.groupby("Season")["Yield_Tonnes_Ha"].mean()

print(season_yield)

season_analysis = df.groupby("Season").agg({
    "Yield_Tonnes_Ha": "mean",
    "Profit_INR": "mean"
})

print(season_analysis)

import matplotlib.pyplot as plt

season_analysis["Profit_INR"].plot(kind="bar")

plt.title("Average Profit by Season")
plt.xlabel("Season")
plt.ylabel("Average Profit (INR)")
plt.xticks(rotation=0)

plt.show()

season_analysis["Yield_Tonnes_Ha"].plot(kind="bar")

plt.title("Average Yield by Season")
plt.xlabel("Season")
plt.ylabel("Average Yield (Tonnes/Ha)")
plt.xticks(rotation=0)

plt.show()

crop_profit = df.groupby("Crop")["Profit_INR"].mean()

print(crop_profit)

plt.figure(figsize=(10, 5))

crop_profit.plot(kind="bar")

plt.title("Average Profit by Crop")
plt.xlabel("Crop")
plt.ylabel("Average Profit (INR)")
plt.xticks(rotation=45)

plt.show()

crop_yield = df.groupby("Crop")["Yield_Tonnes_Ha"].mean()

print(crop_yield)

plt.figure(figsize=(10, 5))

crop_yield.plot(kind="bar")

plt.title("Average Yield by Crop")
plt.xlabel("Crop")
plt.ylabel("Average Yield (Tonnes/Ha)")
plt.xticks(rotation=45)

plt.show()

irrigation_profit = df.groupby("Irrigation_Method")["Profit_INR"].mean()

print(irrigation_profit)

plt.figure(figsize=(9, 5))

irrigation_profit.plot(kind="bar")

plt.title("Average Profit by Irrigation Method")
plt.xlabel("Irrigation Method")
plt.ylabel("Average Profit (INR)")
plt.xticks(rotation=0)

plt.show()

irrigation_yield = df.groupby("Irrigation_Method")["Yield_Tonnes_Ha"].mean()

print(irrigation_yield)
plt.figure(figsize=(9, 5))

irrigation_yield.plot(kind="bar")

plt.title("Average Yield by Irrigation Method")
plt.xlabel("Irrigation Method")
plt.ylabel("Average Yield (Tonnes/Ha)")
plt.xticks(rotation=0)

plt.show()

irrigation_water = df.groupby("Irrigation_Method")["Water_Used_m3"].mean()

print(irrigation_water)

plt.figure(figsize=(9, 5))

irrigation_water.plot(kind="bar")

plt.title("Average Water Used by Irrigation Method")
plt.xlabel("Irrigation Method")
plt.ylabel("Average Water Used (m³)")
plt.xticks(rotation=0)

plt.show()

irrigation_efficiency = df.groupby("Irrigation_Method")["Water_Efficiency_t_per_1000m3"].mean()

print(irrigation_efficiency)

plt.figure(figsize=(9, 5))

irrigation_efficiency.plot(kind="bar")

plt.title("Average Water Efficiency by Irrigation Method")
plt.xlabel("Irrigation Method")
plt.ylabel("Water Efficiency (Tonnes per 1000 m³)")
plt.xticks(rotation=0)

plt.show()
state_profit = df.groupby("State")["Profit_INR"].mean()

print(state_profit)
plt.figure(figsize=(10, 5))

state_profit.plot(kind="bar")

plt.title("Average Profit by State")
plt.xlabel("State")
plt.ylabel("Average Profit (INR)")
plt.xticks(rotation=45)

plt.show()

state_yield = df.groupby("State")["Yield_Tonnes_Ha"].mean()

print(state_yield)

plt.figure(figsize=(10, 5))

state_yield.plot(kind="bar")

plt.title("Average Yield by State")
plt.xlabel("State")
plt.ylabel("Average Yield (Tonnes/Ha)")
plt.xticks(rotation=45)

plt.show()

plt.figure(figsize=(9, 5))

plt.scatter(df["Rainfall_mm"], df["Profit_INR"])

plt.title("Rainfall vs Profit")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Profit (INR)")

plt.show()

rainfall_profit_corr = df["Rainfall_mm"].corr(df["Profit_INR"])

print("Correlation between Rainfall and Profit:", rainfall_profit_corr)
temperature_profit_corr = df["Avg_Temperature_C"].corr(df["Profit_INR"])

print("Correlation between Temperature and Profit:", temperature_profit_corr)

humidity_profit_corr = df["Humidity_pct"].corr(df["Profit_INR"])

print("Correlation between Humidity and Profit:", humidity_profit_corr)
soil_moisture_profit_corr = df["Soil_Moisture_pct"].corr(df["Profit_INR"])

print("Correlation between Soil Moisture and Profit:", soil_moisture_profit_corr)
water_profit_corr = df["Water_Used_m3"].corr(df["Profit_INR"])

print("Correlation between Water Used and Profit:", water_profit_corr)
yield_profit_corr = df["Yield_Tonnes_Ha"].corr(df["Profit_INR"])

print("Correlation between Yield and Profit:", yield_profit_corr)

correlation_summary = {
    "Rainfall vs Profit": df["Rainfall_mm"].corr(df["Profit_INR"]),
    "Temperature vs Profit": df["Avg_Temperature_C"].corr(df["Profit_INR"]),
    "Humidity vs Profit": df["Humidity_pct"].corr(df["Profit_INR"]),
    "Soil Moisture vs Profit": df["Soil_Moisture_pct"].corr(df["Profit_INR"]),
    "Water Used vs Profit": df["Water_Used_m3"].corr(df["Profit_INR"]),
    "Yield vs Profit": df["Yield_Tonnes_Ha"].corr(df["Profit_INR"])
}

for factor, correlation in correlation_summary.items():
    print(f"{factor}: {correlation:.3f}")
season_risk = df.groupby("Season")["Disease_Pest_Risk_pct"].mean()

print(season_risk)

plt.figure(figsize=(8, 5))

season_risk.plot(kind="bar")

plt.title("Average Disease/Pest Risk by Season")
plt.xlabel("Season")
plt.ylabel("Disease/Pest Risk (%)")
plt.xticks(rotation=0)

plt.show()

season_economic = df.groupby("Season").agg({
    "Total_Cost_INR": "mean",
    "Revenue_INR": "mean",
    "Profit_INR": "mean"
})

print(season_economic)

season_economic.plot(kind="bar", figsize=(10, 5))

plt.title("Average Cost, Revenue and Profit by Season")
plt.xlabel("Season")
plt.ylabel("Amount (INR)")
plt.xticks(rotation=0)

plt.show()

crop_economic = df.groupby("Crop").agg({
    "Total_Cost_INR": "mean",
    "Revenue_INR": "mean",
    "Profit_INR": "mean"
})

print(crop_economic)


crop_economic.plot(kind="bar", figsize=(12, 6))

plt.title("Average Cost, Revenue and Profit by Crop")
plt.xlabel("Crop")
plt.ylabel("Amount (INR)")
plt.xticks(rotation=45)

plt.show()

crop_water_efficiency = df.groupby("Crop")["Water_Efficiency_t_per_1000m3"].mean()

print(crop_water_efficiency)

plt.figure(figsize=(10, 5))

crop_water_efficiency.plot(kind="bar")

plt.title("Average Water Efficiency by Crop")
plt.xlabel("Crop")
plt.ylabel("Water Efficiency (Tonnes per 1000 m³)")
plt.xticks(rotation=45)

plt.show()

profitable_farms = (df["Profit_INR"] > 0).sum()
loss_farms = (df["Profit_INR"] < 0).sum()

print("Profitable Farms:", profitable_farms)
print("Loss-Making Farms:", loss_farms)

labels = ["Profitable Farms", "Loss-Making Farms"]
values = [profitable_farms, loss_farms]

plt.figure(figsize=(7, 7))

plt.pie(values, labels=labels, autopct="%1.1f%%")

plt.title("Profit vs Loss-Making Farms")

plt.show()

season_loss_percentage = (
    df.assign(Loss=df["Profit_INR"] < 0)
      .groupby("Season")["Loss"]
      .mean() * 100
)

print(season_loss_percentage)

plt.figure(figsize=(8, 5))

season_loss_percentage.plot(kind="bar")

plt.title("Loss-Making Farms by Season")
plt.xlabel("Season")
plt.ylabel("Loss-Making Farms (%)")
plt.xticks(rotation=0)

plt.show()


crop_loss_percentage = (
    df.assign(Loss=df["Profit_INR"] < 0)
      .groupby("Crop")["Loss"]
      .mean() * 100
)

print(crop_loss_percentage)


plt.figure(figsize=(10, 5))

crop_loss_percentage.plot(kind="bar")

plt.title("Loss-Making Farms by Crop")
plt.xlabel("Crop")
plt.ylabel("Loss-Making Farms (%)")
plt.xticks(rotation=45)

plt.show()