# Krishang Jain
# Beginner Assignment

# -------------------------------

# 1. Basic Python Programming:

print("Hello, World!")

num1 = 10  
num2 = 5  
num3 = 3.5
text = "Python"

sum_result = num1 + num2  
diff_result = num1 - num2  
prod_result = num1 * num2  
quot_result = num1 / num2  

print("Sum:", sum_result)
print("Difference:", diff_result)
print("Product:", prod_result)
print("Quotient:", quot_result)

# -------------------------------

# 2. Control Structures:

print("For Loop Example:")
for i in range(1, 6):
    print(i)

print("While Loop Example:")
num = 5
while num > 0:
    print(num)
    num -= 1

print("Conditional Statement Example:")
n = 7 
if n % 2 == 0:
    print(n, "is even.")
else:
    print(n, "is odd.")

# -------------------------------

# 3. NumPy Essentials:

import numpy as np

arr = np.arange(1,11)
print(arr)

arr2 = np.random.randint(1,10,size=(3,3))
arr3 = np.random.randint(1,10,size=(3,3))

print(arr2+arr3)
print(arr2-arr3)
print(arr2*arr3)
print(arr2/arr3)

print(arr.mean())
print(arr.sum())
print(np.median(arr))

print(arr2.mean())
print(arr2.sum())
print(np.median(arr2))

# -------------------------------

# 4. Data Structures with Pandas:

import pandas as pd

df = pd.read_csv("Movies.csv")
print(df)

df_filter = df[df['Rating'] > 7.5]
print(df_filter)

df_sort = df.sort_values('Rating',ascending=False)
print(df_sort)

df_index = df.set_index('Movie Name')
print(df_index)

# -------------------------------

# 5. Basic Data Analysis:

import numpy as np

print(df['Rating'].mean())
print(df['Rating'].median())
print(df['Rating'].std())

print(np.mean(df['Rating'].values))
print(np.median(df['Rating'].values))
print(np.std(df['Rating'].values))

df_genres = df.groupby('Genre')['Rating'].mean()
print(df_genres)

# -------------------------------

# 6. Matplotlib Basics:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.plot(df['Movie Name'],df['Rating'])
plt.xlabel("Movie Names")
plt.ylabel("Rating")
plt.title("Trends")
plt.show()

plt.scatter(df['Genre'],df['Rating'])
plt.xlabel("Genres")
plt.ylabel("Rating")
plt.title("Trends")
plt.show()

# -------------------------------

# 7. Data Visualization with Seaborn:

import seaborn as sns

sns.boxplot(x=df['Genre'],y=df['Rating'])
plt.title("Trends")
plt.show()

sns.histplot(df['Rating'])
plt.title("Trends")
plt.show()

sns.kdeplot(df["Rating"])  
plt.title("Trends")  
plt.show()

# -------------------------------

# 8. Time Series Data:

df["Release Year"] = [2018, 2019, 2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023]
print(df)

df["Release Year"] = pd.to_datetime(df["Release Year"], format='%Y')
df_time_series = df.groupby("Release Year").size().reset_index(name="Movie Count")
df_time_series.set_index("Release Year", inplace=True)
print(df_time_series)

plt.plot(df_time_series.index, df_time_series["Movie Count"], marker='o', linestyle='-')
plt.xlabel("Year")
plt.ylabel("Number of Movies Released")
plt.title("Movies Released Per Year")
plt.show()

# -------------------------------

# 9. Correlation Analysis:

df["Box Office (millions)"] = [500, 120, 80, 200, 350, 50, 150, 90, 220, 180]

corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap")
plt.show()

# -------------------------------

# 10. Data Aggregation:

avg_rating = df.groupby("Genre")["Rating"].mean().reset_index()

sns.barplot(x=avg_rating["Genre"], y=avg_rating["Rating"])
plt.xlabel("Genre")
plt.ylabel("Average Rating")
plt.title("Average Movie Ratings by Genre")
plt.show()

# -------------------------------

# 11. Data Cleaning:

df["Budget"] = [50, np.nan, 80, 100, np.nan, 60, np.nan, 90, 120, 70]

print(df.isnull().sum())
df["Budget"].fillna(df["Budget"].mean(), inplace=True)
print(df.isnull().sum())

# -------------------------------

# 12. Combining Plots:

genre_counts = df["Genre"].value_counts()

fig, axes = plt.subplots(nrows=3, ncols=1)

axes[0].plot(df["Movie Name"], df["Rating"])
axes[0].set_title("Movie Ratings")
axes[0].set_xlabel("Movie Name")
axes[0].set_ylabel("Rating")

axes[1].bar(genre_counts.index, genre_counts.values)
axes[1].set_title("Number of Movies per Genre")
axes[1].set_xlabel("Genre")
axes[1].set_ylabel("Count")

axes[2].scatter(df["Budget"], df["Rating"])
axes[2].set_title("Budget vs. Rating")
axes[2].set_xlabel("Budget")
axes[2].set_ylabel("Rating")

plt.tight_layout()
plt.show()

# -------------------------------

# 13. Custom Visualization:

x = np.arange(1, 6)  # [1, 2, 3, 4, 5]
y1 = [10, 15, 7, 12, 18]
y2 = [5, 12, 9, 14, 20]

fig, axes = plt.subplots(nrows=3, ncols=1)

axes[0].plot(x, y1, color='red', linestyle='--', marker='o', label="Series 1")
axes[0].plot(x, y2, color='blue', linestyle='-', marker='s', label="Series 2")
axes[0].set_title("Customized Line Plot")
axes[0].set_xlabel("X-Axis Label")
axes[0].set_ylabel("Y-Axis Label")
axes[0].legend()  
axes[0].grid(True, linestyle='--', alpha=0.6)  

axes[1].bar(x, y1, color='purple', label="Series 1", alpha=0.7)
axes[1].bar(x, y2, color='orange', label="Series 2", alpha=0.7)
axes[1].set_title("Customized Bar Chart")
axes[1].set_xlabel("X-Axis Label")
axes[1].set_ylabel("Y-Axis Label")
axes[1].legend()

axes[2].scatter(x, y1, color='green', marker='D', s=100, label="Data Points")
axes[2].set_title("Customized Scatter Plot")
axes[2].set_xlabel("X-Axis Label")
axes[2].set_ylabel("Y-Axis Label")
axes[2].legend()
axes[2].set_facecolor('#f5f5f5') 

plt.tight_layout()
plt.show()

# -------------------------------

# 14. Exploratory Data Analysis (EDA):

df = sns.load_dataset("titanic")
print(df.head())

print(df.isnull().sum())

df["age"].fillna(df["age"].median(), inplace=True)

df.dropna(subset=["embark_town"], inplace=True)

print(df.describe())

sns.histplot(df["age"])
plt.title("Age Distribution of Titanic Passengers")
plt.show()

sns.heatmap(df.corr(numeric_only=True))
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------

# 15. Mini Project:

print(df.head())     
print(df.info())       
print(df.describe())  

df.drop(columns=["deck"], inplace=True) 

sns.histplot(df["age"])
plt.title("Age Distribution of Titanic Passengers")
plt.show()

sns.heatmap(df.corr(numeric_only=True))
plt.title("Correlation Heatmap")
plt.show()

fig, axes = plt.subplots(nrows=3, ncols=1)

sns.lineplot(x=df["class"], y=df["fare"], ax=axes[0])
axes[0].set_title("Average Fare by Class")

sns.barplot(data=df, x="sex", y="survived", ax=axes[1])
axes[1].set_title("Survival Rate by Gender")

sns.scatterplot(data=df, x="age", y="fare", hue="survived", ax=axes[2])
axes[2].set_title("Age vs. Fare with Survival Status")

plt.tight_layout()
plt.show()
