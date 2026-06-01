import pandas as pd
import json

df = pd.read_excel(r'c:\Users\GIA KHANG\Năm 3\kì 2\nhận dạng\Final_Project\code\dataset\CLN_SG_V2.xlsx')

print("=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(list(df.columns))

print("\n=== DTYPES ===")
print(df.dtypes.to_string())

print("\n=== HEAD (10) ===")
print(df.head(10).to_string())

print("\n=== TAIL (5) ===")
print(df.tail(5).to_string())

print("\n=== DESCRIBE ===")
print(df.describe(include='all').to_string())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum().to_string())

print("\n=== NULL PERCENTAGE ===")
print((df.isnull().sum() / len(df) * 100).to_string())

print("\n=== UNIQUE VALUES PER COLUMN ===")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

print("\n=== INDEX INFO ===")
print(df.index)
