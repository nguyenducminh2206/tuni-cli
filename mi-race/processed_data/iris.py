from datasets import load_dataset

ds = load_dataset("hitorilabs/iris")

ds.set_format(type='pandas')

df = ds['train'][:]
df.to_csv('processed_data/iris_ds.csv')

print(df.head())