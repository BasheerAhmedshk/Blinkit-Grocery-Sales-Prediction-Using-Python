import pandas as pd

excel_file = 'BlinkIT.xlsx'
csv_file = 'blinkit_sales_data.csv'

df = pd.read_excel(excel_file)
df.to_csv(csv_file, index=False)

print(f'Successfully converted {excel_file} to {csv_file}')

