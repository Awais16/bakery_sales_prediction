import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_missing_heatmap(df, name):
    # Get min and max dates
    min_date = df['Datum'].min()
    max_date = df['Datum'].max()
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # Create full date DataFrame
    df_full = pd.DataFrame({'Datum': full_date_range})
    
    # Merge to include all dates, left join to mark missing dates
    merged = df_full.merge(df, on='Datum', how='left')
    
    # Create missing matrix
    missing_matrix = merged.isnull().astype(int)
    missing_matrix.set_index(merged['Datum'], inplace=True)
    missing_matrix.index = missing_matrix.index.date
    
    # Transpose for horizontal bars
    plt.figure(figsize=(20, len(missing_matrix.columns) * 0.5))
    sns.heatmap(missing_matrix.T, cbar=False, cmap=['green', 'red'], linewidths=0, linecolor='white')
    plt.title(f'Missing Values Heatmap for {name} (Green: Present, Red: Missing)')
    plt.xlabel('Datum')
    plt.ylabel('Columns')
    plt.show()