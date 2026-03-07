import pandas as pd, numpy as np
df = pd.read_csv('standardized_data.csv')

for ft in ['E','H']:
    sub = df[df['Field_Type']==ft]
    grp = sub.groupby(['Location','Profile_Type','Circuit_ID','Distance'])['Field_Value']
    stats = grp.agg(['mean','std','count','min','max'])
    # Count groups with more than 1 observation
    multi = stats[stats['count'] > 1]
    single = stats[stats['count'] == 1]
    print(f'\n{ft}-field: {len(stats)} groups, {len(multi)} multi-obs, {len(single)} single-obs')
    if len(multi) > 0:
        std_vals = multi['std'].dropna()
        print(f'  Within-group std: mean={std_vals.mean():.3f}, max={std_vals.max():.3f}')
        cv = std_vals / (multi['mean'] + 1e-8) * 100
        print(f'  Within-group CV: mean={cv.mean():.1f}%, max={cv.max():.1f}%')
    print(f'  Total field std: {sub.Field_Value.std():.3f}, mean: {sub.Field_Value.mean():.3f}')
    print(f'  Mean group size: {stats["count"].mean():.1f}')
    
    # Show some multi-obs groups
    print('  Sample multi-obs groups:')
    for idx, row in multi.head(5).iterrows():
        print(f'    {idx} -> mean={row["mean"]:.3f} std={row["std"]:.3f} range=[{row["min"]:.3f}-{row["max"]:.3f}]')
