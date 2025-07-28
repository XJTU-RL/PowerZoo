import pandas as pd
import os

df = pd.read_csv('LoadShape3.CSV')

# 将所有元素设置为0.5
df[:] = 0.8

# 保存回CSV文件
df.to_csv('LoadShape3.CSV', index=False)