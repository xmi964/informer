import pandas as pd
import os

# 注意：此处路径需改为你的实际路径！
input_path = 'data/sales/Online_Retail.csv'  # 输入数据路径
output_path = 'data/processed/daily_sales.csv'  # 输出路径

# 读取原始数据（注意编码）
df = pd.read_csv(input_path, encoding='ISO-8859-1')

# 处理步骤
# 1. 删除退货订单（Quantity > 0）
df = df[df['Quantity'] > 0]

# 2. 转换日期格式并提取日期部分（去掉具体时间）
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date  # 只保留日期部分

# 3. 筛选2010年的数据
df['Year'] = pd.to_datetime(df['InvoiceDate']).dt.year  # 提取年份
df = df[df['Year'] == 2010]  # 只保留2010年数据

# 4. 按日期+商品聚合销量
daily_sales = df.groupby(['InvoiceDate', 'StockCode']).agg({'Quantity': 'sum'}).reset_index()

# 保存预处理后的数据
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 自动创建processed文件夹
daily_sales.to_csv(output_path, index=False)

# 打印处理结果信息
print(f"预处理完成！输出文件：{output_path}")
print(f"数据时间范围：{daily_sales['InvoiceDate'].min()} 至 {daily_sales['InvoiceDate'].max()}")
print(f"总商品种类数：{daily_sales['StockCode'].nunique()}")
print(f"总记录数：{len(daily_sales)}")
print(f"示例数据：\n{daily_sales.head()}")