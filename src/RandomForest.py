import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

data = pd.read_excel(r'C:\Users\青山七海\PycharmProjects\MathModelProject\File\C题数据.xlsx', sheet_name="data1")
# 引号内改为自己文件的绝对路径
print(data.head())
# 检测是否读取成功
print(data.shape)
data.isnull().any().any()

sns.heatmap(data.astype(float).corr(), linewidths=0.4, vmax=1.0,
            square=True, cmap="RdBu_r", linecolor='k', annot=True)
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=12)
plt.rcParams['font.sans-serif'] = [font]  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决无法显示符号的问题
sns.set(font='SimHei', font_scale=0.8)  # 解决Seaborn中文显示问题
plt.title("测试", fontproperties=font)
plt.show()
