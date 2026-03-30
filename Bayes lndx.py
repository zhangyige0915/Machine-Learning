import pymysql
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

# 显示完整数据，不显示省略号
pd.set_option('display.max_columns', None)

# 数据库连接
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='20040915',
    database='lndx',
    charset='utf8mb4'
)

df = pd.read_sql("select * from student", conn)
conn.close()

# 标签：学历二分类
df['label'] = df['seducation'].apply(lambda x: 1 if x in ['硕士', '博士'] else 0)

# 特征：性别（编码为 0/1）
le = LabelEncoder()
df['sex_code'] = le.fit_transform(df['ssex'])  # 男1女0

X = df[['sex_code']]  # 只用性别做特征
y = df['label']
print(X.shape, y.shape)
print(X.head())
print(y.head())
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
# 手动查找最优参数
params = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
accuracy = []

for p in params:
    gnb = GaussianNB(var_smoothing=p)
    gnb.fit(X_train, y_train)
    acc = accuracy_score(y_test, gnb.predict(X_test))
    accuracy.append(acc)
    print(f"var_smoothing={p}: accuracy={acc}")

best_acc = max(accuracy)
best_param = params[accuracy.index(best_acc)]

print(f"最高准确率：{best_acc}")
print(f"最优参数：var_smoothing = {best_param}")



params_dic={'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]}
gnb=GaussianNB()
grid=GridSearchCV(estimator=gnb,param_grid=params_dic,cv=5,scoring='accuracy')
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.score(X_test, y_test))