import pymysql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

conn = pymysql.connect(
    host='localhost',
    user='root',
    password='20040915',
    database='lndx',  # 直接指定数据库
    charset='utf8mb4'
)
df=pd.read_sql("select * from course",conn)
conn.close()
print(df.head())

X=df[['ctime']]
y=df['cost']
print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
# 重要：将y_train和y_test转换为二维数组
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

lr=LinearRegression()
lr.fit(X_train_scaled,y_train_scaled)
y_pred_scaled=lr.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

r2 = r2_score(y_test, y_pred)
print(r2)
#列举部分alpha
ridge_alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
lasso_alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
#储存不同alpha下lasso回归和ridge回归的结果
r_result=[]
l_result=[]

#求出每个alpha对应的ridge回归的R**2
for alphas in ridge_alphas:
    ridge = Ridge(alpha=alphas)
    ridge.fit(X_train_scaled, y_train_scaled)
    r_result.append(ridge.score(X_test_scaled, y_test_scaled))

#求出每个alpha对应的lasso回归的R**2
for alphas in lasso_alphas:
    lasso = Lasso(alpha=alphas)
    lasso.fit(X_train_scaled, y_train_scaled)
    l_result.append(lasso.score(X_test_scaled, y_test_scaled))
#输出结果
print(r_result)
print(l_result)
# 找到最佳 alpha 值
best_ridge_alpha = ridge_alphas[np.argmax(r_result)]
best_ridge_score = max(r_result)
best_lasso_alpha = lasso_alphas[np.argmax(l_result)]
best_lasso_score = max(l_result)

# 绘制 Lasso 回归不同 alpha 值的性能
plt.figure()
plt.scatter(X_train, y_train, c='blue', label='train')
plt.scatter(X_test, y_test, c='red', label='test')
X_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
X_line_scaled = scaler_X.transform(X_line)
y_line_scaled = lr.predict(X_line_scaled)
y_line = scaler_y.inverse_transform(y_line_scaled.reshape(-1, 1)).ravel()
plt.plot(X_line, y_line, color='green', linewidth=2, label='LinearRegression')
plt.xscale('log')
plt.xlabel('time')
plt.ylabel('cost')
plt.grid(True)
plt.legend()
plt.show()

# 绘制 Lasso 回归不同 alpha 值的性能
plt.plot(ridge_alphas, r_result, 'o-', label='Ridge 回归')
plt.xscale('log')
plt.xlabel('Alpha ')
plt.ylabel('R**2 ')
plt.title('Ridge and alpha')
plt.grid(True)
plt.legend()
plt.show()

# 绘制 Lasso 回归不同 alpha 值的性能
plt.plot(lasso_alphas, l_result, 'o-', label='Lasso 回归')
plt.xscale('log')
plt.xlabel('Alpha ')
plt.ylabel('R**2 ')
plt.title('Lasso and alpha ')
plt.grid(True)
plt.legend()
plt.show()