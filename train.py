import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# โหลดข้อมูล
file_path = "winequality-red.csv"
df = pd.read_csv(file_path)

# แปลงค่า "quality" เป็น Binary Classification
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)  # 1 = คุณภาพสูง, 0 = คุณภาพต่ำ

# แยก Features และ Target
X = df.drop("quality", axis=1)
y = df["quality"]

# แบ่งข้อมูลเป็น Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ปรับสเกลข้อมูล (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# บันทึก Scaler
joblib.dump(scaler, "scaler.pkl")

# สร้างโมเดลหลายตัว
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

# เทรนแต่ละโมเดลและวัดความแม่นยำ
model_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_results[name] = acc

# ใช้ Voting Classifier รวมผลจากทุกโมเดล
voting_clf = VotingClassifier(estimators=[(name, models[name]) for name in models], voting='hard')
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
voting_acc = accuracy_score(y_test, y_pred_voting)

# เพิ่มผลลัพธ์ของ Voting Classifier
model_results["Voting Classifier"] = voting_acc

# เลือกโมเดลที่ดีที่สุด
best_model_name = max(model_results, key=model_results.get)
best_model = models[best_model_name]

# บันทึกโมเดลที่ดีที่สุด
joblib.dump(best_model, "best_wine_quality_model.pkl")

# แสดงผลลัพธ์
print("\n✅ โมเดลที่ดีที่สุด:", best_model_name)
print("🎯 ความแม่นยำของแต่ละโมเดล:")
for model, acc in model_results.items():
    print(f"{model}: {acc:.4f}")
print("\n📌 บันทึกโมเดลไว้ที่ 'best_wine_quality_model.pkl'")
print("📌 บันทึก Scaler ไว้ที่ 'scaler.pkl'")

# แสดงกราฟเปรียบเทียบผลลัพธ์
plt.figure(figsize=(10,5))
plt.bar(model_results.keys(), model_results.values(), color='blue')
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Comparison of Model Accuracies")
plt.xticks(rotation=45)
plt.show()
