import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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

# กำหนดโมเดลและพารามิเตอร์สำหรับ Grid Search และ Random Search
param_grids = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11]
    },
    "Decision Tree": {
        "max_depth": [None, 5, 10, 15, 20],
        "criterion": ["gini", "entropy"]
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "criterion": ["gini", "entropy"]
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1]
    }
}

# เทรนโมเดลโดยใช้ Grid Search และ Random Search
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

model_results = {}

for name, model in models.items():
    print(f"🔍 กำลังเทรนโมเดล {name} ด้วย Grid Search...")
    
    # เลือกใช้ GridSearch หรือ RandomizedSearch
    search = GridSearchCV(model, param_grids[name], cv=5, scoring="accuracy", n_jobs=-1)
    
    # เทรนโมเดล
    search.fit(X_train, y_train)
    
    # ทดสอบบนชุดข้อมูลทดสอบ
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # บันทึกผลลัพธ์
    model_results[name] = acc
    models[name] = best_model  # ใช้โมเดลที่ดีที่สุด

    print(f"✅ {name} - ความแม่นยำ: {acc:.4f} - Best Params: {search.best_params_}")

# ใช้ Voting Classifier รวมผลจากทุกโมเดลที่ดีที่สุด
voting_clf = VotingClassifier(estimators=[(name, models[name]) for name in models], voting='hard')
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
voting_acc = accuracy_score(y_test, y_pred_voting)

# เพิ่มผลลัพธ์ของ Voting Classifier
model_results["Voting Classifier"] = voting_acc

# เลือกโมเดลที่ดีที่สุด
best_model_name = max(model_results, key=model_results.get)
best_model = models[best_model_name] if best_model_name in models else voting_clf

# บันทึกโมเดลที่ดีที่สุด
joblib.dump(best_model, "best_wine_quality_model.pkl")

# แสดงผลลัพธ์
print("\n🎯 โมเดลที่ดีที่สุด:", best_model_name)
print("📊 ความแม่นยำของแต่ละโมเดล:")
for model, acc in model_results.items():
    print(f"{model}: {acc:.4f}")
print("\n📂 บันทึกโมเดลไว้ที่ 'best_wine_quality_model.pkl'")
print("📂 บันทึก Scaler ไว้ที่ 'scaler.pkl'")

# แสดงกราฟเปรียบเทียบผลลัพธ์
plt.figure(figsize=(10, 5))
plt.bar(model_results.keys(), model_results.values(), color='blue')
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Comparison of Model Accuracies with Grid Search Optimization")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
