import pandas as pd
import numpy as np
import joblib1
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms
import random

# โหลดข้อมูล
file_path = "winequality-red.csv"
df = pd.read_csv(file_path)

# แปลงค่า "quality" เป็น Binary Classification
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)

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

# สร้างโครงสร้างสำหรับ GA
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 1, 20)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ฟังก์ชันเพื่อประเมินความแม่นยำของโมเดล
def evaluate(individual):
    k = int(max(1, individual[0]))  # Ensure k is at least 1
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return (accuracy,)

toolbox.register("evaluate", evaluate)

def cxCustom(child1, child2):
    if random.random() < 0.5:
        child1[0], child2[0] = child2[0], child1[0]
    return child1, child2

toolbox.register("mate", cxCustom)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=20, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# สร้างประชากร (population)
population = toolbox.population(n=10)

# ใช้ GA เพื่อหาค่าพารามิเตอร์ที่ดีที่สุด
for generation in range(10):
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.7:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < 0.2:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_individuals)
    for ind, fit in zip(invalid_individuals, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring

    top_individual = tools.selBest(population, 1)[0]
    print(f"Generation {generation}: Best fitness = {top_individual.fitness.values[0] * 100:.2f}%")

best_k = tools.selBest(population, 1)[0][0]
print(f"Best k-value: {best_k}")

# สร้างโมเดล k-NN ที่ใช้ k-value ที่ได้จาก GA
best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train, y_train)

# ทำนายผลลัพธ์
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the best model: {accuracy * 100:.2f}%")

# เปรียบเทียบผลลัพธ์
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=best_k),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

model_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_results[name] = acc
    print(f"{name}: Accuracy = {acc * 100:.2f}%")

# ใช้ Voting Classifier รวมผลจากทุกโมเดล
voting_clf = VotingClassifier(estimators=[(name, models[name]) for name in models], voting='hard')
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
voting_acc = accuracy_score(y_test, y_pred_voting)
model_results["Voting Classifier"] = voting_acc
print(f"Voting Classifier: Accuracy = {voting_acc * 100:.2f}%")

# เลือกโมเดลที่ดีที่สุด
best_model_name = max(model_results, key=model_results.get)
best_model = models[best_model_name] if best_model_name in models else voting_clf
print(f"\nBest performing model: {best_model_name} with an accuracy of {model_results[best_model_name] * 100:.2f}%")

# บันทึกโมเดลที่เทรนได้ดีที่สุด
best_model = models[best_model_name] if best_model_name in models else voting_clf
joblib.dump(best_model, "best_wine_quality_model.pkl")


# แสดงกราฟเปรียบเทียบผลลัพธ์
plt.figure(figsize=(10, 5))
plt.bar(model_results.keys(), [acc * 100 for acc in model_results.values()], color='blue')
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.title("Comparison of Model Accuracies")
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
