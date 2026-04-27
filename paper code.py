import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 权重参数 ====================
alpha = 0.375
w_congruence = 0.358
w_competence = 0.321
w_interaction = 0.321
w_obj = 0.355
w_beh = 0.645
w_cost = w_adherence = w_literacy = w_emotion = w_cooperation = 0.20
w_f = 0.1

# 外部评价矩阵 F
F = np.array([5, 4, 5, 3, 4]) / 5.0

M = np.array([
    [0,0,0,1,1], [0.5,0.5,0.5,0,0], [1,1,1,0.5,0.5], [0,0,0,0,0],
    [0,0,0,0.5,0.5], [0,0,0,0,0], [0.5,0.5,0.5,0,0], [0,0,0,1,1],
    [1,1,1,0,0], [0.5,0.5,0.5,0,0]
])
n, m = M.shape
I2 = np.array([0.75, 0.68, 0.82, 0.71, 0.79])

I3 = np.zeros((n,m))
for doc, pats in enumerate([[0,7],[1,6],[2,5],[3,8],[4,9]]):
    for p in pats:
        I3[p, doc] = 0.92
    others = [p for p in range(n) if p not in pats]
    for i, p in enumerate(others):
        I3[p, doc] = 0.70 + 0.02*i + 0.01*doc

R2 = np.zeros((m, n))
for doc, pats in enumerate([[0,7],[1,6],[2,5],[3,8],[4,9]]):
    for p in pats:
        R2[doc, p] = 0.92
    others = [p for p in range(n) if p not in pats]
    for i, p in enumerate(others):
        R2[doc, p] = 0.50 + 0.02*i + 0.01*doc
R2 = np.clip(R2, 0, 1)

# 计算满意度矩阵 Y (患者对医生) 和 X (医生对患者)
Y = np.zeros((n,m))
X = np.zeros((m,n))
for j in range(n):
    for i in range(m):
        Y[j,i] = (w_congruence*M[j,i] + w_competence*I2[i] +
                  w_interaction*I3[j,i] + w_f*F[i])
for i in range(m):
    for j in range(n):
        X[i,j] = w_obj*M[j,i] + w_beh*R2[i,j]
Y = np.clip(Y, 0, 1)
X = np.clip(X, 0, 1)

# 扩展虚拟医生（每个医生容量2）
capacity = 2
k = m * capacity
X_prime = np.zeros((k, n))
Y_prime = np.zeros((n, k))
for i in range(m):
    for c in range(capacity):
        X_prime[i*capacity + c] = X[i]
        Y_prime[:, i*capacity + c] = Y[:, i]

# Gale-Shapley 算法（本方法，返回总满意度和匹配结果）
def gale_shapley(Y_prime, X_prime, capacity):
    n_y = Y_prime.shape[0]
    k_x = X_prime.shape[0]
    patient_prefs = [np.argsort(-Y_prime[j]).tolist() for j in range(n_y)]
    doctor_prefs = [np.argsort(-X_prime[i]).tolist() for i in range(k_x)]
    free_patients = list(range(n_y))
    patient_match = [None] * n_y
    doctor_match = [None] * k_x
    next_proposal = [0] * n_y
    while free_patients:
        p = free_patients.pop(0)
        pref = patient_prefs[p]
        while next_proposal[p] < len(pref):
            d = pref[next_proposal[p]]
            next_proposal[p] += 1
            if doctor_match[d] is None:
                patient_match[p] = d
                doctor_match[d] = p
                break
            else:
                p_cur = doctor_match[d]
                pref_d = doctor_prefs[d]
                if pref_d.index(p) < pref_d.index(p_cur):
                    patient_match[p] = d
                    doctor_match[d] = p
                    free_patients.append(p_cur)
                    break
    # 记录匹配结果
    matches = {}
    for d, p in enumerate(doctor_match):
        if p is not None:
            real_doc = d // capacity
            matches.setdefault(real_doc, []).append(p + 1)
    # 计算总满意度
    total = 0
    for doc, plist in matches.items():
        for p in plist:
            p_idx = p - 1
            total += alpha * Y[p_idx, doc] + (1 - alpha) * X[doc, p_idx]
    return total, matches

proposed_score, matches = gale_shapley(Y_prime, X_prime, capacity)

# 随机匹配（1000次）
random.seed(123)
random_scores = []
for _ in range(1000):
    patients = list(range(n))
    random.shuffle(patients)
    assign = {i: [] for i in range(m)}
    for idx, p in enumerate(patients):
        assign[idx % m].append(p)
    total = 0
    for i, plist in assign.items():
        for p in plist:
            total += alpha * Y[p, i] + (1 - alpha) * X[i, p]
    random_scores.append(total)
random_mean = np.mean(random_scores)
random_std = np.std(random_scores)

# 贪婪匹配（1000次）
greedy_scores = []
for _ in range(1000):
    order = list(range(n))
    random.shuffle(order)
    remain = [capacity] * m
    total = 0
    for p in order:
        candidates = np.argsort(-Y[p])
        for i in candidates:
            if remain[i] > 0:
                total += alpha * Y[p, i] + (1 - alpha) * X[i, p]
                remain[i] -= 1
                break
    greedy_scores.append(total)
greedy_mean = np.mean(greedy_scores)
greedy_std = np.std(greedy_scores)

# 输出结果
print("=" * 60)
print("基线比较结果：")
print(f"随机匹配 (1000次): 均值 = {random_mean:.4f} ± {random_std:.4f}")
print(f"贪婪匹配 (1000次):   均值 = {greedy_mean:.4f} ± {greedy_std:.4f}")
print(f"本方法 (Gale-Shapley): {proposed_score:.4f}")
print(f"\n本方法 vs 随机: 提升 {((proposed_score - random_mean) / random_mean) * 100:.1f}%")
print(f"本方法 vs 贪婪: 提升 {((proposed_score - greedy_mean) / greedy_mean) * 100:.1f}%")

print("\n匹配结果（每个医生匹配两位患者）：")
for doc in sorted(matches):
    print(f"Doctor {doc + 1}: patients {sorted(matches[doc])}")

# 打印矩阵数值
np.set_printoptions(precision=3, suppress=True)
print("\n原始医生满意度矩阵 X (5x10) [医生对患者]:")
print(X)
print("\n原始患者满意度矩阵 Y (10x5) [患者对医生]:")
print(Y)
print("\n扩展医生满意度矩阵 X' (10x10) [虚拟医生对患者]:")
print(X_prime)
print("\n扩展患者满意度矩阵 Y' (10x10) [患者对虚拟医生]:")
print(Y_prime)

# 绘制热力图
def plot_heatmap(mat, title, filename, cmap='coolwarm'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, annot=True, fmt='.2f', cmap=cmap, cbar=True,
                xticklabels=range(mat.shape[1]), yticklabels=range(mat.shape[0]),
                annot_kws={'size': 8})
    plt.title(title, fontsize=14)
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

plot_heatmap(X, 'Doctor satisfaction matrix X (5x10)', 'X_heatmap.png')
plot_heatmap(Y, 'Patient satisfaction matrix Y (10x5)', 'Y_heatmap.png')
plot_heatmap(X_prime, 'Extended doctor satisfaction matrix X\' (10x10)', 'X_prime_heatmap.png')
plot_heatmap(Y_prime, 'Extended patient satisfaction matrix Y\' (10x10)', 'Y_prime_heatmap.png')

# 绘制基线比较柱状图
plt.figure(figsize=(7, 5))
categories = ['Random\nmatching', 'Greedy\nmatching', 'Proposed\n(Gale-Shapley)']
means = [random_mean, greedy_mean, proposed_score]
errors = [random_std, greedy_std, 0]
colors =['#1f77b4', '#ff7f0e', '#2ca02c']

bars = plt.bar(categories, means, yerr=errors, capsize=8, color=colors, edgecolor='black', alpha=0.85)
for bar, val in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.28, f'{val:.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.ylabel('Total mutual satisfaction', fontsize=12)
plt.title('Comparison of matching strategies', fontsize=14)
plt.ylim(0, 8.5)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('baseline_comparison.png', dpi=300)
plt.show()

print("\n所有图像已保存: X_heatmap.png, Y_heatmap.png, X_prime_heatmap.png, Y_prime_heatmap.png, baseline_comparison.png")