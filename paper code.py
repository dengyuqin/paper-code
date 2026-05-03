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
    [0, 0, 0, 1, 1], [0.5, 0.5, 0.5, 0, 0], [1, 1, 1, 0.5, 0.5], [0, 0, 0, 0, 0],
    [0, 0, 0, 0.5, 0.5], [0, 0, 0, 0, 0], [0.5, 0.5, 0.5, 0, 0], [0, 0, 0, 1, 1],
    [1, 1, 1, 0, 0], [0.5, 0.5, 0.5, 0, 0]
])
n, m = M.shape
I2 = np.array([0.75, 0.68, 0.82, 0.71, 0.79])

I3 = np.zeros((n, m))
for doc, pats in enumerate([[0, 7], [1, 6], [2, 5], [3, 8], [4, 9]]):
    for p in pats:
        I3[p, doc] = 0.92
    others = [p for p in range(n) if p not in pats]
    for i, p in enumerate(others):
        I3[p, doc] = 0.70 + 0.02 * i + 0.01 * doc

R2 = np.zeros((m, n))
for doc, pats in enumerate([[0, 7], [1, 6], [2, 5], [3, 8], [4, 9]]):
    for p in pats:
        R2[doc, p] = 0.92
    others = [p for p in range(n) if p not in pats]
    for i, p in enumerate(others):
        R2[doc, p] = 0.50 + 0.02 * i + 0.01 * doc
R2 = np.clip(R2, 0, 1)

# 计算满意度矩阵 Y (患者对医生) 和 X (医生对患者)
Y = np.zeros((n, m))
X = np.zeros((m, n))
for j in range(n):
    for i in range(m):
        Y[j, i] = (w_congruence * M[j, i] + w_competence * I2[i] +
                   w_interaction * I3[j, i] + w_f * F[i])
for i in range(m):
    for j in range(n):
        X[i, j] = w_obj * M[j, i] + w_beh * R2[i, j]
Y = np.clip(Y, 0, 1)
X = np.clip(X, 0, 1)

# 扩展虚拟医生（每个医生容量2）
capacity = 2
k = m * capacity
X_prime = np.zeros((k, n))
Y_prime = np.zeros((n, k))
for i in range(m):
    for c in range(capacity):
        X_prime[i * capacity + c] = X[i]
        Y_prime[:, i * capacity + c] = Y[:, i]


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

# 输出结果
print("=" * 60)

print(f"本方法 (Gale-Shapley): {proposed_score:.4f}")

print("\n匹配结果（每个医生匹配两位患者）：")
for doc in sorted(matches):
    print(f"Doctor {doc + 1}: patients {sorted(matches[doc])}")

np.set_printoptions(precision=3, suppress=True)
print("\n原始医生满意度矩阵 X (5x10) [医生对患者]:")
print(X)
print("\n原始患者满意度矩阵 Y (10x5) [患者对医生]:")
print(Y)


def plot_combined_heatmaps_with_labels(X, Y, X_prime, Y_prime, output_file='satisfaction_matrices.png'):
    """
    将四个满意度矩阵绘制在一个 2×2 的子图布局中
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 定义每个子图的配置：矩阵，标题，xlabel，ylabel，子图标签
    plots_config = [
        (axes[0, 0], X, 'Doctor satisfaction matrix X',
         'Patients (10)', 'Doctors (5)', '(a)'),
        (axes[0, 1], Y, 'Patient satisfaction matrix Y',
         'Doctors (5)', 'Patients (10)', '(b)'),
        (axes[1, 0], X_prime, 'Extended doctor satisfaction matrix X\'',
         'Patients (10)', 'Virtual doctors (10)', '(c)'),
        (axes[1, 1], Y_prime, 'Extended patient satisfaction matrix Y\'',
         'Virtual doctors (10)', 'Patients (10)', '(d)')
    ]

    for ax, mat, title, xlabel, ylabel, label in plots_config:
        sns.heatmap(mat, annot=True, fmt='.2f', cmap='coolwarm', cbar=True,
                    xticklabels=range(mat.shape[1]), yticklabels=range(mat.shape[0]),
                    annot_kws={'size': 7}, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis='both', labelsize=8)

        # 在子图左上角添加 (a), (b), (c), (d) 标签
        ax.text(-0.12, 1.02, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n合并热力图（带子图标签）已保存为: {output_file}")


# 绘制合并热力图
plot_combined_heatmaps_with_labels(X, Y, X_prime, Y_prime, output_file='satisfaction_matrices.png')

print("\n图像已保存: satisfaction_matrices.png")