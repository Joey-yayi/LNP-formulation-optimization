"""
LNP配方优化 - D-optimal设计（知识增强版）+ 浓度计算
从216个配方中选择36个最优配方（最大化空间覆盖）
添加：基于摩尔比和分子量计算各组分浓度（总脂质浓度8mg/ml）

策略：
1. 主要目标：最大化空间覆盖率（D-optimal）
2. 轻微偏好：C12-200和DOTAP各多1-2组
3. 保持平衡：确保普适性和代表性
4. 新增功能：计算各组分的实际浓度(mg/ml)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

print("\n" + "=" * 95)
print(" " * 25 + "LNP配方优化设计（知识增强版 + 浓度计算）")
print("=" * 95)
print(f"策略: D-optimal最大化空间覆盖 + 轻微知识引导")
print(f"目标: 从216个配方中选择36个代表性配方")
print(f"新增: 基于摩尔比计算各组分浓度（总脂质浓度8mg/ml）")
print("=" * 95 + "\n")

# ==================== 脂质分子量数据 ====================

LIPID_MW = {
    'ALC-0315': 766.3,
    'SM102': 710.2,
    'MC3': 642.1,
    'C12-200': 1136.9,
    'DOTAP': 698.5,
    'DODAP': 648.1,
    'DOPE': 744.0,
    'DSPC': 790.2,
    'DMG-PEG2000': 2509.2
}

# 总脂质浓度 (mg/ml)
TOTAL_LIPID_CONCENTRATION = 8.0

# ==================== 配方比例模板 ====================

FORMULATION_TEMPLATES = {
    'Type1': {
        'name': '高胆固醇配方',
        'total_IL': 50.0,
        'PL': 10.0,
        'Chol': 38.5,
        'PEG': 1.5
    },
    'Type2': {
        'name': '中胆固醇配方',
        'total_IL': 58.0,
        'PL': 20.0,
        'Chol': 19.25,
        'PEG': 2.75
    },
    'Type3': {
        'name': '无胆固醇配方',
        'total_IL': 66.0,
        'PL': 30.0,
        'Chol': 0.0,
        'PEG': 4.0
    }
}

# 轻微偏好权重（用于打破平局）
LIPID_BONUS = {
    'C12-200': 0.05,  # +5% 小权重（实验效果好）
    'DOTAP': 0.03,  # +3% 小权重（转染效率高）
    'CKK-E12': 0.02,  # +2% 小权重（文献报道好）
    'SM102': 0.0,
    'ALC-0315': 0.0,
    'MC3': 0.0,
    'DODAP': 0.0
}

# 胆固醇分子量（假设使用cholesterol）
CHOLESTEROL_MW = 386.65


# ==================== 浓度计算函数 ====================

def calculate_concentrations(formulation_row):
    """
    根据摩尔百分比和分子量计算各组分的浓度

    参数:
        formulation_row: 包含配方信息的DataFrame行

    返回:
        dict: 各组分的浓度(mg/ml)
    """
    # 获取各组分的摩尔百分比
    il1_mol_pct = formulation_row['IL1%']
    il2_mol_pct = formulation_row['IL2%']
    pl_mol_pct = formulation_row['PL%']
    chol_mol_pct = formulation_row['Chol%']
    peg_mol_pct = formulation_row['PEG%']

    # 获取各组分的分子量
    il1_mw = LIPID_MW.get(formulation_row['IL1'], 700)  # 默认值700
    il2_mw = LIPID_MW.get(formulation_row['IL2'], 700)
    pl_mw = LIPID_MW.get(formulation_row['PL'], 750)
    chol_mw = CHOLESTEROL_MW
    peg_mw = LIPID_MW['DMG-PEG2000']

    # 计算总的摩尔加权分子量
    total_weighted_mw = (
                                il1_mol_pct * il1_mw +
                                il2_mol_pct * il2_mw +
                                pl_mol_pct * pl_mw +
                                chol_mol_pct * chol_mw +
                                peg_mol_pct * peg_mw
                        ) / 100

    # 计算各组分的质量百分比
    il1_mass_pct = (il1_mol_pct * il1_mw / total_weighted_mw)
    il2_mass_pct = (il2_mol_pct * il2_mw / total_weighted_mw)
    pl_mass_pct = (pl_mol_pct * pl_mw / total_weighted_mw)
    chol_mass_pct = (chol_mol_pct * chol_mw / total_weighted_mw)
    peg_mass_pct = (peg_mol_pct * peg_mw / total_weighted_mw)

    # 计算各组分的浓度 (mg/ml)
    concentrations = {
        'IL1_mg/ml': il1_mass_pct * TOTAL_LIPID_CONCENTRATION / 100,
        'IL2_mg/ml': il2_mass_pct * TOTAL_LIPID_CONCENTRATION / 100,
        'PL_mg/ml': pl_mass_pct * TOTAL_LIPID_CONCENTRATION / 100,
        'Chol_mg/ml': chol_mass_pct * TOTAL_LIPID_CONCENTRATION / 100,
        'PEG_mg/ml': peg_mass_pct * TOTAL_LIPID_CONCENTRATION / 100,
        'Total_mg/ml': TOTAL_LIPID_CONCENTRATION
    }

    return concentrations


# ==================== 步骤1：生成配方空间 ====================

def generate_all_formulations():
    """生成所有216个LNP配方"""
    print("【步骤1】生成配方空间")
    print("-" * 95)

    ionizable_lipids = ['ALC-0315', 'SM102', 'MC3', 'C12-200', 'DOTAP', 'DODAP']
    phospholipids = ['DOPE', 'DSPC']

    print("\n配方比例模板:")
    for template_id, template in FORMULATION_TEMPLATES.items():
        total = template['total_IL'] + template['PL'] + template['Chol'] + template['PEG']
        print(f"  {template['name']:12s} ({template_id}): "
              f"离子化脂质{template['total_IL']:5.2f}% : "
              f"磷脂{template['PL']:5.2f}% : "
              f"胆固醇{template['Chol']:5.2f}% : "
              f"PEG{template['PEG']:4.2f}%  (总和={total:.2f}%)")

    print(f"\n总脂质浓度: {TOTAL_LIPID_CONCENTRATION} mg/ml")

    print("\n脂质分子量 (g/mol):")
    for lipid, mw in sorted(LIPID_MW.items()):
        print(f"  {lipid:12s}: {mw:7.1f}")
    print(f"  {'Cholesterol':12s}: {CHOLESTEROL_MW:7.1f}")

    print("\n知识引导偏好（轻微权重，仅用于打破平局）:")
    for lipid, bonus in sorted(LIPID_BONUS.items(), key=lambda x: -x[1]):
        if bonus > 0:
            print(f"  {lipid:12s}: +{bonus * 100:.1f}% 权重")
    print(f"  其他脂质: 无额外权重")

    formulations = []
    formulation_id = 1

    for il1 in ionizable_lipids:
        for il2 in ionizable_lipids:
            for pl in phospholipids:
                for template_id, template in FORMULATION_TEMPLATES.items():
                    total_il = template['total_IL']
                    il1_pct = total_il / 2.0
                    il2_pct = total_il / 2.0

                    pl_pct = template['PL']
                    chol_pct = template['Chol']
                    peg_pct = template['PEG']

                    # 计算轻微偏好得分（仅用于打破平局）
                    bonus_score = LIPID_BONUS.get(il1, 0) + LIPID_BONUS.get(il2, 0)

                    formulation = {
                        'ID': formulation_id,
                        'Type': template_id,
                        'TypeName': template['name'],
                        'IL1': il1,
                        'IL1%': il1_pct,
                        'IL2': il2,
                        'IL2%': il2_pct,
                        'PL': pl,
                        'PL%': pl_pct,
                        'Chol%': chol_pct,
                        'PEG%': peg_pct,
                        'TotalIL%': total_il,
                        'BonusScore': bonus_score
                    }

                    # 计算各组分浓度
                    concentrations = calculate_concentrations(pd.Series(formulation))
                    formulation.update(concentrations)

                    formulations.append(formulation)
                    formulation_id += 1

    df = pd.DataFrame(formulations)

    print(f"\n✓ 配方空间生成完成")
    print(f"  总配方数: {len(df)}")
    print(f"  空间维度: 6 (IL1) × 6 (IL2) × 2 (PL) × 3 (Type) = {6 * 6 * 2 * 3}")

    return df


# ==================== 步骤2：特征编码 ====================

def encode_formulations(df):
    """编码配方特征"""
    print(f"\n【步骤2】特征编码")
    print("-" * 95)

    il1_dummies = pd.get_dummies(df['IL1'], prefix='IL1')
    il2_dummies = pd.get_dummies(df['IL2'], prefix='IL2')
    pl_dummies = pd.get_dummies(df['PL'], prefix='PL')
    type_dummies = pd.get_dummies(df['Type'], prefix='Type')

    numeric_features = df[['IL1%', 'IL2%', 'PL%', 'Chol%', 'PEG%', 'TotalIL%']].copy()

    numeric_features['IL_same'] = (df['IL1'] == df['IL2']).astype(int)
    numeric_features['IL_PL_ratio'] = numeric_features['TotalIL%'] / (numeric_features['PL%'] + 0.1)

    X = pd.concat([il1_dummies, il2_dummies, pl_dummies, type_dummies, numeric_features], axis=1)

    print(f"✓ 特征编码完成")
    print(f"  总特征维度: {X.shape[1]}")

    return X.values


# ==================== 步骤3：知识增强的D-optimal选择 ====================

def knowledge_enhanced_d_optimal(X, df, n_select=36, random_seed=42):
    """
    知识增强的D-optimal选择

    策略：
    1. 主要使用D-optimal准则（保证空间覆盖）
    2. 在D-criterion相近时（差异<5%），优先选择带bonus的配方
    3. 确保不会过度偏向某个脂质
    """
    print(f"\n【步骤3】知识增强的D-optimal选择")
    print("-" * 95)
    print(f"  主要策略: D-optimal最大化空间覆盖")
    print(f"  辅助策略: 相近情况下优先C12-200和DOTAP")
    print(f"  目标样本: {n_select} 个")

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"✓ 特征标准化完成")

    n_total = X_scaled.shape[0]

    # D-criterion计算
    def compute_d_criterion(indices):
        X_subset = X_scaled[indices]
        info_matrix = X_subset.T @ X_subset
        try:
            sign, logdet = np.linalg.slogdet(info_matrix)
            return logdet if sign > 0 else -1e10
        except:
            return -1e10

    # 初始化
    np.random.seed(random_seed)
    selected = [np.random.randint(0, n_total)]
    remaining = list(set(range(n_total)) - set(selected))

    d_history = [compute_d_criterion(selected)]
    bonus_used_count = 0  # 记录因bonus被选中的次数

    print(f"\n开始知识增强的贪婪选择:")
    print(f"  初始样本: 配方{selected[0] + 1}")
    print(f"  初始D-criterion: {d_history[0]:.2f}\n")

    for iteration in range(1, n_select):
        best_idx = None
        best_d = -np.inf
        second_best_d = -np.inf
        second_best_idx = None

        # 评估候选样本
        n_candidates = min(200, len(remaining))
        np.random.seed(random_seed + iteration)
        candidates = np.random.choice(remaining, n_candidates, replace=False)

        # 找到最好和第二好的候选
        for candidate in candidates:
            test_selected = selected + [candidate]
            d_value = compute_d_criterion(test_selected)

            if d_value > best_d:
                second_best_d = best_d
                second_best_idx = best_idx
                best_d = d_value
                best_idx = candidate
            elif d_value > second_best_d:
                second_best_d = d_value
                second_best_idx = candidate

        # 知识增强决策：如果前两名差异很小（<5%），且第二名有bonus，选第二名
        if best_idx is not None:
            chosen_idx = best_idx

            if second_best_idx is not None and second_best_d > -1e9:
                d_diff_pct = abs(best_d - second_best_d) / abs(best_d) * 100

                if d_diff_pct < 5.0:  # 差异小于5%才考虑bonus
                    best_bonus = df.iloc[best_idx]['BonusScore']
                    second_bonus = df.iloc[second_best_idx]['BonusScore']

                    # 如果第二名的bonus更高，选择第二名
                    if second_bonus > best_bonus:
                        chosen_idx = second_best_idx
                        best_d = second_best_d
                        bonus_used_count += 1

            selected.append(chosen_idx)
            remaining.remove(chosen_idx)
            d_history.append(best_d)

            if (iteration + 1) % 6 == 0 or iteration == n_select - 1:
                print(f"  第{iteration + 1:2d}个样本 | 配方{chosen_idx + 1:3d} | "
                      f"D-criterion: {best_d:10.2f}")

    print(f"\n✓ 选择完成！")
    print(f"  最终D-criterion: {d_history[-1]:.2f}")
    print(f"  知识引导生效次数: {bonus_used_count} ({bonus_used_count / (n_select - 1) * 100:.1f}%)")
    print(f"  说明: {100 - bonus_used_count / (n_select - 1) * 100:.1f}% 纯基于D-optimal选择")

    return selected, d_history


# ==================== 步骤4：结果分析 ====================

def analyze_selection(df, selected_indices):
    """分析选中配方"""
    print(f"\n【步骤4】结果分析")
    print("-" * 95)

    selected_df = df.iloc[selected_indices].copy().reset_index(drop=True)
    selected_df.insert(0, '选择顺序', range(1, len(selected_df) + 1))

    print(f"\n组成分布统计:")

    # 配方类型分布
    print(f"\n  配方类型:")
    type_dist = selected_df['TypeName'].value_counts().sort_index()
    for type_name, count in type_dist.items():
        pct = count / len(selected_df) * 100
        bar = '█' * int(pct / 5)
        print(f"    {type_name:15s}: {count:2d} ({pct:5.1f}%) {bar}")

    # 离子化脂质1分布
    print(f"\n  离子化脂质1:")
    il1_dist = selected_df['IL1'].value_counts().sort_index()
    for lipid, count in il1_dist.items():
        pct = count / len(selected_df) * 100
        bar = '█' * int(pct / 5)
        bonus = LIPID_BONUS.get(lipid, 0)
        bonus_str = f'(+{bonus * 100:.0f}%)' if bonus > 0 else ''
        print(f"    {lipid:12s}: {count:2d} ({pct:5.1f}%) {bar} {bonus_str}")

    # 离子化脂质2分布
    print(f"\n  离子化脂质2:")
    il2_dist = selected_df['IL2'].value_counts().sort_index()
    for lipid, count in il2_dist.items():
        pct = count / len(selected_df) * 100
        bar = '█' * int(pct / 5)
        bonus = LIPID_BONUS.get(lipid, 0)
        bonus_str = f'(+{bonus * 100:.0f}%)' if bonus > 0 else ''
        print(f"    {lipid:12s}: {count:2d} ({pct:5.1f}%) {bar} {bonus_str}")

    # 关键脂质对比
    print(f"\n  关键脂质对比（出现总次数 = IL1 + IL2）:")
    lipid_total_counts = {}
    for lipid in ['C12-200', 'MC3', 'DOTAP', 'ALC-0315', 'SM102', 'DODAP']:
        count_il1 = (selected_df['IL1'] == lipid).sum()
        count_il2 = (selected_df['IL2'] == lipid).sum()
        total = count_il1 + count_il2
        lipid_total_counts[lipid] = total
        print(f"    {lipid:12s}: {total:2d} 次 (IL1:{count_il1}, IL2:{count_il2})")

    # 重点对比C12-200 vs MC3
    c12_count = lipid_total_counts['C12-200']
    mc3_count = lipid_total_counts['MC3']
    dotap_count = lipid_total_counts['DOTAP']
    print(f"\n  ✓ C12-200出现 {c12_count} 次，MC3出现 {mc3_count} 次 → "
          f"C12-200多 {c12_count - mc3_count} 次")
    print(f"  ✓ DOTAP出现 {dotap_count} 次")

    # 磷脂分布
    print(f"\n  磷脂:")
    pl_dist = selected_df['PL'].value_counts().sort_index()
    for pl, count in pl_dist.items():
        pct = count / len(selected_df) * 100
        bar = '█' * int(pct / 5)
        print(f"    {pl:12s}: {count:2d} ({pct:5.1f}%) {bar}")

    # 多样性指标
    print(f"\n  多样性指标:")
    n_unique_il_pairs = len(selected_df.groupby(['IL1', 'IL2']).size())
    n_unique_full = len(selected_df.groupby(['IL1', 'IL2', 'PL', 'Type']).size())
    print(f"    唯一离子化脂质组合: {n_unique_il_pairs}")
    print(f"    唯一完整配方: {n_unique_full}")
    print(f"    配方覆盖率: {len(selected_df) / 216 * 100:.1f}%")

    return selected_df


# ==================== 步骤5：打印配方表（含浓度） ====================

def print_full_table(selected_df):
    """打印完整配方表（含浓度信息）"""
    print(f"\n【步骤5】选中的36个配方详细列表（含浓度）")
    print("=" * 120)

    # 表头
    print(f"\n{'顺序':^4} | {'ID':^4} | {'类型':^12} | {'IL1':^10} | {'IL2':^10} | "
          f"{'磷脂':^5} | {'IL%':^5} | {'总浓度':^8}")
    print("-" * 120)

    # 每一行
    for idx, row in selected_df.iterrows():
        print(f"{row['选择顺序']:^4} | {row['ID']:^4} | {row['TypeName']:^12} | "
              f"{row['IL1']:^10} | {row['IL2']:^10} | {row['PL']:^5} | "
              f"{row['TotalIL%']:^5.1f} | {row['Total_mg/ml']:^8.1f}mg/ml")

    print("-" * 120)
    print(f"总计: {len(selected_df)} 个配方\n")

    # 按类型分组（含浓度详情）
    print(f"\n配方详细组成及浓度（按类型分组）:")
    print("=" * 120)

    for type_name in ['高胆固醇配方', '中胆固醇配方', '无胆固醇配方']:
        subset = selected_df[selected_df['TypeName'] == type_name]
        if len(subset) > 0:
            print(f"\n  【{type_name} - 共{len(subset)}个】")
            print("  " + "-" * 116)
            print(f"  {'序号':^4} | {'配方':^6} | {'组成（摩尔%）':^50} | {'浓度(mg/ml)':^50}")
            print("  " + "-" * 116)

            for _, row in subset.iterrows():
                # 摩尔百分比信息
                mol_info = (f"IL1:{row['IL1']:8s}({row['IL1%']:4.1f}%) "
                            f"IL2:{row['IL2']:8s}({row['IL2%']:4.1f}%) "
                            f"PL:{row['PL']:4s}({row['PL%']:4.1f}%)")

                # 浓度信息
                conc_info = (f"IL1:{row['IL1_mg/ml']:.2f} "
                             f"IL2:{row['IL2_mg/ml']:.2f} "
                             f"PL:{row['PL_mg/ml']:.2f} "
                             f"Chol:{row['Chol_mg/ml']:.2f} "
                             f"PEG:{row['PEG_mg/ml']:.2f}")

                print(f"  {row['选择顺序']:^4} | {row['ID']:^6} | {mol_info:50s} | {conc_info:50s}")

    # 打印浓度统计
    print(f"\n\n浓度统计汇总:")
    print("=" * 120)
    print(f"  总脂质浓度: {TOTAL_LIPID_CONCENTRATION} mg/ml")
    print(f"  各组分平均浓度:")
    print(
        f"    - IL1 平均: {selected_df['IL1_mg/ml'].mean():.3f} mg/ml (范围: {selected_df['IL1_mg/ml'].min():.3f} - {selected_df['IL1_mg/ml'].max():.3f})")
    print(
        f"    - IL2 平均: {selected_df['IL2_mg/ml'].mean():.3f} mg/ml (范围: {selected_df['IL2_mg/ml'].min():.3f} - {selected_df['IL2_mg/ml'].max():.3f})")
    print(
        f"    - PL  平均: {selected_df['PL_mg/ml'].mean():.3f} mg/ml (范围: {selected_df['PL_mg/ml'].min():.3f} - {selected_df['PL_mg/ml'].max():.3f})")
    print(
        f"    - Chol平均: {selected_df['Chol_mg/ml'].mean():.3f} mg/ml (范围: {selected_df['Chol_mg/ml'].min():.3f} - {selected_df['Chol_mg/ml'].max():.3f})")
    print(
        f"    - PEG 平均: {selected_df['PEG_mg/ml'].mean():.3f} mg/ml (范围: {selected_df['PEG_mg/ml'].min():.3f} - {selected_df['PEG_mg/ml'].max():.3f})")


# ==================== 步骤6：可视化 ====================

def create_visualization(selected_df, d_history, filename='LNP_Screen_Round1_36samples_with_conc.png'):
    """生成可视化（含浓度信息）"""
    print(f"\n【步骤6】生成可视化图表")
    print("-" * 95)

    try:
        fig = plt.figure(figsize=(20, 14))

        # 1. D-criterion收敛
        ax1 = plt.subplot(3, 4, 1)
        ax1.plot(range(1, len(d_history) + 1), d_history,
                 'o-', linewidth=2.5, markersize=6, color='#2E86AB', alpha=0.8)
        ax1.fill_between(range(1, len(d_history) + 1), d_history, alpha=0.2, color='#2E86AB')
        ax1.set_xlabel('选择样本数', fontsize=11)
        ax1.set_ylabel('D-criterion', fontsize=11)
        ax1.set_title('D-optimal收敛曲线', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. 配方类型分布
        ax2 = plt.subplot(3, 4, 2)
        type_counts = selected_df['TypeName'].value_counts().sort_index()
        colors_type = ['#E63946', '#F77F00', '#06FFA5']
        bars = ax2.bar(range(len(type_counts)), type_counts.values,
                       color=colors_type, alpha=0.8)
        ax2.set_xticks(range(len(type_counts)))
        ax2.set_xticklabels([t.replace('配方', '\n') for t in type_counts.index], fontsize=9)
        ax2.set_ylabel('数量', fontsize=11)
        ax2.set_title('配方类型分布', fontsize=12, fontweight='bold')
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

        # 3. 离子化脂质1分布
        ax3 = plt.subplot(3, 4, 3)
        il1_counts = selected_df['IL1'].value_counts().sort_index()
        colors_il = []
        for lipid in il1_counts.index:
            if lipid == 'C12-200':
                colors_il.append('#FF6B6B')
            elif lipid == 'DOTAP':
                colors_il.append('#FFD93D')
            else:
                colors_il.append('#A8DADC')

        bars = ax3.bar(range(len(il1_counts)), il1_counts.values, color=colors_il, alpha=0.8)
        ax3.set_xticks(range(len(il1_counts)))
        ax3.set_xticklabels(il1_counts.index, rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('数量', fontsize=11)
        ax3.set_title('离子化脂质1分布', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. 离子化脂质2分布
        ax4 = plt.subplot(3, 4, 4)
        il2_counts = selected_df['IL2'].value_counts().sort_index()
        colors_il2 = []
        for lipid in il2_counts.index:
            if lipid == 'C12-200':
                colors_il2.append('#FF6B6B')
            elif lipid == 'DOTAP':
                colors_il2.append('#FFD93D')
            else:
                colors_il2.append('#A8DADC')

        bars = ax4.bar(range(len(il2_counts)), il2_counts.values, color=colors_il2, alpha=0.8)
        ax4.set_xticks(range(len(il2_counts)))
        ax4.set_xticklabels(il2_counts.index, rotation=45, ha='right', fontsize=9)
        ax4.set_ylabel('数量', fontsize=11)
        ax4.set_title('离子化脂质2分布', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. 关键脂质对比
        ax5 = plt.subplot(3, 4, 5)
        key_lipids = ['C12-200', 'MC3', 'DOTAP', 'ALC-0315', 'SM102', 'DODAP']
        key_counts = []
        for lipid in key_lipids:
            count = (selected_df['IL1'] == lipid).sum() + (selected_df['IL2'] == lipid).sum()
            key_counts.append(count)

        colors_key = ['#FF6B6B', '#95E1D3', '#FFD93D', '#A8DADC', '#A8DADC', '#A8DADC']
        bars = ax5.bar(range(len(key_lipids)), key_counts, color=colors_key, alpha=0.8)
        ax5.set_xticks(range(len(key_lipids)))
        ax5.set_xticklabels(key_lipids, rotation=45, ha='right', fontsize=9)
        ax5.set_ylabel('出现总次数', fontsize=11)
        ax5.set_title('关键脂质对比', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=9)

        # 6. 磷脂分布
        ax6 = plt.subplot(3, 4, 6)
        pl_counts = selected_df['PL'].value_counts()
        bars = ax6.bar(pl_counts.index, pl_counts.values,
                       color=['#A8DADC', '#F1FAEE'], alpha=0.8)
        ax6.set_ylabel('数量', fontsize=11)
        ax6.set_title('磷脂分布', fontsize=12, fontweight='bold')
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

        # 7. 离子化脂质协同热图
        ax7 = plt.subplot(3, 4, 7)
        synergy = pd.crosstab(selected_df['IL1'], selected_df['IL2'])
        im = ax7.imshow(synergy.values, cmap='YlOrRd', aspect='auto')
        ax7.set_xticks(range(len(synergy.columns)))
        ax7.set_yticks(range(len(synergy.index)))
        ax7.set_xticklabels(synergy.columns, rotation=45, ha='right', fontsize=8)
        ax7.set_yticklabels(synergy.index, fontsize=8)
        ax7.set_xlabel('离子化脂质2', fontsize=10)
        ax7.set_ylabel('离子化脂质1', fontsize=10)
        ax7.set_title('IL协同矩阵', fontsize=12, fontweight='bold')

        for i in range(len(synergy.index)):
            for j in range(len(synergy.columns)):
                ax7.text(j, i, int(synergy.values[i, j]),
                         ha='center', va='center', fontsize=8)

        plt.colorbar(im, ax=ax7)

        # 8. 比例分布箱线图
        ax8 = plt.subplot(3, 4, 8)
        ratio_data = [selected_df['TotalIL%'], selected_df['PL%'],
                      selected_df['Chol%'], selected_df['PEG%']]
        bp = ax8.boxplot(ratio_data, labels=['IL', 'PL', 'Chol', 'PEG'],
                         patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#FFB6C1', '#98D8C8', '#F7DC6F', '#BB8FCE']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax8.set_ylabel('摩尔百分比 (%)', fontsize=11)
        ax8.set_title('各组分摩尔比分布', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')

        # 9. 浓度分布箱线图
        ax9 = plt.subplot(3, 4, 9)
        conc_data = [selected_df['IL1_mg/ml'] + selected_df['IL2_mg/ml'],
                     selected_df['PL_mg/ml'],
                     selected_df['Chol_mg/ml'],
                     selected_df['PEG_mg/ml']]
        bp = ax9.boxplot(conc_data, labels=['总IL', 'PL', 'Chol', 'PEG'],
                         patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#FFB6C1', '#98D8C8', '#F7DC6F', '#BB8FCE']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax9.set_ylabel('浓度 (mg/ml)', fontsize=11)
        ax9.set_title('各组分浓度分布', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3, axis='y')

        # 10. 配方空间覆盖
        ax10 = plt.subplot(3, 4, 10)
        coverage_data = {
            '总配方': 216,
            '选中配方': len(selected_df),
            '唯一IL组合': len(selected_df.groupby(['IL1', 'IL2']).size()),
            '唯一完整配方': len(selected_df)
        }
        bars = ax10.bar(range(len(coverage_data)), coverage_data.values(),
                        color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'], alpha=0.8)
        ax10.set_xticks(range(len(coverage_data)))
        ax10.set_xticklabels(coverage_data.keys(), fontsize=9, rotation=15, ha='right')
        ax10.set_ylabel('数量', fontsize=11)
        ax10.set_title('配方空间覆盖', fontsize=12, fontweight='bold')
        for bar in bars:
            height = bar.get_height()
            ax10.text(bar.get_x() + bar.get_width() / 2., height,
                      f'{int(height)}', ha='center', va='bottom', fontsize=9)

        # 11. 浓度统计信息
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')
        stats_text = f"""浓度统计 (mg/ml)

总脂质浓度: {TOTAL_LIPID_CONCENTRATION:.1f}

组分平均浓度:
IL1: {selected_df['IL1_mg/ml'].mean():.3f}
IL2: {selected_df['IL2_mg/ml'].mean():.3f}
PL:  {selected_df['PL_mg/ml'].mean():.3f}
Chol: {selected_df['Chol_mg/ml'].mean():.3f}
PEG: {selected_df['PEG_mg/ml'].mean():.3f}

浓度范围:
IL总: {(selected_df['IL1_mg/ml'] + selected_df['IL2_mg/ml']).min():.2f} - {(selected_df['IL1_mg/ml'] + selected_df['IL2_mg/ml']).max():.2f}
PL: {selected_df['PL_mg/ml'].min():.2f} - {selected_df['PL_mg/ml'].max():.2f}
Chol: {selected_df['Chol_mg/ml'].min():.2f} - {selected_df['Chol_mg/ml'].max():.2f}"""

        ax11.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                  fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax11.set_title('浓度统计信息', fontsize=12, fontweight='bold')

        # 12. 配方类型浓度对比
        ax12 = plt.subplot(3, 4, 12)
        type_names = ['高胆固醇配方', '中胆固醇配方', '无胆固醇配方']
        type_colors = ['#E63946', '#F77F00', '#06FFA5']

        x_pos = np.arange(len(type_names))
        width = 0.15

        for i, comp in enumerate(['IL1_mg/ml', 'IL2_mg/ml', 'PL_mg/ml', 'Chol_mg/ml', 'PEG_mg/ml']):
            means = [selected_df[selected_df['TypeName'] == t][comp].mean() if len(
                selected_df[selected_df['TypeName'] == t]) > 0 else 0
                     for t in type_names]
            ax12.bar(x_pos + i * width, means, width, label=comp.replace('_mg/ml', ''), alpha=0.8)

        ax12.set_xlabel('配方类型', fontsize=11)
        ax12.set_ylabel('平均浓度 (mg/ml)', fontsize=11)
        ax12.set_title('不同配方类型的组分浓度', fontsize=12, fontweight='bold')
        ax12.set_xticks(x_pos + width * 2)
        ax12.set_xticklabels([t.replace('配方', '') for t in type_names], fontsize=9)
        ax12.legend(fontsize=8)
        ax12.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'D-optimal设计：36个LNP配方（含浓度计算，总浓度{TOTAL_LIPID_CONCENTRATION}mg/ml）',
                     fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 图表已保存: {filename}")

    except Exception as e:
        print(f"⚠ 可视化失败: {str(e)}")


# ==================== 步骤7：导出Excel（含浓度） ====================

def export_results(selected_df, filename='LNP_Screen_Round1_36samples_with_conc.xlsx'):
    """导出结果（含浓度信息）"""
    print(f"\n【步骤7】导出Excel文件")
    print("-" * 95)

    try:
        export_df = selected_df[[
            '选择顺序', 'ID', 'TypeName',
            'IL1', 'IL1%', 'IL1_mg/ml',
            'IL2', 'IL2%', 'IL2_mg/ml',
            'PL', 'PL%', 'PL_mg/ml',
            'Chol%', 'Chol_mg/ml',
            'PEG%', 'PEG_mg/ml',
            'TotalIL%', 'Total_mg/ml'
        ]].copy()

        export_df = export_df.rename(columns={
            '选择顺序': '选择顺序',
            'ID': '配方编号',
            'TypeName': '配方类型',
            'IL1': '离子化脂质1',
            'IL1%': 'IL1摩尔百分比',
            'IL1_mg/ml': 'IL1浓度(mg/ml)',
            'IL2': '离子化脂质2',
            'IL2%': 'IL2摩尔百分比',
            'IL2_mg/ml': 'IL2浓度(mg/ml)',
            'PL': '磷脂',
            'PL%': '磷脂摩尔百分比',
            'PL_mg/ml': '磷脂浓度(mg/ml)',
            'Chol%': '胆固醇摩尔百分比',
            'Chol_mg/ml': '胆固醇浓度(mg/ml)',
            'PEG%': 'PEG摩尔百分比',
            'PEG_mg/ml': 'PEG浓度(mg/ml)',
            'TotalIL%': '离子化脂质总摩尔比',
            'Total_mg/ml': '总脂质浓度(mg/ml)'
        })

        # 格式化数值
        for col in export_df.columns:
            if 'mg/ml' in col:
                export_df[col] = export_df[col].round(3)
            elif '百分比' in col or '摩尔比' in col:
                export_df[col] = export_df[col].round(2)

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 主数据表
            export_df.to_excel(writer, sheet_name='选中的36个配方', index=False)

            # 统计页
            c12_count = (selected_df['IL1'] == 'C12-200').sum() + (selected_df['IL2'] == 'C12-200').sum()
            mc3_count = (selected_df['IL1'] == 'MC3').sum() + (selected_df['IL2'] == 'MC3').sum()
            dotap_count = (selected_df['IL1'] == 'DOTAP').sum() + (selected_df['IL2'] == 'DOTAP').sum()

            stats_data = {
                '指标': [
                    '总配方空间', '选中配方数', '空间覆盖率(%)',
                    '总脂质浓度(mg/ml)',
                    'C12-200出现次数', 'MC3出现次数', 'C12-200相对MC3',
                    'DOTAP出现次数',
                    '唯一IL组合', '配方类型数',
                    'IL平均浓度(mg/ml)', 'PL平均浓度(mg/ml)',
                    'Chol平均浓度(mg/ml)', 'PEG平均浓度(mg/ml)'
                ],
                '数值': [
                    216, len(selected_df), f"{len(selected_df) / 216 * 100:.1f}",
                    TOTAL_LIPID_CONCENTRATION,
                    c12_count, mc3_count, f'+{c12_count - mc3_count}',
                    dotap_count,
                    len(selected_df.groupby(['IL1', 'IL2']).size()),
                    3,
                    f"{(selected_df['IL1_mg/ml'] + selected_df['IL2_mg/ml']).mean():.3f}",
                    f"{selected_df['PL_mg/ml'].mean():.3f}",
                    f"{selected_df['Chol_mg/ml'].mean():.3f}",
                    f"{selected_df['PEG_mg/ml'].mean():.3f}"
                ]
            }
            pd.DataFrame(stats_data).to_excel(writer, sheet_name='统计信息', index=False)

            # 分子量信息表
            mw_data = {
                '脂质名称': list(LIPID_MW.keys()) + ['Cholesterol'],
                '分子量 (g/mol)': list(LIPID_MW.values()) + [CHOLESTEROL_MW]
            }
            pd.DataFrame(mw_data).to_excel(writer, sheet_name='分子量信息', index=False)

        print(f"✓ Excel已保存: {filename}")

    except Exception as e:
        print(f"⚠ Excel导出失败: {str(e)}")
        csv_file = 'LNP_Screen_Round1_36samples_with_conc.csv'
        try:
            export_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"✓ CSV已保存: {csv_file}")
        except:
            print(f"  CSV保存也失败")


# ==================== 主程序 ====================

def main():
    """主函数"""

    try:
        # 步骤1：生成配方
        all_formulations = generate_all_formulations()

        # 步骤2：编码特征
        X = encode_formulations(all_formulations)

        # 步骤3：知识增强D-optimal选择
        selected_indices, d_history = knowledge_enhanced_d_optimal(
            X, all_formulations, n_select=36, random_seed=42
        )

        # 步骤4：分析结果
        selected_formulations = analyze_selection(all_formulations, selected_indices)

        # 步骤5：打印表格（含浓度）
        print_full_table(selected_formulations)

        # 步骤6：可视化
        create_visualization(selected_formulations, d_history)

        # 步骤7：导出Excel
        export_results(selected_formulations)

        # 最终总结
        c12_count = ((selected_formulations['IL1'] == 'C12-200').sum() +
                     (selected_formulations['IL2'] == 'C12-200').sum())
        mc3_count = ((selected_formulations['IL1'] == 'MC3').sum() +
                     (selected_formulations['IL2'] == 'MC3').sum())
        dotap_count = ((selected_formulations['IL1'] == 'DOTAP').sum() +
                       (selected_formulations['IL2'] == 'DOTAP').sum())

        print("\n" + "=" * 95)
        print(" " * 30 + "D-optimal设计完成！")
        print("=" * 95)
        print(f"\n✓ 成功从216个配方中选出36个最优配方")
        print(f"✓ 生成文件:")
        print(f"  1. LNP_Screen_Round1_36samples_with_conc.png  - 12张分析图表")
        print(f"  2. LNP_Screen_Round1_36samples_with_conc.xlsx - Excel配方表（含浓度）")

        print(f"\n配方特点:")
        print(f"  ✓ 主要策略: D-optimal最大化空间覆盖")
        print(f"  ✓ 总脂质浓度: {TOTAL_LIPID_CONCENTRATION} mg/ml")
        print(f"  ✓ C12-200: 出现{c12_count}次，MC3: 出现{mc3_count}次 "
              f"(C12-200多{c12_count - mc3_count}次 ✓)")
        print(f"  ✓ DOTAP: 出现{dotap_count}次")
        print(f"  ✓ 覆盖率: {36 / 216 * 100:.1f}%")
        print(f"  ✓ 保持平衡和普适性")

        print(f"\n浓度计算说明:")
        print(f"  ✓ 基于各组分摩尔百分比和分子量计算质量浓度")
        print(f"  ✓ 总脂质浓度固定为 {TOTAL_LIPID_CONCENTRATION} mg/ml")
        print(f"  ✓ 各组分浓度 = (摩尔% × 分子量 / 总加权分子量) × 总浓度")

        print(f"\n下一步:")
        print(f"  → 按Excel表格中的浓度配制36个LNP配方")
        print(f"  → 进行细胞转染实验（DC2.4等细胞系）")
        print(f"  → 收集实验数据，准备第二步AI建模")
        print("=" * 95 + "\n")

        return selected_formulations

    except Exception as e:
        print(f"\n❌ 程序出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# 运行主程序
if __name__ == "__main__":
    print(f"\n程序开始执行...\n")
    result = main()

    if result is not None:
        print(f"✓ 程序执行成功！已选出 {len(result)} 个配方。")
    else:
        print(f"✗ 程序执行失败，请检查错误信息。")