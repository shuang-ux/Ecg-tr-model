
# %%
import sys
import os
import pickle
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

sys.path.append("../")
sys.path.append("./")

from pymagic.EcgClassify.classify_autogluon_bin import load_ecg_feature

def evaluate_external_validation(model_dir, external_file, label_name='label1'):
    """
    评估外部验证集
    
    Args:
        model_dir: 模型保存目录（如'/database/home/wushuang/pywave/apptrpe_newdata/mid14'）
        external_file: 外部验证集文件路径
        label_name: 要评估的标签名（label1, label2, label3）
    """
    # 1. 加载外部验证集
    print(f"加载外部验证集: {external_file}")
    if external_file.endswith('.pkl'):
        with open(external_file, "rb") as f:
            df_ext = pickle.load(f)
    elif external_file.endswith('.xlsx'):
        df_ext = pd.read_excel(external_file, dtype={'subject_id': str})
    elif external_file.endswith('.csv'):
        df_ext = pd.read_csv(external_file, dtype={'subject_id': str})
    else:
        raise ValueError("请使用.pkl, .xlsx或.csv格式文件")
    
    print(f"外部验证集大小: {len(df_ext)}")
    
    # 2. 处理标签（与训练代码一致）
    def label_tr(txt):
        if not isinstance(txt, str):
            return 0
        txt = txt.strip()
        if txt == "正常": return 0
        elif txt == "少量": return 1
        elif txt == "中量": return 2
        elif txt == "大量": return 3
        else: return 0
    
    # 应用标签
    if '三尖瓣(基于超声报告)' in df_ext.columns:
        df_ext['tr_level'] = df_ext['三尖瓣(基于超声报告)'].apply(label_tr)
    else:
        raise ValueError("外部验证集需要'三尖瓣反流'列")
    
    # 确保有db列
    if 'db' not in df_ext.columns:
        df_ext['db'] = 'qinghaiai2'
    
    # 创建标签（根据label_name）
    if label_name == 'label1':
        df_ext[label_name] = (df_ext['tr_level'] > 1).astype(int)  # 中量及以上
    elif label_name == 'label2':
        df_ext[label_name] = (df_ext['tr_level'] > 0).astype(int)  # 少量及以上
    elif label_name == 'label3':
        df_ext[label_name] = np.where(df_ext['tr_level'] > 1, 2, df_ext['tr_level'])  # 0,1,2
    else:
        raise ValueError("label_name必须是'label1', 'label2'或'label3'")
    
    print(f"标签分布: {dict(df_ext[label_name].value_counts().sort_index())}")
    
    # 3. 准备数据
    df_test = df_ext[['record', 'subject_id', 'db', label_name, 'age', 'gender']].copy()
    df_test.rename(columns={label_name: 'label'}, inplace=True)
    
    # 4. 加载特征
    X = load_ecg_feature(df_test['record'], df_test['db'], alg='gri')
    
    # # 5. 合并数据
    # df_feature = pd.merge(
    #     df_test[['record', 'label', 'age', 'gender']], 
    #     X, 
    #     on='record', 
    #     how='left'
    # )


    # # 清理数据
    # df_feature = df_feature.dropna(subset=['conclusion', 'label'])
    # # df_feature = df_feature.drop(columns=['record', 'conclusion', 'rpos', 'code'], errors='ignore')
    # print(f"可用数据大小: {len(df_feature)}")
    
    # print(f"record: {len(df_feature['record'])}")

    # valid_records = df_feature['record'].dropna().unique()
    # print(f"唯一record: {len(valid_records)}")

    # df_used_raw = df_ext[df_ext['record'].isin(valid_records)].copy()

    # df_feature = df_feature.drop(columns=['record', 'conclusion', 'rpos', 'code'], errors='ignore')
    # print("\n========== 外部验证数据使用情况 ==========")
    # print(f"原始外部验证集样本数 : {len(df_ext)}")
    # print(f"最终进入模型样本数   : {len(df_used_raw)}")
    # print(f"进入模型的受试者数   : {df_used_raw['subject_id'].nunique()}")
    # print(f"丢弃样本数           : {len(df_ext) - len(df_used_raw)}")
    # print(f"使用率               : {len(df_used_raw)/len(df_ext):.2%}")
    # print("===========================================\n")
    # 5. 合并数据
    df_feature = pd.merge(
        df_test[['record', 'label', 'age', 'gender']],
        X,
        on='record',
        how='left'
    )

    # ---------- 数据丢弃原因诊断 ----------
    feature_records = set(X['record'].dropna().unique()) if 'record' in X.columns else set()

    def diagnose_drop(row):
        reasons = []
        if pd.isna(row['record']):
            reasons.append('record为空')
        elif row['record'] not in feature_records:
            reasons.append('无ECG特征(load_ecg_feature未返回该record)')
        elif 'conclusion' in df_feature.columns and pd.isna(row.get('conclusion')):
            reasons.append('conclusion缺失')
        if pd.isna(row.get('label')):
            reasons.append('label缺失')
        return ' + '.join(reasons) if reasons else '保留'

    df_feature['_drop_reason'] = df_feature.apply(diagnose_drop, axis=1)

    record2sid = df_test.set_index('record')['subject_id'].to_dict()
    df_dropped_detail = df_feature[df_feature['_drop_reason'] != '保留'].copy()
    df_dropped_detail['subject_id'] = df_dropped_detail['record'].map(record2sid)
    df_dropped_detail = df_dropped_detail[
        ['record', 'subject_id', 'label', 'age', 'gender', '_drop_reason']
    ]

    print("\n========== 数据丢弃原因统计 ==========")
    for reason, cnt in df_feature['_drop_reason'].value_counts().items():
        print(f"  {reason}: {cnt} 条")
    print("=======================================\n")

    df_feature = df_feature.drop(columns=['_drop_reason'])

    # 清理数据：label 必清洗；conclusion 仅在存在时才参与
    drop_cols = ['label']
    if 'conclusion' in df_feature.columns:
        drop_cols.append('conclusion')
    df_feature = df_feature.dropna(subset=drop_cols)

    # 打印最终进入模型的标签分布（原逻辑保留）
    print("\n========== 最终进入模型的数据标签分布 ==========")
    print(f"标签分布: {dict(df_feature['label'].value_counts().sort_index())}")
    print(f"总样本数: {len(df_feature)}")
    for label_val, count in df_feature['label'].value_counts().sort_index().items():
        print(f"  类别 {label_val}: {count} 个 ({count/len(df_feature)*100:.2f}%)")
    print("================================================\n")

    # 先计算真正进入模型的原始样本（依赖 record，放在删列之前）
    valid_records = df_feature['record'].dropna().unique()
    df_used_raw = df_ext[df_ext['record'].isin(valid_records)].copy()

    # 预测前删除标识/文本列，避免作为特征传入模型
    df_feature = df_feature.drop(
        columns=['record', 'conclusion', 'rpos', 'code','nBeats'], errors='ignore'
    )

    # 6. 加载模型
    predictor = TabularPredictor.load(model_dir)
    save_dir = os.path.join(model_dir, label_name + "_external_results_new")
    os.makedirs(save_dir, exist_ok=True)

    raw_data_file = os.path.join(save_dir, "external_samples_used_new.csv")
    df_used_raw.to_csv(raw_data_file, index=False)
    print(f"进入模型的原始样本已保存至: {raw_data_file}")
    
    # 预测
    y_pred = predictor.predict(df_feature)
    y_prob = predictor.predict_proba(df_feature)
    y_true = df_feature['label']
    
    # 8. 计算指标
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # 保存分类报告
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{save_dir}/classification_report.csv")
    print(f"\n准确率: {report['accuracy']:.4f}")
    
    # AUC
    if y_prob.shape[1] == 2:  # 二分类
        y_score = y_prob.iloc[:, 1]
        auc = roc_auc_score(y_true, y_score)
        print(f"AUC: {auc:.4f}")
    else:  # 多分类
        auc_macro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        auc_micro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='micro')
        print(f"AUC_macro: {auc_macro:.4f}")
        print(f"AUC_micro: {auc_micro:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in predictor.class_labels],
        columns=[f"pred_{c}" for c in predictor.class_labels]
    )
    cm_df.to_csv(f"{save_dir}/confusion_matrix.csv")
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'y_true': y_true.values,
        'y_pred': y_pred.values
    })
    for i, col in enumerate(y_prob.columns):
        results_df[f'prob_{col}'] = y_prob.iloc[:, i].values
    results_df.to_csv(f"{save_dir}/predictions.csv", index=False)
    
    # 9. 打印每类指标
    print("\n详细指标:")
    for cls in sorted([k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]):
        if isinstance(report[cls], dict):
            print(f"类别 {cls}: 精确率={report[cls]['precision']:.3f}, "
                  f"召回率={report[cls]['recall']:.3f}, "
                  f"F1={report[cls]['f1-score']:.3f}")
    
    print(f"\n结果已保存到: {save_dir}")
    
    return {
        'predictor': predictor,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'report': report
    }

# 使用示例
if __name__ == "__main__":
    # 示例1: 评估label1模型
    print("评估label1模型...")
    results_label1 = evaluate_external_validation(
        model_dir='/database/home/wushuang/pywave/apptrpe/Tr_new',
        external_file='/database/home/wushuang/pywave/apptrpe/qinghaiai2_ecg_echo_独立去除极端值与九月份之前数据.xlsx',  # 替换为你的外部验证集路径
        label_name='label1'
    )
    
    # 示例2: 批量评估所有标签
    # for label in ['label1', 'label2', 'label3']:
    #     print(f"\n评估{label}模型...")
    #     try:
    #         results = evaluate_external_validation(
    #             model_dir='/database/home/wushuang/pywave/apptrpe_newdata/mid14',
    #             external_file='path/to/external_data.xlsx',
    #             label_name=label
    #         )
    #     except Exception as e:
    #         print(f"评估{label}失败: {e}")
# %%
