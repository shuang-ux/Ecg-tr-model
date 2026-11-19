# %%
#------------------------------------包含年龄性别------------------------------------
import sys
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from autogluon.common.utils.resource_utils import ResourceManager
import xgboost as xgb
import re

# 设置路径
sys.path.append("../")
sys.path.append("./")

# 重新导入模块
module_name = 'pymagic.EcgClassify.classify_autogluon_bin'
if module_name in sys.modules:
    del sys.modules[module_name]

from pymagic.EcgClassify.classify_autogluon_bin import classify_autogluon

# 1. 数据加载
def load_data():
    excel_path = '/database/home/wushuang/pywave/apptrpe/qinghaiai_ecg_echo_v10.1.xlsx'
    df = pd.read_excel(excel_path, dtype={'subject_id': str})
    print(f"从Excel文件加载数据，共{len(df)}条记录")

    return df
data=load_data()
# print(data.shape)
# print(list(data))
# %%
# 2. 三尖瓣反流分级函数
def label_tr(txt, mode='all'):
    """
    三尖瓣反流分级：
    0: 正常
    1: 少量
    2: 中量  
    3: 大量
    
    Parameters:
    -----------
    mode : str
        'all': 全部级别
        'exclude_small': 排除少量(1)，用于二分类
    """
    if not isinstance(txt, str):
        return 0
    
    txt = txt.strip()
    
    # 匹配文字内容
    if txt == "正常":
        return 0
    elif txt == "少量":
        return 1
    elif txt == "中量":
        return 2
    elif txt == "大量":
        return 3
    else:
        return 0  # 未知情况默认为0
    

# 3. 创建不同标签的函数
def create_labels(df, mode='all'):
    """
    创建不同的标签
    
    Parameters:
    -----------
    mode : str
        'all': 使用所有数据
        'exclude_small': 排除少量数据
    """
    df_processed = df.copy()
    
    # 应用三尖瓣反流分级
    df_processed['tr_level'] = df_processed['三尖瓣反流'].apply(label_tr)
    df_processed['db'] = 'qinghaiai'
    # df_processed['age_group'] = df_processed['age'].apply(get_age_group)
    # print(f"原始三尖瓣反流分布:")
    # print(df_processed['tr_level'].value_counts().sort_index())
    
    # 根据模式处理
    if mode == 'exclude_small':
        initial_count = len(df_processed)
        df_processed = df_processed[df_processed['tr_level'] != 1]  # 删除tr_level=1的行
        deleted_count = initial_count - len(df_processed)
        print(f"\n删除少量(1)数据: {deleted_count} 条")
        print(f"处理后三尖瓣反流分布:")
        print(df_processed['tr_level'].value_counts().sort_index())
        
        # 只创建label1
        df_processed['label1'] = (df_processed['tr_level'] > 1).astype(int)
        return df_processed, ['label1']
    
    else:  # mode == 'all'
        # 创建多种标签
        # label1: TR 中量及以上
        df_processed['label1'] = (df_processed['tr_level'] > 1).astype(int)
        # label2: TR 少量及以上
        df_processed['label2'] = (df_processed['tr_level'] > 0).astype(int)
        # label3: 多分类，中量/大量合并为2
        df_processed['label3'] = np.where(df_processed['tr_level'] > 1, 2, df_processed['tr_level'])
        
        print("\n各标签分布:")
        for label in ['label1', 'label2', 'label3']:
            print(f"{label}: {df_processed[label].value_counts().sort_index().to_dict()}")
        
        return df_processed, ['label1', 'label2', 'label3']


# 6. 主工作流程函数
def main_tr_workflow(mode='all', labels_to_train=None, check_gpu=True):
    """
    三尖瓣反流分类主工作流程
    
    Parameters:
    -----------
    mode : str
        'all': 使用所有数据
        'exclude_small': 排除少量数据
    labels_to_train : list
        要训练的标签列表
    check_gpu : bool
        是否检查GPU
    """
    # 加载数据
    df = load_data()
    # print(f"数据列名: {list(df.columns)}")
    # print(f"数据大小: {len(df)}")
    
    # 创建标签
    df_processed, default_labels = create_labels(df, mode)
    
    # 确定要训练的标签
    if labels_to_train is None:
        labels_to_train = default_labels
    
    # 打印系统信息
    if check_gpu:
        print("\n系统信息:")
        print(f"GPU数量: {ResourceManager.get_gpu_count()}")
        print(f"XGBoost版本: {xgb.__version__}")
    
    # 模型训练
    for label in labels_to_train:
        print(f"\n{'='*60}")
        print(f"训练标签: {label}")
        print(f"类别分布: {dict(df_processed[label].value_counts().sort_index())}")
        
        
        # 准备数据
        df_classify = df_processed[
            ['record', 'subject_id', 'db', label,'age','gender']
        ].copy()
        
        # 设置超参数
        hyperparameters = {'GBM': {}, 'CAT': {}, 'RF': {}, 'XGB': {}}
        
        # 设置保存路径
        if mode == 'exclude_small':
            savepath = f'/database/home/wushuang/pywave/apptrpe/mid4/{label}'
        else:
            savepath = (
            f'/database/home/wushuang/pywave/apptrpe/'
            f'Tr_new'
        )
        
        # eval_metric = 'f1'
        
        # 根据标签类型选择训练参数
        if label == 'label3':  # 三分类
            classify_autogluon(
                df_classify,
                savepath=savepath,
                label_column=label,
                eval_metric='f1',
                hyperparameters=hyperparameters
            )
        else:  # 二分类
            classify_autogluon(
                df_classify,
                savepath=savepath,
                label_column=label,
                eval_metric='f1',
                hyperparameters=hyperparameters
            )
        
        print(f"模型已保存到: {savepath}")


# 8. 执行主程序
if __name__ == "__main__":
    # 版本1：使用所有数据
    print("="*80)
    print("版本1:使用所有三尖瓣反流数据")
    print("="*80)
    main_tr_workflow(mode='all', labels_to_train=['label1'])
    
    # # 版本2：排除少量反流数据
    # print("\n" + "="*80)
    # print("版本2:排除少量三尖瓣反流数据")
    # print("="*80)
    # main_tr_workflow(mode='exclude_small', labels_to_train=['label1'])

# %%
#------------------------------------------不含年龄性别----------------------------
import sys
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from autogluon.common.utils.resource_utils import ResourceManager
import xgboost as xgb
import re

# 设置路径
sys.path.append("../")
sys.path.append("./")

# 清除并重新导入模块
module_name = 'pymagic.EcgClassify.classify_autogluon_bin'
if module_name in sys.modules:
    del sys.modules[module_name]

from pymagic.EcgClassify.classify_autogluon_bin import classify_autogluon

# 1. 数据加载函数
def load_data():
    excel_path = '/database/home/wushuang/pywave/apptrpe/qinghaiai_ecg_echo_v10.1.xlsx'
    df = pd.read_excel(excel_path, dtype={'subject_id': str})
    print(f"从Excel文件加载数据，共{len(df)}条记录")
    return df
data=load_data()
# print(data.shape)
# print(list(data))

# 2. 三尖瓣反流分级函数
def label_tr(txt, mode='all'):
    """
    三尖瓣反流分级：
    0: 正常
    1: 少量
    2: 中量  
    3: 大量
    
    Parameters:
    -----------
    mode : str
        'all': 全部级别
        'exclude_small': 排除少量(1)，用于二分类
    """
    if not isinstance(txt, str):
        return 0
    
    txt = txt.strip()
    
    # 直接匹配文字内容
    if txt == "正常":
        return 0
    elif txt == "少量":
        return 1
    elif txt == "中量":
        return 2
    elif txt == "大量":
        return 3
    else:
        return 0  # 未知情况默认为0
    

# 3. 创建不同标签的函数
def create_labels(df, mode='all'):
    """
    创建不同的标签
    
    Parameters:
    -----------
    mode : str
        'all': 使用所有数据
        'exclude_small': 排除少量数据
    """
    df_processed = df.copy()
    
    # 应用三尖瓣反流分级
    df_processed['tr_level'] = df_processed['三尖瓣反流'].apply(label_tr)
    df_processed['db'] = 'qinghaiai'
    # df_processed['age_group'] = df_processed['age'].apply(get_age_group)
    # print(f"原始三尖瓣反流分布:")
    # print(df_processed['tr_level'].value_counts().sort_index())
    
    # 根据模式处理
    if mode == 'exclude_small':
        initial_count = len(df_processed)
        df_processed = df_processed[df_processed['tr_level'] != 1]  # 删除tr_level=1的行
        deleted_count = initial_count - len(df_processed)
        print(f"\n删除少量(1)数据: {deleted_count} 条")
        print(f"处理后三尖瓣反流分布:")
        print(df_processed['tr_level'].value_counts().sort_index())
        
        # 只创建label1
        df_processed['label1'] = (df_processed['tr_level'] > 1).astype(int)
        return df_processed, ['label1']
    
    else:  # mode == 'all'
        # 创建多种标签
        # label1: TR 中量及以上
        df_processed['label1'] = (df_processed['tr_level'] > 1).astype(int)
        # label2: TR 少量及以上
        df_processed['label2'] = (df_processed['tr_level'] > 0).astype(int)
        # label3: 多分类，中量/大量合并为2
        df_processed['label3'] = np.where(df_processed['tr_level'] > 1, 2, df_processed['tr_level'])
        
        print("\n各标签分布:")
        for label in ['label1', 'label2', 'label3']:
            print(f"{label}: {df_processed[label].value_counts().sort_index().to_dict()}")
        
        return df_processed, ['label1', 'label2', 'label3']


# 6. 主工作流程函数
def main_tr_workflow(mode='all', labels_to_train=None, check_gpu=True):
    """
    三尖瓣反流分类主工作流程
    
    Parameters:
    -----------
    mode : str
        'all': 使用所有数据
        'exclude_small': 排除少量数据
    labels_to_train : list
        要训练的标签列表
    check_gpu : bool
        是否检查GPU
    """
    # 加载数据
    df = load_data()
    # print(f"数据列名: {list(df.columns)}")
    # print(f"数据大小: {len(df)}")
    
    # 创建标签
    df_processed, default_labels = create_labels(df, mode)
    
    # 确定要训练的标签
    if labels_to_train is None:
        labels_to_train = default_labels
    
    # 打印系统信息
    if check_gpu:
        print("\n系统信息:")
        print(f"GPU数量: {ResourceManager.get_gpu_count()}")
        print(f"XGBoost版本: {xgb.__version__}")
    
    # 模型训练
    for label in labels_to_train:
        print(f"\n{'='*60}")
        print(f"训练标签: {label}")
        print(f"类别分布: {dict(df_processed[label].value_counts().sort_index())}")
        
        
        # 准备数据
        df_classify = df_processed[
            ['record', 'subject_id', 'db', label]
        ].copy()
        
        # 设置超参数
        hyperparameters = {'GBM': {}, 'CAT': {}, 'RF': {}, 'XGB': {}}
        
        # 设置保存路径
        if mode == 'exclude_small':
            savepath = f'/database/home/wushuang/pywave/apptrpe/mid4/{label}'
        else:
            savepath = (
            f'/database/home/wushuang/pywave/apptrpe/'
            f'Tr_new_onlyEcg'
        )
        
        # eval_metric = 'f1'
        
        # 根据标签类型选择训练参数
        if label == 'label3':  # 三分类
            classify_autogluon(
                df_classify,
                savepath=savepath,
                label_column=label,
                eval_metric='f1',
                hyperparameters=hyperparameters
            )
        else:  # 二分类
            classify_autogluon(
                df_classify,
                savepath=savepath,
                label_column=label,
                eval_metric='f1',
                hyperparameters=hyperparameters
            )
        
        print(f"模型已保存到: {savepath}")


# 8. 执行主程序
if __name__ == "__main__":
    # 版本1：使用所有数据
    print("="*80)
    print("版本1:使用所有三尖瓣反流数据")
    print("="*80)
    main_tr_workflow(mode='all', labels_to_train=['label1'])
    
    # # 版本2：排除少量反流数据
    # print("\n" + "="*80)
    # print("版本2:排除少量三尖瓣反流数据")
    # print("="*80)
    # main_tr_workflow(mode='exclude_small', labels_to_train=['label1'])

# %%
