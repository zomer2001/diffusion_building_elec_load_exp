import pandas as pd


def modify_method_names(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 确保有Method列
    if 'Method' not in df.columns:
        print("错误：CSV文件中没有'Method'列")
        return

    # 将所有方法名称转为大写
    df['Method'] = df['Method'].str.upper()

    # 获取唯一的方法名称
    unique_methods = df['Method'].unique().tolist()

    print("\n当前方法名称列表:")
    for i, method in enumerate(unique_methods, 1):
        print(f"{i}. {method}")

    # 询问用户是否要修改方法名称
    modify = input("\n是否要修改方法名称？(y/n): ").lower()
    if modify != 'y':
        print("\n没有修改方法名称，仅转换为大写")
        return df

    # 创建方法名称映射字典
    method_mapping = {}

    for method in unique_methods:
        new_name = input(f"将 '{method}' 修改为 (直接回车保持原样): ")
        if new_name.strip():
            method_mapping[method] = new_name
        else:
            method_mapping[method] = method

    # 应用修改
    df['Method'] = df['Method'].map(method_mapping)

    print("\n方法名称修改完成:")
    print(df['Method'].unique())

    return df


if __name__ == "__main__":
    csv_file = input("请输入CSV文件路径: ")

    try:
        modified_df = modify_method_names(csv_file)

        # 询问是否保存修改
        save = input("\n是否保存修改后的CSV文件？(y/n): ").lower()
        if save == 'y':
            output_file = input("请输入输出文件名 (默认为 'modified_methods.csv'): ") or 'modified_methods.csv'
            modified_df.to_csv(output_file, index=False)
            print(f"\n已保存修改后的文件到: {output_file}")
        else:
            print("\n修改未保存")

        # 显示修改后的数据预览
        print("\n修改后的数据预览:")
        print(modified_df.head())

    except Exception as e:
        print(f"\n发生错误: {str(e)}")