import json

def convert_test_data(input_file, output_file):
    """
    转换测试集数据格式
    :param input_file: 原始测试集文件路径
    :param output_file: 转换后的文件路径
    """
    # 1. 读取原始测试集
    with open(input_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)  # 假设原始数据是JSON数组

    # 2. 转换数据格式 - 添加固定的instruction
    converted_data = []
    for item in test_data:
        converted_item = {
            "instruction": "作为仇恨言论分析系统，你的任务是从中文社交媒体文本中提取仇恨四元组。"
    "输出格式：Target | Argument | Targeted Group | Hateful [END]（多组用[SEP]分隔）。"
    "规则："
    "1. 评论对象(Target)：无具体目标时写NULL"
    "2. 论点(Argument)：包含仇恨语义的文本片段"
    "3. 目标群体(Targeted Group)：Racism/Region/Sexism/LGBTQ/others/non-hate"
    "4. 是否仇恨(Hateful)：hate/non-hate"
    "5. 严格保留' | '和'[SEP]'空格"
    "示例："
    "输入「老黑我是真的讨厌」→ 老黑 | 讨厌 | 种族 | hate [END]"
    "输入「你可真是头蠢驴」→ 你 | 蠢驴 | non-hate | non-hate [END]"
    "现在处理文本：",  # 添加固定指令文本
            "input": item["content"]  # 原始内容作为输入
        }
        converted_data.append(converted_item)

    # 3. 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"测试集转换完成！共转换 {len(converted_data)} 条数据，结果已保存至 {output_file}")

# 使用示例
convert_test_data('test1.json', 'test0.json')