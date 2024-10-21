import random


def split_string(input_str):
    # 初始化结果列表
    result = []

    # 每次迭代处理2到3个字母
    i = 0
    while i < len(input_str):
        fragment_length = random.choice([2, 3])  # 随机选择片段长度为2或3
        if i + fragment_length <= len(input_str):
            result.append(input_str[i:i + fragment_length])
            i += fragment_length
        else:
            # 处理最后一个片段
            remaining_length = len(input_str) - i
            if remaining_length == 3:  # 如果剩余长度为3，则直接添加
                result.append(input_str[i:])
            else:  # 如果剩余长度为2，则随机选择长度为2或3
                result.append(input_str[i:i + random.choice([2, 3])])
            break

    # 如果最后一个片段长度为1，与前一个片段合并
    if len(result[-1]) == 1:
        last_fragment = result.pop()
        last_fragment_2 = result.pop()

        if len(last_fragment_2) == 2:
            result.append(last_fragment_2+last_fragment)
        else:
            new_str = last_fragment_2+last_fragment
            result.append(new_str[:2])
            result.append(new_str[2:])

    # 返回结果
    return result


# 输入字符串
input_str_list = ["ab", "abc", "asdf", "asdfg", "asdfgh", "asdfghj", "asdfghjk", "asdfghjkl", "asdfgqwert",
                  "asdfghjuytr", "qwertyuioplk", "asdfghjkiuytr"]

# 调用函数并打印结果
for input_str in input_str_list:
    print("-" * 30)
    print(input_str)
    result_list = split_string(input_str)
    print(result_list)
    merge_str = "".join(result_list)

    assert merge_str == input_str,"error split, "+input_str
