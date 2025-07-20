# Imports and Setup
import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from urllib.parse import urljoin
import argparse
load_dotenv()  # 加载 .env 文件

HEADER_TEMPLATE = '''# -*- coding: utf-8 -*-
"""
@File      : {filename}
@Time      : {date}
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: {description}
"""
'''  # 这个模板用于新建头文件


def add_header_to_file(file_path, description=None, update_existing=False):
    """
    完整处理文件头部的函数，包含新建和更新逻辑
    
    参数:
        file_path: 目标文件路径
        description: 要添加的描述内容(可选)，如果为None且需要时会自动生成
        update_existing: 是否更新已有头部(默认False)
    """
    try:
        # 获取基本信息
        filename = os.path.basename(file_path)
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"文件 {filename} 为空，跳过。")
            return
            
        # 检查是否已有标准头部
        has_coding = False
        has_docstring_start = False
        
        if lines:
            has_coding = lines[0].strip() == "# -*- coding: utf-8 -*-"
        if len(lines) > 1:
            has_docstring_start = lines[1].strip().startswith('"""')
            
        has_header = has_coding and has_docstring_start
        
        # 情况1: 没有头部则创建新头部
        if not has_header:
            # 如果没有传入description，使用文件内容生成
            if description is None:
                content_body = "".join(lines)
                if content_body.strip():
                    description = generate_description(content_body)
                else:
                    description = "此文件暂无描述"
                    
            new_header = HEADER_TEMPLATE.format(
                filename=filename,
                date=current_date,
                description=description
            )
            # 保留原有内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_header + ''.join(lines))
            print(f"已为 {filename} 创建新头部")
            return
            
        # 情况2: 文件有头部，检查并更新描述
        description_line_index = -1
        header_end_index = -1
        is_description_empty = False
        found_description_tag = False
        search_limit = min(len(lines), 15)  # 优化：限制搜索行数
        
        # 查找描述行和头部结束标志
        for i in range(search_limit):
            line_content = lines[i].strip()
            
            # 查找 @Description: 行
            if "@Description:" in line_content:
                found_description_tag = True
                description_line_index = i
                # 检查描述是否为空
                if line_content.split('@Description:', 1)[1].strip() == "":
                    is_description_empty = True
                else:
                    is_description_empty = False
                    # 如果描述非空，检查是否允许更新
                    if not update_existing:
                        print(f"文件 {filename} 已有描述且未启用更新，跳过。")
                        return  # 不更新则跳过
                    else:
                        # 允许更新，标记一下信息，然后继续查找头部结束符
                        print(f"文件 {filename} 已有描述，将执行更新...")
                        
            # 查找头部结束标志 '"""' (确保不是第二行的)
            if i > 1 and '"""' in line_content:
                header_end_index = i
                # 如果已经找到了 @Description: 行，就可以停止头部搜索了
                if found_description_tag:
                    break
                    
        # 处理逻辑
        if not found_description_tag:
            print(f"文件 {filename} 头部未找到 @Description: 标记，跳过。")
            return
            
        # 确定是否需要调用 LLM API
        # 条件：必须找到描述行和头部结束行，并且 (描述是空的 或者 允许更新)
        should_process = (description_line_index != -1 and header_end_index != -1) and \
                         (is_description_empty or update_existing)
                         
        if should_process:
            action_word = "更新" if not is_description_empty else "生成"
            print(f"正在为文件 {filename} {action_word}描述...")
            
            # 如果没有传入description，则自动生成
            if description is None:
                # 提取用于分析的代码体（头部之后的部分）
                content_body = "".join(lines[header_end_index + 1:])
                if not content_body.strip():
                    print(f"警告: 文件 {filename} 头部之后无实际代码内容，跳过{action_word}。")
                    return
                    
                # 调用 LLM 生成描述
                description = generate_description(content_body)
                
                # 检查 API 调用是否成功返回描述文本
                if description.startswith("["):  # 假设错误信息都以 '[' 开头
                    print(f"无法为文件 {filename} {action_word}描述: {description}")
                    return
            
            # 获取原始 @Description: 行的缩进
            original_line = lines[description_line_index]
            indentation = ""
            for char in original_line:
                if char.isspace():
                    indentation += char
                else:
                    break
                    
            # 构建新的包含描述的行
            new_description_line = f"{indentation}@Description: {description}"
            
            # 替换内存中的行列表
            lines[description_line_index] = new_description_line
            
            # 将修改后的内容写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"成功{action_word}文件 {filename} 的描述。")
            
        elif header_end_index == -1:  # 如果找到了 Description 但没找到结束的 """
            print(f"文件 {filename} 头部格式不完整 (未找到结束的 \"\"\")，跳过。")
            
    except IOError as e:
        print(f"处理文件 {file_path} 时发生 IO 错误: {e}")
    except Exception as e:
        import traceback
        print(f"处理文件 {file_path} 时发生未知错误: {e}")
        # traceback.print_exc()


def create_detailed_header_prompt(file_content: str) -> str:
    """
    根据提供的文件内容，生成用于请求 LLM 创建详细文件头描述的完整 Prompt。
    Args:
        file_content: 要分析的 Python 文件的完整代码字符串。
    Returns:
        一个包含完整指令和文件内容的字符串，可直接用作 LLM 的输入 Prompt。
    """
    detailed_prompt_template = """# 角色设定 (Role Setting)

        你是一位经验丰富的 Python 软件工程师，非常擅长阅读和理解代码，并且能够编写出专业、清晰、详尽的技术文档和代码注释。
        # 任务背景 (Task Context)
        我需要你为一个 Python 脚本文件生成文件头（Header）中的 `@Description:` 部分。这个描述应该不仅仅是一句话总结，而是要**详细**地阐述文件的功能和内容。
        # 输入信息 (Input Information)
        以下是需要你分析并生成描述的 Python 文件的**完整代码内容**:
        ```python
        {content}

        具体指令 (Instructions)

        仔细阅读并完全理解上面提供的 Python 代码。

        分析代码的核心目的、实现的主要功能模块（例如关键的类、函数、算法或处理流程）、输入/输出（如果明显）、以及它可能依赖的关键库或模块（如果对理解功能很重要）。

        基于你的分析，撰写一段详细且准确的中文描述。这段描述应该：清晰地说明这个文件的主要作用是什么。

        概述其中包含的关键组件和它们各自的职责。

        如果代码逻辑比较复杂，可以简要说明其工作流程或原理。

        确保描述内容与代码严格对应，避免猜测或添加无关信息。

        语言要专业、精练，同时也要易于理解。

        格式化与输出:在生成的描述文本中适当地插入表示换行的实际的换行符  (\n)，使得文本在文件头中能够自动换行显示，看起来像一个格式良好的段落或列表。请尽量确保每行文本长度不超过 80 个字符，以提高可读性。

        对于每个具体函数的描述，请一定单独启一行新的描述文本，绝对不要黏连两个函数的描述。
        
        你的回复必须是适合直接放在文件头 @Description: 标签后面的描述文本本身（包含必要的实际换行符  \n）。

        绝对不要包含 @Description: 这个标签。

        绝对不要包含任何代码以外的解释、问候、确认或其他无关内容。

        输出要求 (Output Requirements)

        请严格按照上述指令，只提供包含适当实际换行符 (\n) 以便自动换行的详细中文描述文本。


        """
    # 在函数调用时，使用 .format() 填充占位符
    final_prompt = detailed_prompt_template.format(content=file_content)  # 注意这里用 format
    return final_prompt


def generate_description(content):
    """调用 LLM API 生成文件描述"""
    print("-" * 10 + " Entering generate_description " + "-" * 10)

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_API_BASE")
    model_name = os.getenv("LLM_MODEL_NAME")

    print(f"DEBUG: Retrieved LLM_API_BASE = '{base_url}'")
    print(f"DEBUG: Retrieved LLM_API_KEY is set: {'Yes' if api_key else 'No'}")
    print(f"DEBUG: Retrieved LLM_MODEL_NAME = '{model_name}'")

    # 检查环境变量
    if not base_url:
        return "[错误：环境变量 LLM_API_BASE 未设置或为空]"
    if not api_key:
        return "[缺少 LLM_API_KEY 环境变量或 .env 文件未加载/配置]"
    if not model_name:
        print("警告: LLM_MODEL_NAME 环境变量未设置或为空。")  # 模型名称可能非必需，仅警告

    api_path = "chat/completions"  # API 的相对路径
    print(f"DEBUG: Using api_path = '{api_path}'")

    # 构造完整的 API URL（已修正拼接逻辑）
    try:
        if not isinstance(base_url, str):
            raise TypeError(f"base_url is not a string, but {type(base_url)}")

        # 确保 base_url 以 '/' 结尾以正确使用 urljoin
        if not base_url.endswith('/'):
            base_url_for_join = base_url + '/'
        else:
            base_url_for_join = base_url
        print(f"DEBUG: Base URL adjusted for urljoin = '{base_url_for_join}'")
        full_api_url = urljoin(base_url_for_join, api_path)
        print(f"Constructed API URL: {full_api_url}")

    except TypeError as e:
        return f"[URL 类型错误: {e}]"
    except Exception as e:
        return f"[URL 构建错误: {e}]"

    # 调用函数生成 Prompt
    prompt_for_llm = create_detailed_header_prompt(content)

    # 准备请求头和数据
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model_name,
        "messages": [
            # {"role": "system", "content": "你是一个代码文档助手。"}, # System prompt 可选
            {"role": "user", "content": prompt_for_llm}
        ],
        # 根据需要添加其他参数，例如 temperature
        # "temperature": 0.2
    }

    # 发送请求并处理响应
    try:
        response = requests.post(
            full_api_url,
            headers=headers,
            json=data,
            timeout=60  # 使用稍长的超时时间
        )
        response.raise_for_status()  # 检查 HTTP 错误 (4xx, 5xx)
        result = response.json()

        # 提取结果并只做 strip() 处理，保留内部的 \n
        if "choices" in result and result["choices"] and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
            return result["choices"][0]["message"]["content"].strip()
        else:
            # 尝试解析可能的错误信息结构
            error_msg = result.get('error', {}).get('message', f'API 返回结构异常: {result}')
            return f"[{error_msg}]"

    except requests.exceptions.Timeout as e:
        return f"[API 请求超时: {e}]"
    except requests.exceptions.ConnectionError as e:
        return f"[API 连接错误: {e}]"
    except requests.exceptions.HTTPError as e:
        # 返回包含状态码和响应体的详细错误信息
        return f"[API HTTP 错误: {e.response.status_code} - {e.response.text}]"
    except requests.exceptions.RequestException as e:
        return f"[API 请求失败: {e}]"
    except Exception as e:
        return f"[生成描述时未知错误: {e}]"


def batch_process(root_dir, update_existing=False):
    """
    批量处理指定目录下的所有 .py 文件。
    Args:
        root_dir: 要处理的根目录。
        update_existing: 是否更新已存在的描述。
    """
    processed_count = 0
    error_count = 0
    print(f"开始批量处理目录: {root_dir}")
    print(f"是否更新已存在描述: {'是' if update_existing else '否'}")  # 提示用户当前模式
    print("-" * 30)

    for folder, _, files in os.walk(root_dir):
        for name in files:
            if name.endswith('.py'):
                file_path = os.path.join(folder, name)
                print("-" * 20)  # 文件处理分隔符
                try:
                    # 将 update_existing 标志传递给处理函数
                    add_header_to_file(file_path, update_existing=update_existing)
                    processed_count += 1
                except Exception as e:
                    # 捕获 add_header_to_file 可能抛出的未预料错误
                    print(f"!!! 处理文件 {file_path} 过程中发生严重错误: {e}")
                    error_count += 1

    # 打印处理总结
    print("\n" + "=" * 30)
    print("批量处理完成。")
    print(f"总计尝试处理文件数（包括跳过）: {processed_count}")
    print(f"处理过程中发生严重错误的数量: {error_count}")
    print("=" * 30)


if __name__ == '__main__':
    # 初始化命令行参数解析器
    parser = argparse.ArgumentParser(
        description="自动为 Python 文件添加或更新文件头描述。",
        formatter_class=argparse.RawTextHelpFormatter  # 保持帮助信息格式
    )
    # 添加必需的位置参数：目标目录
    parser.add_argument(
        "target_directory",
        help="需要扫描并处理 .py 文件的目标目录路径。"
    )
    # 添加可选的标志参数：--update
    parser.add_argument(
        "--update",
        action="store_true",  # 当命令行出现 --update 时，此参数值为 True
        help="强制更新已存在的 @Description 内容。\n如果未提供此标志，则只填充空的描述行。"
    )
    # 解析命令行传入的参数
    args = parser.parse_args()

    # 从解析结果中获取目标目录
    target_directory = args.target_directory

    # 检查目标目录是否存在且是目录
    if os.path.isdir(target_directory):
        # 调用批量处理函数，传入目录和是否更新的标志
        batch_process(target_directory, update_existing=args.update)
    else:
        # 如果目录无效，打印错误信息
        print(f"错误：指定的路径 '{target_directory}' 不是一个有效的目录。")