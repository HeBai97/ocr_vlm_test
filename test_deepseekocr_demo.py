import json
import os
import pathlib
import sys
import random
from PIL import Image, ImageDraw, ImageFont
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from typing import List, Optional


def save_ocr_result_to_json(ocr_text: str, image_path: str, output_path: str):
    """
    将 OCR 结果保存为 JSON 格式
    
    Args:
        ocr_text: OCR 识别的文本内容
        image_path: 输入图片路径
        output_path: 输出 JSON 文件路径
    """
    result = {
        "image_path": image_path,
        "ocr_text": ocr_text,
        "text_length": len(ocr_text),
        "line_count": len(ocr_text.split('\n'))
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"JSON 结果已保存到: {output_path}")


def save_ocr_result_to_markdown(ocr_text: str, image_path: str, output_path: str):
    """
    将 OCR 结果保存为 Markdown 格式
    
    Args:
        ocr_text: OCR 识别的文本内容
        image_path: 输入图片路径
        output_path: 输出 Markdown 文件路径
    """
    image_name = pathlib.Path(image_path).name
    
    # 先计算行数，避免在 f-string 中使用反斜杠
    line_count = len(ocr_text.split('\n'))
    
    markdown_content = f"""# OCR 识别结果

**图片路径**: `{image_path}`  
**图片名称**: `{image_name}`  
**文本长度**: {len(ocr_text)} 字符  
**行数**: {line_count} 行

---

## 识别文本内容

```
{ocr_text}
```

---
*由 DeepSeek OCR 生成*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Markdown 结果已保存到: {output_path}")


def find_image_files_recursive(directory: str, image_extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif')) -> List[str]:
    """
    递归查找目录下所有图片文件
    
    Args:
        directory: 要搜索的目录路径
        image_extensions: 图片文件扩展名元组
        
    Returns:
        List[str]: 所有找到的图片文件路径列表
    """
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files


def parse_image_with_deepseekocr(image_path: str,
                                  llm: LLM = None,
                                  model_name: str = "models/DeepSeek-OCR",
                                  prompt: str = "<image>\nFree OCR.",
                                  temperature: float = 0.0,
                                  max_tokens: int = 8192,
                                  ngram_size: int = 30,
                                  window_size: int = 90,
                                  whitelist_token_ids: set = None,
                                  output_dir: str = "deepseekocr_output"):
    """
    使用 DeepSeek OCR (vLLM) 解析图片并生成结果文件
    
    Args:
        image_path: 输入图片路径
        llm: 可选的 LLM 实例，如果提供则复用（避免重复初始化）
        model_name: 模型名称，默认为 "models/DeepSeek-OCR"
        prompt: OCR 提示词，默认为 "<image>\nFree OCR."
        temperature: 采样温度，默认为 0.0
        max_tokens: 最大生成 token 数，默认为 8192
        ngram_size: N-gram 大小，默认为 30
        window_size: 窗口大小，默认为 90
        whitelist_token_ids: 白名单 token ID 集合，默认为 {128821, 128822} (<td>, </td>)
        output_dir: 输出目录，默认为 "deepseekocr_output"
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果没有提供 llm，则创建一个新的
    if llm is None:
        print(f"正在初始化 DeepSeek OCR 模型: {model_name}")
        llm = LLM(
            model=model_name,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            logits_processors=[NGramPerReqLogitsProcessor]
        )
        print("模型初始化完成")
    
    print(f"解析图片: {image_path}")
    
    # 打开图片
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"打开图片失败: {e}")
        return None
    
    # 准备输入
    model_input = [{
        "prompt": prompt,
        "multi_modal_data": {"image": image}
    }]
    
    # 设置白名单 token IDs（默认包含 <td> 和 </td>）
    if whitelist_token_ids is None:
        whitelist_token_ids = {128821, 128822}
    
    # 创建采样参数
    sampling_param = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        # ngram logit processor args
        extra_args=dict(
            ngram_size=ngram_size,
            window_size=window_size,
            whitelist_token_ids=whitelist_token_ids,
        ),
        skip_special_tokens=False,
    )
    
    # 生成输出
    try:
        model_outputs = llm.generate(model_input, sampling_param)
    except Exception as e:
        print(f"解析失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    if not model_outputs or len(model_outputs) == 0:
        print("解析失败，未返回结果")
        return None
    
    # 提取 OCR 文本
    ocr_text = model_outputs[0].outputs[0].text
    
    if not ocr_text:
        print("解析失败，返回的文本为空")
        return None
    
    image_name = pathlib.Path(image_path).stem
    
    # 保存 JSON 和 Markdown
    json_path = os.path.join(output_dir, f"{image_name}.json")
    markdown_path = os.path.join(output_dir, f"{image_name}.md")
    
    save_ocr_result_to_json(ocr_text, image_path, json_path)
    save_ocr_result_to_markdown(ocr_text, image_path, markdown_path)
    
    # 打印结果摘要
    line_count = len(ocr_text.split('\n'))
    print(f"OCR 识别完成，文本长度: {len(ocr_text)} 字符，行数: {line_count} 行")
    print(f"结果已保存到: {output_dir}")
    
    return {
        'ocr_text': ocr_text,
        'output_dir': output_dir,
        'json_path': json_path,
        'markdown_path': markdown_path
    }


if __name__ == "__main__":
    # 默认配置
    model_name = "models/DeepSeek-OCR"
    prompt = "<image>\nFree OCR."
    temperature = 0.0
    max_tokens = 8192
    ngram_size = 30
    window_size = 90
    whitelist_token_ids = {128821, 128822}  # <td>, </td>
    output_dir = "deepseekocr_output"
    
    # 默认处理 bookpic 目录中的所有图片
    images_dir = "bookpic"
    
    # 解析命令行参数
    args_list = sys.argv[1:]
    i = 0
    number_arg = None
    type_arg = "all"  # 默认处理所有图片
    
    while i < len(args_list):
        if args_list[i] == "--model" and i + 1 < len(args_list):
            model_name = args_list[i + 1]
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--prompt" and i + 1 < len(args_list):
            prompt = args_list[i + 1]
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--temperature" and i + 1 < len(args_list):
            temperature = float(args_list[i + 1])
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--max-tokens" and i + 1 < len(args_list):
            max_tokens = int(args_list[i + 1])
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--ngram-size" and i + 1 < len(args_list):
            ngram_size = int(args_list[i + 1])
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--window-size" and i + 1 < len(args_list):
            window_size = int(args_list[i + 1])
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--output-dir" and i + 1 < len(args_list):
            output_dir = args_list[i + 1]
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--number" and i + 1 < len(args_list):
            number_arg = int(args_list[i + 1])
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--type" and i + 1 < len(args_list):
            type_arg = args_list[i + 1]
            args_list = args_list[:i] + args_list[i+2:]
            continue
        i += 1
    
    if len(args_list) > 0:
        # 如果提供了命令行参数，使用指定的图片或目录
        input_path = args_list[0]
        if os.path.isfile(input_path):
            # 单个图片文件
            parse_image_with_deepseekocr(
                input_path,
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                ngram_size=ngram_size,
                window_size=window_size,
                whitelist_token_ids=whitelist_token_ids,
                output_dir=output_dir
            )
            sys.exit(0)
        elif os.path.isdir(input_path):
            # 目录
            images_dir = input_path
        else:
            print(f"错误: {input_path} 不是有效的文件或目录")
            sys.exit(1)
    
    # 处理 images_dir 目录中的所有图片
    if os.path.isdir(images_dir):
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif')
        
        # 递归查找所有图片文件
        all_image_files = find_image_files_recursive(images_dir, image_extensions)
        
        if not all_image_files:
            print(f"在 {images_dir} 目录中未找到图片文件")
            sys.exit(1)
        
        # 根据 --type 参数决定处理方式
        if type_arg == "random":
            # 随机抽取模式
            if number_arg is None or number_arg <= 0:
                print("错误: --type random 需要指定 --number 参数且大于0")
                sys.exit(1)
            if number_arg > len(all_image_files):
                print(f"警告: 请求数量 {number_arg} 大于找到的图片数量 {len(all_image_files)}，将处理所有图片")
                image_files = all_image_files
            else:
                image_files = random.sample(all_image_files, number_arg)
                print(f"随机抽取 {len(image_files)} 张图片（共找到 {len(all_image_files)} 张）")
        elif type_arg == "all":
            # 处理所有图片
            image_files = all_image_files
            print(f"找到 {len(image_files)} 张图片，开始处理所有图片...")
        else:
            # 默认处理所有图片
            image_files = all_image_files
            print(f"找到 {len(image_files)} 张图片，开始处理所有图片...")
        
        print(f"实际处理 {len(image_files)} 张图片")
        print(f"模型名称: {model_name}")
        print(f"提示词: {prompt}")
        print(f"温度: {temperature}")
        print(f"最大 tokens: {max_tokens}")
        print(f"N-gram 大小: {ngram_size}")
        print(f"窗口大小: {window_size}")
        print(f"输出目录: {output_dir}")
        
        # 创建 LLM 实例（只初始化一次，复用）
        print("\n正在初始化 DeepSeek OCR 模型（仅初始化一次，后续图片将复用）...")
        llm = LLM(
            model=model_name,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            logits_processors=[NGramPerReqLogitsProcessor]
        )
        print("模型初始化完成，开始处理图片...\n")
        
        for idx, image_path in enumerate(image_files, 1):
            try:
                image_file = os.path.relpath(image_path, images_dir)
            except ValueError:
                image_file = image_path
            print(f"\n[{idx}/{len(image_files)}] 处理: {image_file}")
            try:
                # 传入 llm 实例，避免重复初始化
                parse_image_with_deepseekocr(
                    image_path,
                    llm=llm,  # 复用同一个 LLM 实例
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    ngram_size=ngram_size,
                    window_size=window_size,
                    whitelist_token_ids=whitelist_token_ids,
                    output_dir=output_dir
                )
            except Exception as e:
                print(f"处理 {image_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n所有图片处理完成！")
    else:
        print(f"错误: {images_dir} 目录不存在")
        sys.exit(1)

