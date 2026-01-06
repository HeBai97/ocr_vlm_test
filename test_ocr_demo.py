#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一调用三个OCR脚本进行图片解析和统计

使用方法:
    python test_ocr_demo.py test_ocr --type random --number 10
    python test_ocr_demo.py test_ocr --type all --number 100

参数说明:
    --type: 处理类型，all=处理所有图片，random=随机抽取
    --number: 处理图片数量（仅 --type random 时有效）
"""

import os
import sys
import json
import time
import random
import shutil
import pathlib
import subprocess
import tempfile
from typing import List, Dict, Tuple
from datetime import datetime


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


def get_image_extension(image_path: str) -> str:
    """获取图片文件扩展名"""
    return pathlib.Path(image_path).suffix.lower()


def run_ocr_script(conda_env: str, script_path: str, image_dir: str, type_arg: str = "all", number_arg: int = None) -> Tuple[bool, str]:
    """
    运行OCR脚本
    
    Args:
        conda_env: conda环境名称
        script_path: 脚本路径
        image_dir: 图片目录
        type_arg: 类型参数（all/random），默认为 "all"
        number_arg: 数量参数（仅 random 时使用）
        
    Returns:
        Tuple[bool, str]: (是否成功, 输出信息)
    """
    try:
        # 构建命令 - 图片目录作为第一个位置参数
        cmd = [
            "conda", "run", "-n", conda_env,
            "python", script_path, image_dir,
            "--type", type_arg
        ]
        
        # 如果是 random 模式，添加 number 参数
        if type_arg == "random" and number_arg is not None:
            cmd.extend(["--number", str(number_arg)])
        
        print(f"\n执行命令: {' '.join(cmd)}")
        
        # 运行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "命令执行超时"
    except Exception as e:
        return False, str(e)


def get_element_count_from_output_dir(output_dir: str, image_name: str, ocr_type: str) -> int:
    """
    从输出目录的JSON文件中获取元素个数
    
    Args:
        output_dir: 输出目录
        image_name: 图片名称（不含扩展名）
        ocr_type: OCR类型（paddle/dots/mineru）
        
    Returns:
        int: 元素个数，如果找不到则返回0
    """
    if not os.path.exists(output_dir):
        return 0
    
    # 根据OCR类型确定JSON文件路径
    if ocr_type == "paddle":
        # PaddleOCRVL 可能生成多个JSON文件，查找包含该图片名的
        # 通常文件名格式为: {image_name}_*.json 或直接是 {image_name}.json
        json_files = []
        for file in os.listdir(output_dir):
            if file.endswith('.json'):
                # 检查文件名是否以图片名开头
                file_stem = pathlib.Path(file).stem
                if file_stem.startswith(image_name):
                    json_files.append(os.path.join(output_dir, file))
        
        # 尝试解析每个JSON文件
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # PaddleOCRVL的JSON结构可能包含parsing_res_list
                    if 'parsing_res_list' in data:
                        return len(data['parsing_res_list'])
                    elif isinstance(data, list) and len(data) > 0:
                        # 如果是列表，可能是多个结果
                        total = 0
                        for item in data:
                            if isinstance(item, dict) and 'parsing_res_list' in item:
                                total += len(item['parsing_res_list'])
                            elif isinstance(item, list):
                                total += len(item)
                        if total > 0:
                            return total
                        return len(data)
            except Exception as e:
                continue
        return 0
    elif ocr_type == "dots":
        json_path = os.path.join(output_dir, f"{image_name}_result.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'cells' in data:
                        return len(data['cells'])
                    elif 'total_cells' in data:
                        return data['total_cells']
            except:
                pass
        return 0
    elif ocr_type == "mineru":
        json_path = os.path.join(output_dir, f"{image_name}_result.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'cells' in data:
                        return len(data['cells'])
                    elif 'total_cells' in data:
                        return data['total_cells']
            except:
                pass
        return 0
    else:
        return 0


def find_annotated_image(output_dir: str, image_name: str, ocr_type: str) -> str:
    """
    查找标注后的图片文件
    
    Args:
        output_dir: 输出目录
        image_name: 图片名称（不含扩展名）
        ocr_type: OCR类型（paddle/dots/mineru）
        
    Returns:
        str: 标注图片路径，如果找不到则返回None
    """
    if not os.path.exists(output_dir):
        return None
    
    # 查找标注图片（通常是 _annotated.jpg）
    annotated_patterns = [
        f"{image_name}_annotated.jpg",
        f"{image_name}_annotated_1.jpg",
        f"{image_name}_annotated.png",
    ]
    
    for pattern in annotated_patterns:
        annotated_path = os.path.join(output_dir, pattern)
        if os.path.exists(annotated_path):
            return annotated_path
    
    # 如果找不到，尝试列出所有文件
    try:
        for file in os.listdir(output_dir):
            if file.startswith(image_name) and ('annotated' in file.lower() or 'annot' in file.lower()):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return os.path.join(output_dir, file)
    except Exception as e:
        print(f"警告: 列出目录 {output_dir} 时出错: {e}")
    
    return None


def copy_annotated_image(annotated_path: str, output_dir: str, image_name: str, image_ext: str, ocr_type: str):
    """
    复制标注图片到输出目录并重命名
    
    Args:
        annotated_path: 源标注图片路径
        output_dir: 目标输出目录
        image_name: 图片名称（不含扩展名）
        image_ext: 图片扩展名
        ocr_type: OCR类型（paddle/dots/mineru）
    """
    if not annotated_path or not os.path.exists(annotated_path):
        print(f"警告: 未找到标注图片: {annotated_path}")
        return
    
    # 生成新的文件名：原文件名 + "_" + ocr_type + 扩展名
    new_filename = f"{image_name}_{ocr_type}{image_ext}"
    dest_path = os.path.join(output_dir, new_filename)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制文件
    try:
        shutil.copy2(annotated_path, dest_path)
        print(f"已复制标注图片到: {dest_path}")
    except Exception as e:
        print(f"复制标注图片失败: {e}")


def create_temp_image_dir(image_files: List[str], base_dir: str = None) -> str:
    """
    创建临时目录，包含指定图片文件的符号链接
    
    Args:
        image_files: 图片文件路径列表
        base_dir: 基础目录，如果为None则使用系统临时目录
        
    Returns:
        str: 临时目录路径
    """
    if base_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="ocr_selected_images_")
    else:
        temp_dir = os.path.join(base_dir, f"temp_selected_images_{int(time.time())}")
        os.makedirs(temp_dir, exist_ok=True)
    
    # 创建符号链接
    for image_path in image_files:
        image_filename = os.path.basename(image_path)
        link_path = os.path.join(temp_dir, image_filename)
        
        # 如果链接已存在，先删除
        if os.path.exists(link_path) or os.path.islink(link_path):
            try:
                os.remove(link_path)
            except:
                pass
        
        # 创建符号链接
        try:
            os.symlink(os.path.abspath(image_path), link_path)
        except Exception as e:
            # 如果符号链接失败（如Windows），则复制文件
            print(f"警告: 无法创建符号链接 {link_path}，改为复制文件: {e}")
            try:
                shutil.copy2(image_path, link_path)
            except Exception as e2:
                print(f"错误: 无法复制文件 {image_path} 到 {link_path}: {e2}")
    
    return temp_dir


def process_images_with_ocr(
    image_dir: str,
    type_arg: str = "all",
    number_arg: int = 100,
    output_dir: str = "./output"
):
    """
    使用三个OCR脚本处理图片并生成统计报告
    
    Args:
        image_dir: 图片目录
        type_arg: 处理类型（all/random）
        number_arg: 处理数量（仅random时有效）
        output_dir: 输出目录
    """
    # OCR脚本配置
    ocr_configs = [
        {
            "name": "paddle",
            "conda_env": "paddleOCR",
            "script": "./test_paddleocrvl_demo.py",
            "output_dir": "paddleocrvl_output"
        },
        {
            "name": "dots",
            "conda_env": "dots_ocr",
            "script": "./test_dotsocrvlm_demo.py",
            "output_dir": "dotsocr_output"
        },
        {
            "name": "mineru",
            "conda_env": "mineru",
            "script": "./test_mineru_image.py",
            "output_dir": "mineru_output"
        }
    ]
    
    # 查找图片文件
    print(f"\n正在查找图片文件: {image_dir}")
    all_image_files = find_image_files_recursive(image_dir)
    
    if not all_image_files:
        print(f"错误: 在 {image_dir} 目录中未找到图片文件")
        sys.exit(1)
    
    # 根据类型选择图片（先统一选择，确保三种OCR处理相同的图片）
    if type_arg == "random":
        if number_arg > len(all_image_files):
            print(f"警告: 请求数量 {number_arg} 大于找到的图片数量 {len(all_image_files)}，将处理所有图片")
            image_files = all_image_files
        else:
            image_files = random.sample(all_image_files, number_arg)
            print(f"随机抽取 {len(image_files)} 张图片（共找到 {len(all_image_files)} 张）")
    else:
        image_files = all_image_files
        print(f"找到 {len(image_files)} 张图片，开始处理所有图片...")
    
    # 如果是随机模式，创建临时目录包含选中的图片
    temp_image_dir = None
    if type_arg == "random":
        print(f"\n创建临时目录包含选中的 {len(image_files)} 张图片...")
        temp_image_dir = create_temp_image_dir(image_files, base_dir=output_dir)
        print(f"临时目录: {temp_image_dir}")
        # 使用临时目录作为图片目录
        actual_image_dir = temp_image_dir
        # 传递给OCR脚本时使用 all 模式（因为临时目录中已经是选中的图片）
        ocr_type_arg = "all"
        ocr_number_arg = None
    else:
        actual_image_dir = image_dir
        ocr_type_arg = type_arg
        ocr_number_arg = number_arg
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储统计信息
    statistics = []
    
    # 对每个OCR脚本进行处理
    for ocr_config in ocr_configs:
        ocr_name = ocr_config["name"]
        conda_env = ocr_config["conda_env"]
        script_path = ocr_config["script"]
        ocr_output_dir = ocr_config["output_dir"]
        
        print(f"\n{'='*60}")
        print(f"开始处理 OCR: {ocr_name.upper()}")
        print(f"{'='*60}")
        
        # 运行OCR脚本
        start_time = time.time()
        success, output_msg = run_ocr_script(
            conda_env, script_path, actual_image_dir, ocr_type_arg, ocr_number_arg
        )
        total_time = time.time() - start_time
        
        if not success:
            print(f"错误: {ocr_name} OCR脚本执行失败")
            print(f"错误信息: {output_msg}")
            continue
        
        print(f"{ocr_name} OCR脚本执行完成，总耗时: {total_time:.2f}秒")
        
        # 处理每张图片的统计信息
        annotated_count = 0
        total_count = len(image_files)
        
        for image_path in image_files:
            image_name = pathlib.Path(image_path).stem
            original_image_ext = get_image_extension(image_path)
            
            # 获取元素个数
            element_count = get_element_count_from_output_dir(ocr_output_dir, image_name, ocr_name)
            
            # 查找标注图片
            annotated_path = find_annotated_image(ocr_output_dir, image_name, ocr_name)
            
            # 复制标注图片到输出目录（仅当找到标注图片时）
            if annotated_path and os.path.exists(annotated_path):
                annotated_ext = get_image_extension(annotated_path)
                copy_annotated_image(annotated_path, output_dir, image_name, annotated_ext, ocr_name)
                annotated_count += 1
            else:
                # 调试信息：打印查找失败的详细信息
                if os.path.exists(ocr_output_dir):
                    files_in_dir = os.listdir(ocr_output_dir)
                    matching_files = [f for f in files_in_dir if image_name in f and ('annotated' in f.lower() or 'annot' in f.lower())]
                    if matching_files:
                        print(f"调试: 找到可能的标注文件但路径不匹配 - OCR类型: {ocr_name}, 图片名: {image_name}, 匹配文件: {matching_files}")
                    else:
                        print(f"调试: 未找到标注图片 - OCR类型: {ocr_name}, 图片名: {image_name}, 输出目录: {ocr_output_dir}, 目录存在: {os.path.exists(ocr_output_dir)}")
                else:
                    print(f"调试: 输出目录不存在 - OCR类型: {ocr_name}, 图片名: {image_name}, 输出目录: {ocr_output_dir}")
            
            # 记录统计信息（每张图片的处理时间平均分配）
            per_image_time = total_time / len(image_files) if len(image_files) > 0 else 0
            
            statistics.append({
                "ocr_type": ocr_name,
                "image_name": os.path.basename(image_path),
                "element_count": element_count,
                "processing_time": per_image_time
            })
        
        # 打印标注图片统计信息
        missing_count = total_count - annotated_count
        print(f"\n{ocr_name.upper()} OCR 标注图片统计: 成功生成 {annotated_count}/{total_count} 张标注图片")
        if missing_count > 0:
            print(f"  注意: {missing_count} 张图片未生成标注图片（可能没有解析到内容）")
    
    # 清理临时目录
    if temp_image_dir and os.path.exists(temp_image_dir):
        try:
            # 删除符号链接和临时目录
            for item in os.listdir(temp_image_dir):
                item_path = os.path.join(temp_image_dir, item)
                try:
                    if os.path.islink(item_path):
                        os.remove(item_path)
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
                except:
                    pass
            os.rmdir(temp_image_dir)
            print(f"\n已清理临时目录: {temp_image_dir}")
        except Exception as e:
            print(f"警告: 清理临时目录失败: {e}")
    
    # 生成统计报告
    generate_statistics_report(statistics, output_dir)


def generate_statistics_report(statistics: List[Dict], output_dir: str):
    """
    生成统计报告MD表格
    
    Args:
        statistics: 统计信息列表
        output_dir: 输出目录
    """
    # 按OCR类型分组
    by_ocr = {}
    for stat in statistics:
        ocr_type = stat["ocr_type"]
        if ocr_type not in by_ocr:
            by_ocr[ocr_type] = []
        by_ocr[ocr_type].append(stat)
    
    # 生成MD表格
    md_content = "# OCR处理统计报告\n\n"
    md_content += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # 为每个OCR类型生成表格
    for ocr_type in ["paddle", "dots", "mineru"]:
        if ocr_type not in by_ocr:
            continue
        
        md_content += f"## {ocr_type.upper()} OCR 统计\n\n"
        md_content += "| 图片名 | 解析出元素个数 | 处理时间(秒) |\n"
        md_content += "|--------|---------------|-------------|\n"
        
        total_elements = 0
        total_time = 0
        
        for stat in by_ocr[ocr_type]:
            image_name = stat["image_name"]
            element_count = stat["element_count"]
            processing_time = stat["processing_time"]
            
            total_elements += element_count
            total_time += processing_time
            
            md_content += f"| {image_name} | {element_count} | {processing_time:.2f} |\n"
        
        # 添加总计行
        md_content += f"| **总计** | **{total_elements}** | **{total_time:.2f}** |\n\n"
    
    # 保存报告
    report_path = os.path.join(output_dir, "statistics_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"\n统计报告已保存到: {report_path}")
    print("\n" + md_content)


if __name__ == "__main__":
    # 解析命令行参数
    args_list = sys.argv[1:]
    i = 0
    number_arg = 100
    type_arg = "all"
    image_dir = None
    
    while i < len(args_list):
        if args_list[i] == "--type" and i + 1 < len(args_list):
            type_arg = args_list[i + 1]
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--number" and i + 1 < len(args_list):
            number_arg = int(args_list[i + 1])
            args_list = args_list[:i] + args_list[i+2:]
            continue
        i += 1
    
    # 获取图片目录 - 第一个位置参数是图片目录
    if len(args_list) > 0:
        image_dir = args_list[0]
    else:
        # 默认使用test_ocr目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(script_dir, "test_ocr")
    
    if not os.path.isdir(image_dir):
        print(f"错误: {image_dir} 不是有效的目录")
        sys.exit(1)
    
    # 验证参数
    if type_arg not in ["all", "random"]:
        print(f"错误: --type 参数必须是 'all' 或 'random'")
        sys.exit(1)
    
    if type_arg == "random" and number_arg <= 0:
        print(f"错误: --type random 需要指定 --number 参数且大于0")
        sys.exit(1)
    
    print(f"图片目录: {image_dir}")
    print(f"处理类型: {type_arg}")
    if type_arg == "random":
        print(f"处理数量: {number_arg}")
    print(f"输出目录: ./output")
    
    # 执行处理
    process_images_with_ocr(
        image_dir=image_dir,
        type_arg=type_arg,
        number_arg=number_arg,
        output_dir="./output"
    )

