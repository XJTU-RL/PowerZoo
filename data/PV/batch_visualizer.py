#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量PV数据可视化处理脚本
处理指定目录下的所有数据文件，生成综合可视化分析
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
import glob

# 导入主要的可视化类
from data_visualizer import PVDataVisualizer

class BatchPVVisualizer:
    def __init__(self, input_dir, output_base_dir="batch_visualization_results"):
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        self.processed_files = []
        self.failed_files = []
        
        # 创建主输出目录
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"批量处理配置:")
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_base_dir}")
    
    def find_data_files(self):
        """查找所有数据文件"""
        # 支持多种文件格式
        patterns = ['*.txt', '*.csv', '*.dat']
        data_files = []
        
        for pattern in patterns:
            files = list(self.input_dir.glob(pattern))
            data_files.extend(files)
        
        # 按文件名排序
        data_files.sort()
        
        print(f"找到 {len(data_files)} 个数据文件:")
        for file in data_files:
            print(f"  - {file.name}")
        
        return data_files
    
    def process_single_file(self, file_path):
        """处理单个文件"""
        try:
            print(f"\n{'='*60}")
            print(f"正在处理: {file_path.name}")
            print(f"{'='*60}")
            
            # 为每个文件创建独立的输出目录
            file_stem = file_path.stem  # 不包含扩展名的文件名
            file_output_dir = self.output_base_dir / file_stem
            
            # 创建可视化器
            visualizer = PVDataVisualizer(file_output_dir)
            
            # 生成综合报告
            visualizer.generate_comprehensive_report(str(file_path))
            
            self.processed_files.append({
                'file': file_path.name,
                'output_dir': file_output_dir,
                'status': 'success',
                'timestamp': datetime.now()
            })
            
            print(f"✓ {file_path.name} 处理完成")
            
        except Exception as e:
            print(f"✗ 处理 {file_path.name} 时出错: {str(e)}")
            self.failed_files.append({
                'file': file_path.name,
                'error': str(e),
                'timestamp': datetime.now()
            })
    
    def generate_batch_summary(self):
        """生成批量处理摘要报告"""
        print(f"\n{'='*60}")
        print("生成批量处理摘要报告...")
        print(f"{'='*60}")
        
        # 创建摘要报告内容
        summary_content = f"""
# PV数据批量可视化分析报告

## 处理概览
- 处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 输入目录: {self.input_dir}
- 输出目录: {self.output_base_dir}
- 成功处理: {len(self.processed_files)} 个文件
- 处理失败: {len(self.failed_files)} 个文件

## 成功处理的文件
"""
        
        if self.processed_files:
            for item in self.processed_files:
                summary_content += f"""
### {item['file']}
- 输出目录: `{item['output_dir']}`
- 处理时间: {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
- 状态: ✓ 成功
"""
        else:
            summary_content += "\n无成功处理的文件\n"
        
        if self.failed_files:
            summary_content += "\n## 处理失败的文件\n"
            for item in self.failed_files:
                summary_content += f"""
### {item['file']}
- 错误信息: {item['error']}
- 时间: {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
- 状态: ✗ 失败
"""
        
        summary_content += f"""

## 输出目录结构
```
{self.output_base_dir}/
├── batch_summary.md          # 本摘要报告
"""
        
        for item in self.processed_files:
            file_stem = Path(item['file']).stem
            summary_content += f"""
├── {file_stem}/              # {item['file']} 的分析结果
│   ├── solar_radiation/      # 太阳辐射分析
│   ├── temperature_analysis/ # 温度分析
│   ├── weather_conditions/   # 天气条件
│   ├── correlation_analysis/ # 相关性分析
│   ├── statistical_summary/  # 统计摘要
│   └── README.md            # 详细报告
"""
        
        summary_content += """
```

## 使用说明
1. 每个数据文件的分析结果保存在独立的子目录中
2. 每个子目录包含完整的可视化分析图表
3. 查看各子目录中的 `README.md` 获取详细分析说明
4. 所有图表均为高分辨率PNG格式，适合报告使用

## 分析内容
每个文件的分析包括：
- **太阳辐射分析**: 全球水平辐射、直射辐射、散射辐射的时间序列和分布
- **温度分析**: 多点温度对比、日变化模式、与辐射的相关性
- **天气条件**: 风速风向、云量、湿度分布
- **相关性分析**: 关键变量间的相关矩阵热力图
- **统计摘要**: 描述性统计和数据分布箱线图
"""
        
        # 保存摘要报告
        summary_file = self.output_base_dir / 'batch_summary.md'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"批量处理摘要已保存到: {summary_file}")
        
        # 创建索引HTML文件（可选）
        self.create_html_index()
    
    def create_html_index(self):
        """创建HTML索引页面"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PV数据可视化分析结果</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .file-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }}
        .file-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 20px; background: #fafafa; transition: transform 0.2s; }}
        .file-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .file-name {{ font-weight: bold; color: #2980b9; font-size: 1.1em; margin-bottom: 10px; }}
        .file-info {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 15px; }}
        .links {{ display: flex; gap: 10px; flex-wrap: wrap; }}
        .link {{ background: #3498db; color: white; padding: 5px 10px; border-radius: 4px; text-decoration: none; font-size: 0.8em; transition: background 0.2s; }}
        .link:hover {{ background: #2980b9; }}
        .stats {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .success {{ color: #27ae60; font-weight: bold; }}
        .error {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌞 PV数据可视化分析结果</h1>
        
        <div class="stats">
            <h2>📊 处理统计</h2>
            <p><strong>处理时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>输入目录:</strong> {self.input_dir}</p>
            <p><strong>输出目录:</strong> {self.output_base_dir}</p>
            <p class="success">✓ 成功处理: {len(self.processed_files)} 个文件</p>
            <p class="error">✗ 处理失败: {len(self.failed_files)} 个文件</p>
        </div>
        
        <h2>📁 分析结果</h2>
        <div class="file-grid">
"""
        
        for item in self.processed_files:
            file_stem = Path(item['file']).stem
            html_content += f"""
            <div class="file-card">
                <div class="file-name">{item['file']}</div>
                <div class="file-info">处理时间: {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</div>
                <div class="links">
                    <a href="{file_stem}/README.md" class="link">📄 详细报告</a>
                    <a href="{file_stem}/solar_radiation/" class="link">☀️ 太阳辐射</a>
                    <a href="{file_stem}/temperature_analysis/" class="link">🌡️ 温度分析</a>
                    <a href="{file_stem}/weather_conditions/" class="link">🌤️ 天气条件</a>
                    <a href="{file_stem}/correlation_analysis/" class="link">📈 相关性分析</a>
                    <a href="{file_stem}/statistical_summary/" class="link">📊 统计摘要</a>
                </div>
            </div>
"""
        
        if self.failed_files:
            html_content += """
        </div>
        
        <h2>❌ 处理失败的文件</h2>
        <div class="file-grid">
"""
            for item in self.failed_files:
                html_content += f"""
            <div class="file-card" style="border-color: #e74c3c;">
                <div class="file-name" style="color: #e74c3c;">{item['file']}</div>
                <div class="file-info">错误时间: {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</div>
                <div style="color: #e74c3c; font-size: 0.9em; margin-top: 10px;">
                    <strong>错误信息:</strong> {item['error']}
                </div>
            </div>
"""
        
        html_content += f"""
        </div>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; text-align: center;">
            <p>PV数据可视化分析系统 | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
        
        # 保存HTML索引
        html_file = self.output_base_dir / 'index.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML索引页面已创建: {html_file}")
    
    def run_batch_processing(self):
        """执行批量处理"""
        print(f"\n开始批量处理PV数据可视化...")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 查找数据文件
        data_files = self.find_data_files()
        
        if not data_files:
            print("未找到任何数据文件！")
            return
        
        # 处理每个文件
        for i, file_path in enumerate(data_files, 1):
            print(f"\n进度: {i}/{len(data_files)}")
            self.process_single_file(file_path)
        
        # 生成批量摘要
        self.generate_batch_summary()
        
        # 打印最终统计
        print(f"\n{'='*60}")
        print("批量处理完成！")
        print(f"{'='*60}")
        print(f"✓ 成功处理: {len(self.processed_files)} 个文件")
        print(f"✗ 处理失败: {len(self.failed_files)} 个文件")
        print(f"📁 结果目录: {self.output_base_dir}")
        print(f"📄 摘要报告: {self.output_base_dir / 'batch_summary.md'}")
        print(f"🌐 HTML索引: {self.output_base_dir / 'index.html'}")
        print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description='批量PV数据可视化分析')
    parser.add_argument('input_dir', help='输入数据目录路径')
    parser.add_argument('-o', '--output', default='batch_visualization_results', help='输出基础目录')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        sys.exit(1)
    
    # 创建批量处理器
    batch_processor = BatchPVVisualizer(args.input_dir, args.output)
    
    # 执行批量处理
    batch_processor.run_batch_processing()

if __name__ == "__main__":
    main()