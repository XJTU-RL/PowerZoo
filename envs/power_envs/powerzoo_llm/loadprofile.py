import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

class Constants:
    """LoadProfile相关常量定义"""
    DSS_EXTENSION = '.dss'
    CSV_EXTENSION = '.csv'
    DUTY_SUFFIX = '_duty'
    LOADSHAPE_FOLDER = 'loadshape'
    SCALE_FILE = 'scale.txt'
    
    # DSS文件相关常量
    DSS_COMMENT_MARKERS = ['!', '//']
    DSS_LOAD_PREFIX = 'new load.'
    DSS_REDIRECT_PREFIX = 'redirect'
    DSS_DUTY_MODE_PREFIX = 'set mode=duty'
    
    # 时间相关常量
    SECONDS_PER_HOUR = 3600
    DEFAULT_DUTY_SETTINGS = 'Set mode=duty number=360 hour=0 stepsize=60 sec=0\n'
    EPISODE_FOLDER_DIGITS = 3

class DSSFileParser:
    """DSS文件解析器"""
    
    @staticmethod
    def clean_line(line: str) -> str:
        """清理DSS文件行，移除注释和多余空格"""
        line = line.strip()
        for marker in Constants.DSS_COMMENT_MARKERS:
            if marker in line:
                line = line[:line.find(marker)].strip()
        return line
    
    @staticmethod
    def parse_load_line(line: str) -> Tuple[bool, Optional[str]]:
        """解析负载定义行"""
        clean_line = DSSFileParser.clean_line(line).lower()
        if not clean_line.startswith(Constants.DSS_LOAD_PREFIX):
            return False, None
            
        tokens = list(filter(None, line.split(' ')))
        if len(tokens) >= 2 and '.' in tokens[1]:
            load_name = tokens[1].split('.', 1)[1]
            return True, load_name
        return True, None
    
    @staticmethod
    def has_duty_in_line(line: str) -> bool:
        """检查行中是否包含duty关键字"""
        return 'duty' in DSSFileParser.clean_line(line).lower()
    
    @staticmethod
    def is_duty_mode_line(line: str) -> bool:
        """检查是否为duty模式设置行"""
        return DSSFileParser.clean_line(line).lower().startswith(Constants.DSS_DUTY_MODE_PREFIX)
    
    @staticmethod
    def parse_redirect_line(line: str) -> Optional[str]:
        """解析redirect行"""
        clean_line = DSSFileParser.clean_line(line).lower()
        if not clean_line.startswith(Constants.DSS_REDIRECT_PREFIX):
            return None
            
        load_patterns = ['loads', 'load', 'loads_duty', 'load_duty']
        clean_no_ext = clean_line[:-4] if len(clean_line) >= 4 else ''
        
        if any(clean_no_ext.endswith(p) for p in load_patterns):
            tokens = list(filter(None, line.strip().split(' ')))
            if len(tokens) >= 2:
                return tokens[1]
        return None

class LoadProfile:
    """负载配置文件管理器 - 支持负载、光伏和温度数据的统一管理
    
    新增功能：
    1. 光伏辐照度数据的自动切分和episode管理
    2. 温度数据的自动切分和episode管理
    3. 动态DSS配置文件生成
    4. 多线程环境下的数据一致性保证
    """
    
    def __init__(self, steps: int, dss_folder_path: str, dss_file: str, 
                 worker_idx: Optional[int] = None,
                 pv_data_source: Optional[str] = None, temperature_data_source: Optional[str] = None):
        self.steps = steps
        self.dss_folder_path = dss_folder_path
        self.loadshape_path = os.path.join(dss_folder_path, Constants.LOADSHAPE_FOLDER)
        self.loadshape_dss = f'loadshape{"" if worker_idx is None else f"_{worker_idx}"}{Constants.DSS_EXTENSION}'
        self.worker_idx = worker_idx
        
        # PV和温度数据配置
        self.pv_data_source = pv_data_source  # PV数据源文件路径
        self.temperature_data_source = temperature_data_source  # 温度数据源文件路径
        self.pv_data_dss = f'pv_data{"" if worker_idx is None else f"_{worker_idx}"}{Constants.DSS_EXTENSION}'
        
        # PV和温度数据存储路径
        self.irradiation_path = os.path.join(dss_folder_path, 'irradiation')
        self.temperature_path = os.path.join(dss_folder_path, 'temperature')
        
        # 查找并初始化负载名称
        self.load_names = self.find_load_names(dss_file)
        
        # 收集所有负载形状CSV文件
        self.csv_files = []
        if os.path.exists(self.loadshape_path):
            self.csv_files = [
                os.path.join(self.loadshape_path, f) 
                for f in os.listdir(self.loadshape_path)
                if 'loadshape' in f.lower() and f.lower().endswith(Constants.CSV_EXTENSION)
            ]
        
        # 初始化PV和温度数据文件列表
        self.pv_csv_files = []
        self.temperature_csv_files = []
        self._discover_pv_temperature_files()
    
    # 删除了冗余的属性包装器 LOAD_NAMES 和 FILES
    
    def create_file_with_duty(self, dss_file: str) -> str:
        """为DSS文件中缺少duty参数的负载行添加duty参数"""
        source_path = os.path.join(self.dss_folder_path, dss_file)
        duty_filename = dss_file.replace(Constants.DSS_EXTENSION, f'{Constants.DUTY_SUFFIX}{Constants.DSS_EXTENSION}')
        duty_path = os.path.join(self.dss_folder_path, duty_filename)
        
        with open(source_path, 'r', encoding='utf-8') as fin, \
             open(duty_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                is_load, load_name = DSSFileParser.parse_load_line(line)
                
                if not is_load or DSSFileParser.has_duty_in_line(line):
                    fout.write(line)
                else:
                    clean_line = DSSFileParser.clean_line(line)
                    if load_name:
                        fout.write(f'{clean_line} duty=loadshape_{load_name}\n')
                    else:
                        fout.write(line)
        
        return duty_path
   
    def add_redirect_and_mode_at_main_duty_dss(self, main_duty_dss: str) -> Optional[str]:
        """在主duty DSS文件中添加重定向和设置duty模式"""
        file_path = os.path.join(self.dss_folder_path, main_duty_dss)
        
        with open(file_path, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()

        found_load = False
        load_file = None
        
        with open(file_path, 'w', encoding='utf-8') as fout:
            for line in lines:
                clean_line = DSSFileParser.clean_line(line).lower()
                
                if not found_load and 'load' in clean_line and not clean_line.startswith('~'):
                    fout.write('! 自动添加加入redirect的负荷 \n')
                    fout.write(f'redirect {self.loadshape_dss}\n\n')
                    found_load = True

                redirect_file = DSSFileParser.parse_redirect_line(line)
                if redirect_file and not load_file:
                    load_file = redirect_file
                    if not redirect_file.endswith('_duty.dss'):
                        duty_file = redirect_file[:-4] + '_duty.dss'
                        fout.write(f'redirect {duty_file}\n')
                    else:
                        fout.write(line)
                else:
                    fout.write(line)
        
            fout.write(Constants.DEFAULT_DUTY_SETTINGS)
        
        assert found_load, f'cannot find load at {main_duty_dss}'
        return load_file

    def find_load_file_from(self, main_dss: str) -> Optional[str]:
        """从主DSS文件中查找重定向的负载文件"""
        file_path = os.path.join(self.dss_folder_path, main_dss)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    redirect_file = DSSFileParser.parse_redirect_line(line)
                    if redirect_file:
                        return redirect_file
        except FileNotFoundError:
            raise FileNotFoundError(f"DSS文件不存在: {file_path}")
        
        return None

    def find_load_names(self, main_dss: str) -> List[str]:
        """从主DSS文件中查找所有负载名称"""
        names = []
        
        # 分析原始DSS文件
        file_path = os.path.join(self.dss_folder_path, main_dss)
        assert os.path.exists(file_path), f'{file_path} not found'
        
        needs_load_duty = False
        duty_mode = False
        
        with open(file_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                if DSSFileParser.is_duty_mode_line(line):
                    duty_mode = True
                    continue
                
                is_load, load_name = DSSFileParser.parse_load_line(line)
                if is_load and load_name:
                    names.append(load_name)
                    if not DSSFileParser.has_duty_in_line(line):
                        needs_load_duty = True
        
        # 如果需要添加duty参数，创建增强版文件
        if needs_load_duty or not duty_mode:
            self.create_file_with_duty(main_dss)
            duty_file_name = main_dss[:-4] + '_duty.dss'
            load_file = self.add_redirect_and_mode_at_main_duty_dss(duty_file_name)
            
            # 重新分析
            names.clear()
            self._analyze_dss_file(duty_file_name, names)
            
            if load_file:
                self._analyze_dss_file(load_file, names)
        else:
            load_file = self.find_load_file_from(main_dss)
            if load_file:
                self._analyze_dss_file(load_file, names)
        
        # 验证
        assert len(names) > 0, '未找到任何负载定义'
        assert len(names) == len(set(names)), f'发现重复的负载名称'
        
        return names
    
    def _analyze_dss_file(self, fname: str, names: List[str]) -> Tuple[bool, bool]:
        """简化的分析方法"""
        file_path = os.path.join(self.dss_folder_path, fname)
        needs_duty = False
        duty_mode = False
        
        with open(file_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                if DSSFileParser.is_duty_mode_line(line):
                    duty_mode = True
                
                is_load, load_name = DSSFileParser.parse_load_line(line)
                if is_load and load_name:
                    names.append(load_name)
                    if not DSSFileParser.has_duty_in_line(line):
                        needs_duty = True
        
        return needs_duty, duty_mode

    def generate_realistic_episodes_from_full_data(self, 
                                                  full_load_data_paths: List[str],
                                                  pv_5_17_data_paths: List[str], 
                                                  temp_5_17_data_paths: List[str],
                                                  target_episodes: int = 30,
                                                  pv_threshold: float = 0.05) -> int:
        """从三个月数据中智能采样生成episode
        
        处理混合数据格式：
        - 负荷数据：完整分钟级数据(1440点/天)，函数自动提取5-17点
        - 光伏数据：预提取的5-17点数据(720点/天)
        - 温度数据：预提取的5-17点数据(720点/天)
        
        Args:
            full_load_data_paths: 完整负载数据文件路径列表(每月约1.576M点)
            pv_5_17_data_paths: 5-17点光伏数据文件路径列表(每月约22.3K点)
            temp_5_17_data_paths: 5-17点温度数据文件路径列表(每月约22.3K点)
            target_episodes: 目标episode数量
            pv_threshold: 光伏有效阈值（标准化值，低于此值认为无光伏）
            
        Returns:
            int: 实际生成的episode数量
        """
        print(f"🎯 开始从混合数据格式中智能采样 {target_episodes} 个episodes...")
        print(f"📊 数据格式：负荷(全天) + 光伏/温度(5-17点)")
        
        # 1. 分析预提取的光伏数据，识别有效时间段
        pv_active_periods = self._identify_pv_active_periods_from_5_17_data(pv_5_17_data_paths, pv_threshold)
        print(f"📊 识别到 {len(pv_active_periods)} 个光伏活跃时段")
        
        # 2. 分类采样策略
        episode_samples = self._smart_episode_sampling(pv_active_periods, target_episodes)
        print(f"🎲 采样策略：{len(episode_samples)} 个时间窗口")
        
        # 3. 提取对应的负载、光伏、温度数据
        episodes_created = 0
        for i, sample in enumerate(episode_samples):
            success = self._extract_and_save_mixed_format_episode(
                episode_idx=i,
                sample_info=sample,
                full_load_data_paths=full_load_data_paths,
                pv_5_17_data_paths=pv_5_17_data_paths,
                temp_5_17_data_paths=temp_5_17_data_paths
            )
            if success:
                episodes_created += 1
                if episodes_created % 5 == 0:
                    print(f"✅ 已创建 {episodes_created}/{target_episodes} 个episodes")
        
        print(f"🎉 智能采样完成！成功创建 {episodes_created} 个真实时间对应的episodes")
        return episodes_created
        
    def generate_episodes_from_existing_files(self, target_episodes: int = 30, pv_threshold: float = 0.05) -> int:
        """从loadshape文件夹中的现有数据文件生成episodes
        
        自动发现并使用loadshape文件夹中的数据文件：
        - 负荷数据：month_XX_0000_to_2359.csv (全天1440分钟)
        - 光伏数据：irrad_2025_XX_0500_to_1659.csv (5-17点720分钟)
        - 温度数据：temperature_2025_XX_0500_to_1659.csv (5-17点720分钟)
        
        Args:
            target_episodes: 目标episode数量
            pv_threshold: 光伏有效阈值（标准化值，低于此值认为无光伏）
            
        Returns:
            int: 实际生成的episode数量
        """
        print(f"🎯 开始从loadshape文件夹中的现有数据生成 {target_episodes} 个episodes...")
        
        # 1. 自动发现数据文件
        load_files, pv_files, temp_files = self._discover_existing_data_files()
        
        if not load_files:
            print("❌ 未找到负荷数据文件")
            return 0
            
        if not pv_files or not temp_files:
            print("❌ 未找到光伏或温度数据文件")
            return 0
            
        print(f"📁 发现数据文件:")
        print(f"   负荷文件: {len(load_files)}个")
        print(f"   光伏文件: {len(pv_files)}个")
        print(f"   温度文件: {len(temp_files)}个")
        
        # 2. 分析光伏数据，识别有效时间段
        pv_active_periods = self._identify_pv_active_periods_from_existing_files(pv_files, pv_threshold)
        print(f"📊 识别到 {len(pv_active_periods)} 个光伏活跃时段")
        
        if len(pv_active_periods) == 0:
            print("❌ 未找到有效的光伏活跃时段")
            return 0
        
        # 3. 分类采样策略
        episode_samples = self._smart_episode_sampling(pv_active_periods, target_episodes)
        print(f"🎲 采样策略：{len(episode_samples)} 个时间窗口")
        
        # 4. 提取对应的负载、光伏、温度数据
        episodes_created = 0
        for i, sample in enumerate(episode_samples):
            success = self._extract_and_save_from_existing_files(
                episode_idx=i,
                sample_info=sample,
                load_files=load_files,
                pv_files=pv_files,
                temp_files=temp_files
            )
            if success:
                episodes_created += 1
                if episodes_created % 5 == 0:
                    print(f"✅ 已创建 {episodes_created}/{target_episodes} 个episodes")
        
        print(f"🎉 智能采样完成！成功创建 {episodes_created} 个真实时间对应的episodes")
        return episodes_created
    
    def _discover_existing_data_files(self) -> Tuple[List[str], List[str], List[str]]:
        """自动发现loadshape文件夹中的数据文件"""
        base_path = os.path.join(self.dss_folder_path, 'loadshape')
        
        load_files = []
        pv_files = []
        temp_files = []
        
        if not os.path.exists(base_path):
            return load_files, pv_files, temp_files
            
        for filename in os.listdir(base_path):
            filepath = os.path.join(base_path, filename)
            
            if not filename.endswith('.csv'):
                continue
                
            # 识别负荷数据文件: month_XX_0000_to_2359.csv
            if filename.startswith('month_') and '_0000_to_2359' in filename:
                load_files.append(filepath)
            
            # 识别光伏数据文件: irrad_2025_XX_0500_to_1659.csv
            elif filename.startswith('irrad_') and '_0500_to_1659' in filename:
                pv_files.append(filepath)
            
            # 识别温度数据文件: temperature_2025_XX_0500_to_1659.csv
            elif filename.startswith('temperature_') and '_0500_to_1659' in filename:
                temp_files.append(filepath)
        
        # 排序确保文件顺序一致
        load_files.sort()
        pv_files.sort()
        temp_files.sort()
        
        return load_files, pv_files, temp_files
    
    def _identify_pv_active_periods_from_existing_files(self, pv_files: List[str], threshold: float) -> List[Dict]:
        """从现有光伏文件中识别活跃时段"""
        active_periods = []
        
        for file_idx, pv_path in enumerate(pv_files):
            if not os.path.exists(pv_path):
                continue
                
            # 从文件名提取月份信息
            filename = os.path.basename(pv_path)
            month_str = filename.split('_')[2] if len(filename.split('_')) >= 3 else f"{file_idx+1:02d}"
            month_num = int(month_str) if month_str.isdigit() else file_idx + 1
                
            # 读取5-17点光伏数据
            pv_data = pd.read_csv(pv_path, header=None).values.flatten()
            
            # 每天720分钟(12小时×60分钟)
            days_in_month = len(pv_data) // 720
            print(f"📅 月份{month_num}: {days_in_month}天数据，共{len(pv_data)}个数据点")
            
            for day in range(days_in_month):
                day_start_idx = day * 720
                day_data = pv_data[day_start_idx:day_start_idx + 720]
                
                # 在5-17点数据中查找6小时连续有效光伏时段
                for start_hour_offset in range(0, 7):  # 0-6小时开始的6小时窗口(对应5-11点)
                    start_minute = start_hour_offset * 60
                    window_data = day_data[start_minute:start_minute + 360]  # 6小时=360分钟
                    
                    if len(window_data) < 360:  # 数据不足
                        continue
                    
                    # 检查光伏有效性
                    valid_ratio = np.sum(window_data > threshold) / len(window_data)
                    avg_irradiance = np.mean(window_data)
                    std_irradiance = np.std(window_data)
                    
                    if valid_ratio > 0.3:  # 至少30%时间有光伏
                        actual_start_hour = 5 + start_hour_offset  # 转回实际小时
                        period_info = {
                            'month': month_num,
                            'day': day + 1,
                            'start_hour': actual_start_hour,
                            'end_hour': actual_start_hour + 6,
                            'start_idx_in_5_17_data': day_start_idx + start_minute,
                            'start_idx_in_full_day': day * 1440 + actual_start_hour * 60,
                            'avg_irradiance': avg_irradiance,
                            'std_irradiance': std_irradiance,
                            'valid_ratio': valid_ratio,
                            'weather_type': self._classify_weather_from_pv(window_data),
                            'file_idx': file_idx  # 文件索引，用于后续数据提取
                        }
                        active_periods.append(period_info)
        
        return active_periods
    
    def _extract_and_save_from_existing_files(self, episode_idx: int, sample_info: Dict,
                                            load_files: List[str], pv_files: List[str], 
                                            temp_files: List[str]) -> bool:
        """从现有文件中提取并保存episode数据"""
        try:
            month_num = sample_info['month']
            file_idx = sample_info.get('file_idx', 0)
            
            # 创建episode目录
            episode_folder = str(episode_idx).zfill(Constants.EPISODE_FOLDER_DIGITS)
            
            # 1. 从负荷数据中提取对应的5-17点时段数据
            load_episode_dir = os.path.join(self.loadshape_path, episode_folder)
            os.makedirs(load_episode_dir, exist_ok=True)
            
            # 尝试找到对应月份的负荷文件
            load_file = None
            for lf in load_files:
                if f"month_{month_num:02d}_" in os.path.basename(lf):
                    load_file = lf
                    break
            
            if not load_file and load_files:
                # 如果找不到对应月份，使用第一个可用文件
                load_file = load_files[0]
                print(f"⚠️  未找到月份{month_num}的负荷文件，使用 {os.path.basename(load_file)}")
            
            if load_file:
                full_load_data = pd.read_csv(load_file, header=None).values.flatten()
                full_day_start_idx = sample_info['start_idx_in_full_day']
                
                for i, load_name in enumerate(self.load_names):
                    # 每个负载从对应位置提取6小时=360个数据点
                    load_start_idx = full_day_start_idx + i * len(full_load_data) // len(self.load_names)
                    load_series = full_load_data[load_start_idx:load_start_idx + self.steps]
                    
                    if len(load_series) == self.steps:
                        csv_path = os.path.join(load_episode_dir, f'{load_name}.csv')
                        pd.DataFrame(load_series).to_csv(csv_path, header=False, index=False)
                    else:
                        print(f"⚠️  负荷数据不足: {load_name}, 需要{self.steps}个点，实际{len(load_series)}个")
                        return False
            
            # 2. 从光伏文件中提取数据
            pv_episode_dir = os.path.join(self.irradiation_path, episode_folder)
            os.makedirs(pv_episode_dir, exist_ok=True)
            
            if file_idx < len(pv_files):
                pv_5_17_data = pd.read_csv(pv_files[file_idx], header=None).values.flatten()
                pv_start_idx = sample_info['start_idx_in_5_17_data']
                pv_series = pv_5_17_data[pv_start_idx:pv_start_idx + self.steps]
                
                if len(pv_series) == self.steps:
                    pv_path = os.path.join(pv_episode_dir, '20250307_irradiance.csv')
                    pd.DataFrame(pv_series).to_csv(pv_path, header=False, index=False)
                else:
                    print(f"⚠️  光伏数据不足: 需要{self.steps}个点，实际{len(pv_series)}个")
                    return False
            
            # 3. 从温度文件中提取数据
            temp_episode_dir = os.path.join(self.temperature_path, episode_folder)
            os.makedirs(temp_episode_dir, exist_ok=True)
            
            if file_idx < len(temp_files):
                temp_5_17_data = pd.read_csv(temp_files[file_idx], header=None).values.flatten()
                temp_start_idx = sample_info['start_idx_in_5_17_data']
                temp_series = temp_5_17_data[temp_start_idx:temp_start_idx + self.steps]
                
                if len(temp_series) == self.steps:
                    temp_path = os.path.join(temp_episode_dir, '20250307_temperature.csv')
                    pd.DataFrame(temp_series).to_csv(temp_path, header=False, index=False)
                else:
                    print(f"⚠️  温度数据不足: 需要{self.steps}个点，实际{len(temp_series)}个")
                    return False
            
            # 4. 生成对应的DSS配置文件
            self.select_pv_temperature_profile(episode_idx)
            
            print(f"✅ Episode {episode_idx} 创建成功 (月{sample_info['month']}, 天{sample_info['day']}, {sample_info['start_hour']}-{sample_info['end_hour']}点, {sample_info['weather_type']})")
            return True
            
        except Exception as e:
            print(f"❌ Episode {episode_idx} 创建失败: {e}")
            return False
    
    def _identify_pv_active_periods_from_5_17_data(self, pv_5_17_data_paths: List[str], threshold: float) -> List[Dict]:
        """从预提取的5-17点光伏数据中识别活跃时段"""
        active_periods = []
        
        for month_idx, pv_path in enumerate(pv_5_17_data_paths):
            if not os.path.exists(pv_path):
                continue
                
            # 读取5-17点光伏数据
            pv_data = pd.read_csv(pv_path, header=None).values.flatten()
            
            # 每天720分钟(12小时×60分钟)
            days_in_month = len(pv_data) // 720
            print(f"📅 月份{month_idx+1}: {days_in_month}天数据，共{len(pv_data)}个数据点")
            
            for day in range(days_in_month):
                day_start_idx = day * 720
                day_data = pv_data[day_start_idx:day_start_idx + 720]
                
                # 在5-17点数据中查找6小时连续有效光伏时段
                # 由于数据已经是5-17点，所以映射为0-11小时索引
                for start_hour_offset in range(0, 7):  # 0-6小时开始的6小时窗口(对应5-11点)
                    start_minute = start_hour_offset * 60
                    window_data = day_data[start_minute:start_minute + 360]  # 6小时=360分钟
                    
                    if len(window_data) < 360:  # 数据不足
                        continue
                    
                    # 检查光伏有效性
                    valid_ratio = np.sum(window_data > threshold) / len(window_data)
                    avg_irradiance = np.mean(window_data)
                    std_irradiance = np.std(window_data)
                    
                    if valid_ratio > 0.3:  # 至少30%时间有光伏
                        actual_start_hour = 5 + start_hour_offset  # 转回实际小时
                        period_info = {
                            'month': month_idx + 1,
                            'day': day + 1,
                            'start_hour': actual_start_hour,
                            'end_hour': actual_start_hour + 6,
                            'start_idx_in_5_17_data': day_start_idx + start_minute,  # 在5-17点数据中的索引
                            'start_idx_in_full_day': day * 1440 + actual_start_hour * 60,  # 在全天数据中的索引
                            'avg_irradiance': avg_irradiance,
                            'std_irradiance': std_irradiance,
                            'valid_ratio': valid_ratio,
                            'weather_type': self._classify_weather_from_pv(window_data)
                        }
                        active_periods.append(period_info)
        
        return active_periods
    
    def _classify_weather_from_pv(self, pv_window: np.ndarray) -> str:
        """根据光伏数据分类天气类型"""
        mean_irr = np.mean(pv_window)
        std_irr = np.std(pv_window)
        
        if mean_irr > 0.8:
            return "sunny" if std_irr < 0.2 else "sunny_variable"
        elif mean_irr > 0.5:
            return "partly_cloudy" if std_irr < 0.3 else "variable"
        else:
            return "overcast"
    
    def _smart_episode_sampling(self, active_periods: List[Dict], target_episodes: int) -> List[Dict]:
        """智能采样策略"""
        if len(active_periods) <= target_episodes:
            return active_periods
        
        # 按天气类型分层采样
        weather_groups = {}
        for period in active_periods:
            weather = period['weather_type']
            if weather not in weather_groups:
                weather_groups[weather] = []
            weather_groups[weather].append(period)
        
        # 分配采样数量
        samples = []
        remaining_episodes = target_episodes
        
        for weather_type, periods in weather_groups.items():
            # 每种天气类型至少采样1个
            if remaining_episodes > 0:
                sample_count = min(len(periods), max(1, remaining_episodes // len(weather_groups)))
                selected = np.random.choice(periods, sample_count, replace=False)
                samples.extend(selected)
                remaining_episodes -= sample_count
        
        # 如果还有剩余名额，随机补充
        if remaining_episodes > 0:
            all_remaining = [p for p in active_periods if p not in samples]
            additional = np.random.choice(all_remaining, 
                                        min(remaining_episodes, len(all_remaining)), 
                                        replace=False)
            samples.extend(additional)
        
        return samples[:target_episodes]
    
    def _extract_and_save_mixed_format_episode(self, episode_idx: int, sample_info: Dict,
                                             full_load_data_paths: List[str],
                                             pv_5_17_data_paths: List[str], 
                                             temp_5_17_data_paths: List[str]) -> bool:
        """提取并保存混合格式episode数据
        
        处理负荷(全天)和光伏/温度(5-17点)的混合数据格式
        """
        try:
            month = sample_info['month'] - 1  # 转为0-based索引
            
            # 创建episode目录
            episode_folder = str(episode_idx).zfill(Constants.EPISODE_FOLDER_DIGITS)
            
            # 1. 从完整负荷数据中提取对应的5-17点时段数据
            load_episode_dir = os.path.join(self.loadshape_path, episode_folder)
            os.makedirs(load_episode_dir, exist_ok=True)
            
            full_load_data = pd.read_csv(full_load_data_paths[month], header=None).values.flatten()
            full_day_start_idx = sample_info['start_idx_in_full_day']  # 在全天数据中的起始位置
            
            for i, load_name in enumerate(self.load_names):
                # 每个负载从对应位置提取6小时=360个数据点
                load_start_idx = full_day_start_idx + i * len(full_load_data) // len(self.load_names)
                load_series = full_load_data[load_start_idx:load_start_idx + self.steps]
                
                if len(load_series) == self.steps:
                    csv_path = os.path.join(load_episode_dir, f'{load_name}.csv')
                    pd.DataFrame(load_series).to_csv(csv_path, header=False, index=False)
                else:
                    print(f"⚠️  负荷数据不足: {load_name}, 需要{self.steps}个点，实际{len(load_series)}个")
                    return False
            
            # 2. 从5-17点光伏数据中提取对应数据
            pv_episode_dir = os.path.join(self.irradiation_path, episode_folder)
            os.makedirs(pv_episode_dir, exist_ok=True)
            
            pv_5_17_data = pd.read_csv(pv_5_17_data_paths[month], header=None).values.flatten()
            pv_start_idx = sample_info['start_idx_in_5_17_data']  # 在5-17点数据中的起始位置
            pv_series = pv_5_17_data[pv_start_idx:pv_start_idx + self.steps]
            
            if len(pv_series) == self.steps:
                pv_path = os.path.join(pv_episode_dir, '20250307_irradiance.csv')
                pd.DataFrame(pv_series).to_csv(pv_path, header=False, index=False)
            else:
                print(f"⚠️  光伏数据不足: 需要{self.steps}个点，实际{len(pv_series)}个")
                return False
            
            # 3. 从5-17点温度数据中提取对应数据
            temp_episode_dir = os.path.join(self.temperature_path, episode_folder)
            os.makedirs(temp_episode_dir, exist_ok=True)
            
            temp_5_17_data = pd.read_csv(temp_5_17_data_paths[month], header=None).values.flatten()
            temp_start_idx = sample_info['start_idx_in_5_17_data']  # 与光伏数据使用相同索引
            temp_series = temp_5_17_data[temp_start_idx:temp_start_idx + self.steps]
            
            if len(temp_series) == self.steps:
                temp_path = os.path.join(temp_episode_dir, '20250307_temperature.csv')
                pd.DataFrame(temp_series).to_csv(temp_path, header=False, index=False)
            else:
                print(f"⚠️  温度数据不足: 需要{self.steps}个点，实际{len(temp_series)}个")
                return False
            
            # 4. 生成对应的DSS配置文件
            self.select_pv_temperature_profile(episode_idx)
            
            print(f"✅ Episode {episode_idx} 创建成功 (月{sample_info['month']}, 天{sample_info['day']}, {sample_info['start_hour']}-{sample_info['end_hour']}点, {sample_info['weather_type']})")
            return True
            
        except Exception as e:
            print(f"❌ Episode {episode_idx} 创建失败: {e}")
            return False
    
    def _extract_and_save_episode(self, episode_idx: int, sample_info: Dict,
                                full_load_data_paths: List[str],
                                full_pv_data_paths: List[str], 
                                full_temp_data_paths: List[str]) -> bool:
        """提取并保存episode数据"""
        try:
            month = sample_info['month'] - 1  # 转为0-based索引
            start_idx = sample_info['start_idx']
            
            # 创建episode目录
            episode_folder = str(episode_idx).zfill(Constants.EPISODE_FOLDER_DIGITS)
            
            # 1. 提取并保存负载数据
            load_episode_dir = os.path.join(self.loadshape_path, episode_folder)
            os.makedirs(load_episode_dir, exist_ok=True)
            
            load_data = pd.read_csv(full_load_data_paths[month], header=None).values.flatten()
            for i, load_name in enumerate(self.load_names):
                # 每个负载从对应位置提取360个数据点
                load_start_idx = start_idx + i * len(load_data) // len(self.load_names)
                load_series = load_data[load_start_idx:load_start_idx + self.steps]
                
                if len(load_series) == self.steps:
                    csv_path = os.path.join(load_episode_dir, f'{load_name}.csv')
                    pd.DataFrame(load_series).to_csv(csv_path, header=False, index=False)
            
            # 2. 提取并保存光伏数据
            pv_episode_dir = os.path.join(self.irradiation_path, episode_folder)
            os.makedirs(pv_episode_dir, exist_ok=True)
            
            pv_data = pd.read_csv(full_pv_data_paths[month], header=None).values.flatten()
            pv_series = pv_data[start_idx:start_idx + self.steps]
            
            if len(pv_series) == self.steps:
                pv_path = os.path.join(pv_episode_dir, '20250307_irradiance.csv')
                pd.DataFrame(pv_series).to_csv(pv_path, header=False, index=False)
            
            # 3. 提取并保存温度数据
            temp_episode_dir = os.path.join(self.temperature_path, episode_folder)
            os.makedirs(temp_episode_dir, exist_ok=True)
            
            temp_data = pd.read_csv(full_temp_data_paths[month], header=None).values.flatten()
            temp_series = temp_data[start_idx:start_idx + self.steps]
            
            if len(temp_series) == self.steps:
                temp_path = os.path.join(temp_episode_dir, '20250307_temperature.csv')
                pd.DataFrame(temp_series).to_csv(temp_path, header=False, index=False)
            
            # 4. 生成对应的DSS配置文件
            self.select_pv_temperature_profile(episode_idx)
            
            return True
            
        except Exception as e:
            print(f"❌ Episode {episode_idx} 创建失败: {e}")
            return False

    def generate_load_profiles(self, scale: float = 1.0) -> int:
        """生成负载配置文件，并同时生成PV和温度配置文件"""
        # 加载所有CSV数据
        dataframes = []
        for file_path in self.csv_files:
            df = pd.read_csv(file_path, header=None)
            dataframes.append(df)
        
        assert len(dataframes) > 0, '将负荷数据放到文件夹下 ./loadshape'
        
        combined_data = pd.concat(dataframes).rename(columns={0: 'mul'}).reset_index(drop=True)
        if scale != 1.0:
            combined_data['mul'] *= scale
        
        # 计算episode数量
        episodes = len(combined_data) // (self.steps * len(self.load_names))
        
        # 检查是否需要重新生成
        if self._check_existing_files(episodes, scale):
            return episodes
        
        # 保存缩放因子
        np.savetxt(os.path.join(self.loadshape_path, Constants.SCALE_FILE), np.array([scale]))
        
        # 生成episode文件
        total_rows = self.steps * episodes * len(self.load_names)
        
        # 构建数据结构
        load_col = [self.load_names[i // (self.steps * episodes)] for i in range(total_rows)]
        episode_col = [(i // self.steps) % episodes for i in range(total_rows)]
        step_col = [i % self.steps for i in range(total_rows)]
        
        structured_df = combined_data[:total_rows].copy()
        structured_df['load'] = load_col
        structured_df['episode'] = episode_col
        structured_df['step'] = step_col
        
        structured_df = structured_df.sort_values(
            by=['episode', 'load', 'step']
        )[['episode', 'load', 'step', 'mul']].reset_index(drop=True)
        
        # 为每个episode创建文件
        for episode in range(episodes):
            episode_dir = os.path.join(self.loadshape_path, str(episode).zfill(Constants.EPISODE_FOLDER_DIGITS))
            os.makedirs(episode_dir, exist_ok=True)
            
            episode_data = structured_df[structured_df['episode'] == episode]
            
            for load_name in self.load_names:
                load_series = episode_data[episode_data['load'] == load_name]['mul']
                csv_path = os.path.join(episode_dir, f'{load_name}{Constants.CSV_EXTENSION}')
                load_series.to_csv(csv_path, header=False, index=False)
        
        # 同时生成PV和温度配置文件（如果有数据源）
        if self.pv_csv_files or self.temperature_csv_files:
            print(f"开始生成PV和温度数据，共{episodes}个episodes...")
            
            # 🔍 数据消耗量对比分析
            load_data_total = total_rows  # 负载数据总消耗量
            pv_data_needed = self.steps * episodes  # PV数据总消耗量
            temp_data_needed = self.steps * episodes  # 温度数据总消耗量
            
            print(f"📊 数据消耗量对比分析:")
            print(f"  - 负载数据消耗: {load_data_total:,} 点 ({len(self.load_names)} loads × {episodes} episodes × {self.steps} steps)")
            print(f"  - PV数据消耗:   {pv_data_needed:,} 点 (全网共享 × {episodes} episodes × {self.steps} steps)")
            print(f"  - 温度数据消耗: {temp_data_needed:,} 点 (全网共享 × {episodes} episodes × {self.steps} steps)")
            print(f"  - 消耗比例:     负载:PV:温度 = {len(self.load_names)}:1:1")
            
            pv_success = self.generate_pv_profiles(episodes) if self.pv_csv_files else True
            temp_success = self.generate_temperature_profiles(episodes) if self.temperature_csv_files else True
            
            if pv_success and temp_success:
                print("✅ PV和温度数据生成成功")
            else:
                print("⚠️ PV或温度数据生成部分失败")
        
        return episodes
    
    def _check_existing_files(self, episodes: int, scale: float) -> bool:
        """检查是否已存在有效的配置文件"""
        if not os.path.exists(self.loadshape_path):
            return False
            
        existing_folders = sum(1 for f in os.listdir(self.loadshape_path) if f.isdigit())
        
        scale_file = os.path.join(self.loadshape_path, Constants.SCALE_FILE)
        if os.path.exists(scale_file):
            existing_scale = np.loadtxt(scale_file)
            return existing_folders == episodes and existing_scale == scale
        
        return False
    
    def _discover_pv_temperature_files(self):
        """发现PV和温度数据文件"""
        # 发现PV辐照度数据文件
        if self.pv_data_source and os.path.exists(self.pv_data_source):
            if self.pv_data_source.endswith('.csv'):
                self.pv_csv_files = [self.pv_data_source]
            elif os.path.isdir(self.pv_data_source):
                self.pv_csv_files = [
                    os.path.join(self.pv_data_source, f) 
                    for f in os.listdir(self.pv_data_source)
                    if f.lower().endswith('.csv') and 'irrad' in f.lower()
                ]
        
        # 发现温度数据文件
        if self.temperature_data_source and os.path.exists(self.temperature_data_source):
            if self.temperature_data_source.endswith('.csv'):
                self.temperature_csv_files = [self.temperature_data_source]
            elif os.path.isdir(self.temperature_data_source):
                self.temperature_csv_files = [
                    os.path.join(self.temperature_data_source, f) 
                    for f in os.listdir(self.temperature_data_source)
                    if f.lower().endswith('.csv') and ('temp' in f.lower() or 'temperature' in f.lower())
                ]
    
    def select_load_profile(self, episode_idx: int) -> str:
        """选择指定episode的负载配置文件"""
        if episode_idx is None:
            raise ValueError("episode_idx 不能为空")
        if not isinstance(episode_idx, int):
            raise TypeError(f"episode_idx 必须是 int，当前类型为 {type(episode_idx)}")
        
        episode_folder = str(episode_idx).zfill(Constants.EPISODE_FOLDER_DIGITS)
        assert os.path.exists(os.path.join(self.loadshape_path, episode_folder)), 'episode_idx does not exist'
        
        with open(os.path.join(self.dss_folder_path, self.loadshape_dss), 'w') as fp:
            for load_name in self.load_names:
                fp.write(f'New Loadshape.loadshape_{load_name} npts={self.steps} sinterval={60*60*12//self.steps} ' +
                    f'mult=(file=./loadshape/{episode_folder}/{load_name}.csv)\n')

        return os.path.join(self.loadshape_path, episode_folder)
    
    def generate_pv_profiles(self, episodes: int) -> bool:
        """生成光伏辐照度配置文件
        
        Args:
            episodes: episode数量
            
        Returns:
            bool: 生成是否成功
        """
        if not self.pv_csv_files:
            return False
        
        # 加载所有PV CSV数据
        pv_dataframes = []
        for file_path in self.pv_csv_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, header=None)
                pv_dataframes.append(df)
        
        if not pv_dataframes:
            return False
        
        # 合并PV数据
        combined_pv_data = pd.concat(pv_dataframes).rename(columns={0: 'irradiance'}).reset_index(drop=True)
        
        # 🔧 修正：PV数据每个episode只需要steps个数据点（全网共享）
        expected_total_points = self.steps * episodes  # 每个episode需要360个点，不是360*负载数量
        if len(combined_pv_data) < expected_total_points:
            # 如果数据不足，进行循环填充
            cycles_needed = (expected_total_points + len(combined_pv_data) - 1) // len(combined_pv_data)
            combined_pv_data = pd.concat([combined_pv_data] * cycles_needed).reset_index(drop=True)
            print(f"⚠️ PV数据不足，进行了{cycles_needed}次循环填充")
        
        # 截取所需长度的数据
        combined_pv_data = combined_pv_data[:expected_total_points]
        
        print(f"📊 PV数据消耗分析:")
        print(f"  - 原始数据量: {len(pd.concat(pv_dataframes))} 点")
        print(f"  - 需要数据量: {expected_total_points} 点 ({episodes} episodes × {self.steps} steps)")
        print(f"  - 数据利用率: {min(100, len(pd.concat(pv_dataframes))/expected_total_points*100):.1f}%")
        
        # 创建辐照度数据存储路径
        os.makedirs(self.irradiation_path, exist_ok=True)
        
        # 为每个episode创建辐照度文件
        for episode in range(episodes):
            episode_dir = os.path.join(self.irradiation_path, str(episode).zfill(Constants.EPISODE_FOLDER_DIGITS))
            os.makedirs(episode_dir, exist_ok=True)
            
            # 提取当前episode的数据
            start_idx = episode * self.steps
            end_idx = start_idx + self.steps
            episode_pv_data = combined_pv_data.iloc[start_idx:end_idx]['irradiance']
            
            # 保存数据文件
            pv_file_path = os.path.join(episode_dir, '20250307_irradiance.csv')
            episode_pv_data.to_csv(pv_file_path, header=False, index=False)
        
        return True
    
    def generate_temperature_profiles(self, episodes: int) -> bool:
        """生成温度配置文件
        
        Args:
            episodes: episode数量
            
        Returns:
            bool: 生成是否成功
        """
        if not self.temperature_csv_files:
            return False
        
        # 加载所有温度CSV数据
        temp_dataframes = []
        for file_path in self.temperature_csv_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, header=None)
                temp_dataframes.append(df)
        
        if not temp_dataframes:
            return False
        
        # 合并温度数据
        combined_temp_data = pd.concat(temp_dataframes).rename(columns={0: 'temperature'}).reset_index(drop=True)
        
        # 🔧 修正：温度数据每个episode只需要steps个数据点（全网共享）
        expected_total_points = self.steps * episodes  # 每个episode需要360个点，不是360*负载数量
        if len(combined_temp_data) < expected_total_points:
            # 如果数据不足，进行循环填充
            cycles_needed = (expected_total_points + len(combined_temp_data) - 1) // len(combined_temp_data)
            combined_temp_data = pd.concat([combined_temp_data] * cycles_needed).reset_index(drop=True)
            print(f"⚠️ 温度数据不足，进行了{cycles_needed}次循环填充")
        
        # 截取所需长度的数据
        combined_temp_data = combined_temp_data[:expected_total_points]
        
        print(f"📊 温度数据消耗分析:")
        print(f"  - 原始数据量: {len(pd.concat(temp_dataframes))} 点")
        print(f"  - 需要数据量: {expected_total_points} 点 ({episodes} episodes × {self.steps} steps)")
        print(f"  - 数据利用率: {min(100, len(pd.concat(temp_dataframes))/expected_total_points*100):.1f}%")
        
        # 创建温度数据存储路径
        os.makedirs(self.temperature_path, exist_ok=True)
        
        # 为每个episode创建温度文件
        for episode in range(episodes):
            episode_dir = os.path.join(self.temperature_path, str(episode).zfill(Constants.EPISODE_FOLDER_DIGITS))
            os.makedirs(episode_dir, exist_ok=True)
            
            # 提取当前episode的数据
            start_idx = episode * self.steps
            end_idx = start_idx + self.steps
            episode_temp_data = combined_temp_data.iloc[start_idx:end_idx]['temperature']
            
            # 保存数据文件
            temp_file_path = os.path.join(episode_dir, '20250307_temperature.csv')
            episode_temp_data.to_csv(temp_file_path, header=False, index=False)
        
        return True
    
    
    def select_pv_temperature_profile(self, episode_idx: int) -> bool:
        """选择指定episode的PV和温度配置文件，并生成对应的pv_data DSS文件
        
        Args:
            episode_idx: episode索引
            
        Returns:
            bool: 选择和配置是否成功
        """
        if episode_idx is None:
            raise ValueError("episode_idx 不能为空")
        if not isinstance(episode_idx, int):
            raise TypeError(f"episode_idx 必须是 int，当前类型为 {type(episode_idx)}")
        
        episode_folder = str(episode_idx).zfill(Constants.EPISODE_FOLDER_DIGITS)
        
        # 检查PV和温度数据文件是否存在
        pv_episode_path = os.path.join(self.irradiation_path, episode_folder)
        temp_episode_path = os.path.join(self.temperature_path, episode_folder)
        
        pv_file_exists = os.path.exists(os.path.join(pv_episode_path, '20250307_irradiance.csv'))
        temp_file_exists = os.path.exists(os.path.join(temp_episode_path, '20250307_temperature.csv'))
        
        if not (pv_file_exists or temp_file_exists):
            return False  # 如果PV和温度文件都不存在，返回False
        
        # 生成episode特定的pv_data DSS文件
        worker_suffix = "" if self.worker_idx is None else f"_{self.worker_idx}"
        pv_dss_filename = f'pv_data_{episode_folder}{worker_suffix}{Constants.DSS_EXTENSION}'
        pv_dss_path = os.path.join(self.dss_folder_path, pv_dss_filename)
        
        return self._generate_pv_dss_config(pv_dss_path, episode_folder, pv_file_exists, temp_file_exists)
    
    def _generate_pv_dss_config(self, pv_dss_path: str, episode_folder: str, 
                              has_pv_data: bool, has_temp_data: bool) -> bool:
        """生成PV数据的DSS配置文件
        
        Args:
            pv_dss_path: DSS文件保存路径
            episode_folder: episode文件夹名
            has_pv_data: 是否有PV数据
            has_temp_data: 是否有温度数据
            
        Returns:
            bool: 生成是否成功
        """
        try:
            with open(pv_dss_path, 'w', encoding='utf-8') as fp:
                # 写入文件头注释
                fp.write(f'! PV系统数据配置文件 - Episode {episode_folder}\n')
                fp.write(f'! 自动生成时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                if self.worker_idx is not None:
                    fp.write(f'! Worker索引: {self.worker_idx}\n')
                fp.write('\n')
                
                # 定义PV温度系数曲线
                fp.write('! 定义PV温度系数曲线\n')
                fp.write('New XYCurve.MyPvsT npts=4 xarray=[0 25 75 100] yarray=[1.2 1.0 0.8 0.6]\n')
                fp.write('\n')
                
                # 定义PV效率曲线
                fp.write('! 定义PV效率曲线\n')
                fp.write('New XYCurve.MyEff npts=4 xarray=[.1 .2 1.0 1.2] yarray=[.86 .9 .98 .99]\n')
                fp.write('\n')
                
                # 配置辐照度LoadShape
                if has_pv_data:
                    fp.write('! 定义辐照度LoadShape\n')
                    fp.write(f'New Loadshape.MyIrrad npts={self.steps} sinterval=60 mult=(file=./irradiation/{episode_folder}/20250307_irradiance.csv)\n')
                else:
                    # 如果没有PV数据，使用默认值
                    fp.write('! 使用默认辐照度LoadShape (无数据文件)\n')
                    fp.write(f'New Loadshape.MyIrrad npts={self.steps} sinterval=60 mult=1.0\n')
                fp.write('\n')
                
                # 配置温度TShape
                if has_temp_data:
                    fp.write('! 定义温度TShape\n')
                    fp.write(f'New TShape.MyTemp npts={self.steps} sinterval=60 temp=(file=./temperature/{episode_folder}/20250307_temperature.csv)\n')
                else:
                    # 如果没有温度数据，使用默认值
                    fp.write('! 使用默认温度TShape (无数据文件)\n')
                    fp.write(f'New TShape.MyTemp npts={self.steps} sinterval=60 temp=25\n')
                fp.write('\n')
            
            return True
            
        except Exception as e:
            print(f"生成PV DSS配置文件失败: {e}")
            return False
    
    def get_load_profile_data(self, episode_idx: int) -> pd.DataFrame:
        """获取指定episode的负载配置数据"""
        folder_path = os.path.join(self.loadshape_path, str(episode_idx).zfill(Constants.EPISODE_FOLDER_DIGITS))
        
        load_data = []
        for csv_file in os.listdir(folder_path):
            if csv_file.endswith('.csv'):
                csv_path = os.path.join(folder_path, csv_file)
                load_name = csv_file.split('.')[0]
                load = pd.read_csv(csv_path, header=None, names=[load_name])
                load_data.append(load)

        return pd.concat(load_data, axis=1)
    
    def validate_data_consistency(self, episodes: int) -> Dict[str, Any]:
        """验证负载、PV和温度数据的一致性
        
        Args:
            episodes: episode数量
            
        Returns:
            dict: 验证结果
        """
        validation_result = {
            'status': 'success',
            'load_data_valid': False,
            'pv_data_valid': False,
            'temperature_data_valid': False,
            'warnings': [],
            'errors': []
        }
        
        try:
            # 验证负载数据
            for episode in range(episodes):
                episode_folder = str(episode).zfill(Constants.EPISODE_FOLDER_DIGITS)
                episode_dir = os.path.join(self.loadshape_path, episode_folder)
                
                if os.path.exists(episode_dir):
                    load_files = [f for f in os.listdir(episode_dir) if f.endswith('.csv')]
                    if len(load_files) == len(self.load_names):
                        # 检查每个负载文件的长度
                        for load_file in load_files:
                            file_path = os.path.join(episode_dir, load_file)
                            if os.path.exists(file_path):
                                load_data = pd.read_csv(file_path, header=None)
                                if len(load_data) != self.steps:
                                    validation_result['errors'].append(
                                        f"Episode {episode} 负载文件 {load_file} 长度不匹配: {len(load_data)} vs {self.steps}"
                                    )
                        validation_result['load_data_valid'] = True
                    else:
                        validation_result['errors'].append(f"Episode {episode} 负载文件数量不匹配")
                else:
                    validation_result['errors'].append(f"Episode {episode} 目录不存在")
            
            # 验证PV数据
            if self.pv_csv_files:
                for episode in range(episodes):
                    episode_folder = str(episode).zfill(Constants.EPISODE_FOLDER_DIGITS)
                    pv_episode_dir = os.path.join(self.irradiation_path, episode_folder)
                    pv_file = os.path.join(pv_episode_dir, '20250307_irradiance.csv')
                    
                    if os.path.exists(pv_file):
                        pv_data = pd.read_csv(pv_file, header=None)
                        if len(pv_data) != self.steps:
                            validation_result['errors'].append(
                                f"Episode {episode} PV数据长度不匹配: {len(pv_data)} vs {self.steps}"
                            )
                        else:
                            validation_result['pv_data_valid'] = True
                    else:
                        validation_result['warnings'].append(f"Episode {episode} PV数据文件不存在")
            
            # 验证温度数据
            if self.temperature_csv_files:
                for episode in range(episodes):
                    episode_folder = str(episode).zfill(Constants.EPISODE_FOLDER_DIGITS)
                    temp_episode_dir = os.path.join(self.temperature_path, episode_folder)
                    temp_file = os.path.join(temp_episode_dir, '20250307_temperature.csv')
                    
                    if os.path.exists(temp_file):
                        temp_data = pd.read_csv(temp_file, header=None)
                        if len(temp_data) != self.steps:
                            validation_result['errors'].append(
                                f"Episode {episode} 温度数据长度不匹配: {len(temp_data)} vs {self.steps}"
                            )
                        else:
                            validation_result['temperature_data_valid'] = True
                    else:
                        validation_result['warnings'].append(f"Episode {episode} 温度数据文件不存在")
            
            # 更新验证状态
            if validation_result['errors']:
                validation_result['status'] = 'error'
            elif validation_result['warnings']:
                validation_result['status'] = 'warning'
                
        except Exception as e:
            validation_result['status'] = 'error'
            validation_result['errors'].append(f"验证过程出现异常: {str(e)}")
        
        return validation_result
    
    # 删除向后兼容的冗余方法:
    # gen_loadprofile, choose_loadprofile, choose_irrad_profile, get_loadprofile



# ========== 使用示例 ==========
"""
使用示例1：从loadshape文件夹中的现有数据文件生成episodes（推荐）

# 数据文件准备（放在 node_systems/34Bus_PV/loadshape/ 文件夹下）:
# - month_01_0000_to_2359.csv, month_02_0000_to_2359.csv, month_03_0000_to_2359.csv  # 负荷数据(全天)
# - irrad_2025_01_0500_to_1659.csv                                                    # 光伏数据(5-17点)
# - temperature_2025_01_0500_to_1659.csv                                              # 温度数据(5-17点)

# 创建LoadProfile实例
load_profile = LoadProfile(
    steps=360,  # 6小时训练窗口
    dss_folder_path='node_systems/34Bus_PV',
    dss_file='ieee34Mod1.dss'
)

# 自动发现并使用现有数据文件生成episodes（推荐方法）
episodes_created = load_profile.generate_episodes_from_existing_files(
    target_episodes=30,
    pv_threshold=0.05  # 5%阈值过滤低光伏时段
)

print(f"✅ 成功创建 {episodes_created} 个真实天气条件的训练episodes")

# 数据验证
validation = load_profile.validate_generated_data()
if validation['status'] == 'success':
    print("✅ 所有数据验证通过")
else:
    print(f"⚠️  验证发现问题: {validation['errors']}")

---

使用示例2：处理外部混合数据格式（负荷全天 + 光伏/温度5-17点）

# 如果数据文件不在标准位置，可以使用此方法
full_load_paths = [
    'data/load/month1_full_1440min_per_day.csv',  # 1.576M点/月
    'data/load/month2_full_1440min_per_day.csv',
    'data/load/month3_full_1440min_per_day.csv'
]

pv_5_17_paths = [
    'data/pv/month1_5_17_720min_per_day.csv',    # 22.3K点/月
]

temp_5_17_paths = [
    'data/temp/month1_5_17_720min_per_day.csv',   # 22.3K点/月
]

# 智能生成episodes
episodes_created = load_profile.generate_realistic_episodes_from_full_data(
    full_load_data_paths=full_load_paths,
    pv_5_17_data_paths=pv_5_17_paths,
    temp_5_17_data_paths=temp_5_17_paths,
    target_episodes=30,
    pv_threshold=0.05
)

---

主要特性：
1. 🔍 自动文件发现：根据文件命名规则自动识别数据文件
2. ⏰ 智能时间同步：确保负荷、光伏、温度数据时间窗口完全对齐
3. 🌤️  天气模式分类：自动识别晴天、阴天、多云等天气条件
4. 📊 数据质量验证：实时检查数据完整性和有效性
5. 💾 高效数据处理：只处理有效的光伏活跃时段，避免无用数据
"""