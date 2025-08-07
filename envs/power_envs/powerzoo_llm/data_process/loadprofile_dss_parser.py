# -*- coding: utf-8 -*-
"""
DSS文件解析器模块
负责DSS文件的解析和处理
"""

from typing import Tuple, Optional, List


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