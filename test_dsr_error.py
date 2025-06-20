#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append('.')
sys.path.append('./envs')

try:
    from dsr.core.dsr_core import DSRCoreEnv
    from dsr.core.config import DSRConfig
    print("正在初始化DSRCoreEnv...")
    config = DSRConfig(system_name='123Bus', dss_file='IEEE123Master.dss')
    dsr = DSRCoreEnv(config)
    print("DSRCore初始化成功")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()