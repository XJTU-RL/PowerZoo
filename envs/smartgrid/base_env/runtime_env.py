"""SmartGrid任务：DER经济目标、电压约束和CMDP状态；设备执行由公共流程负责。"""

import numpy as np
from envs.powergrid.control_env import DeviceControlEnv
from envs.smartgrid.base_env.env_config import SmartGridConfig
from envs.smartgrid.rewards.vvc_reward import VVCReward
from envs.smartgrid.rewards.lagrangian import LagrangianUpdater


class Env(DeviceControlEnv):
    def __init__(self, folder_path, config_or_info, dss_act=False):
        cfg = (
            config_or_info
            if isinstance(config_or_info, SmartGridConfig)
            else SmartGridConfig.from_env_args(config_or_info)
        )
        if dss_act:
            raise ValueError("SmartGrid训练由外部策略控制设备")
        overlays = (
            ()
            if cfg.pv_plan == "none"
            else (
                (
                    f"pv_plans/{cfg.pv_plan}.dss"
                    if cfg.pv_plan
                    else "pv_systems_base.dss"
                ),
            )
        )
        super().__init__(
            folder_path,
            cfg,
            overlays,
            pv_control_enabled=cfg.pv_control_enabled,
            pv_act_num=cfg.pv_act_num,
            svg_elements=cfg.svg_elements,
        )

    def _init_task(self):
        cfg = self.config
        self.reward_func = VVCReward(self, cfg)
        self.use_cmdp = cfg.use_cmdp
        self.lagrangian_updater = (
            LagrangianUpdater(
                init_lambda=cfg.lambda_init,
                lr=cfg.lambda_lr,
                target_cost=cfg.target_cost,
                lambda_max=cfg.lambda_max,
                lambda_min=cfg.lambda_min,
                update_strategy=cfg.lambda_update_strategy,
                momentum=cfg.lambda_momentum,
                adaptive_lr=cfg.lambda_adaptive_lr,
            )
            if cfg.use_cmdp
            else None
        )

    def _reset_task(self, seed):
        self.reward_func.reset_episode()
        self.episode_costs = []
        if seed is not None and self.lagrangian_updater:
            self.lagrangian_updater.reset(self.config.lambda_init)

    def _on_divergence(self):
        # 发散提前终止也要结算 CMDP：崩溃步按约束完全破坏（cost=1.0）计入，
        # 沿用自然终止的均值口径更新乘子——频繁失败的策略不能冻结乘子学习。
        if self.lagrangian_updater:
            self.episode_costs.append(1.0)
            self.lagrangian_updater.update(float(np.mean(self.episode_costs)))

    def _task_observation(self):
        return (
            [self.lagrangian_updater.lmbda]
            if self.config.use_lagrangian_reward and self.lagrangian_updater
            else []
        )

    def _reward(self, capdiff, regdiff, soc_err, dis_err, pvdiff):
        reward, info = self.reward_func.composite_reward(
            capdiff, regdiff, soc_err, dis_err, pvdiff
        )
        self.episode_costs.append(info["cost_voltage"])
        if self.lagrangian_updater:
            info["lambda"] = self.lagrangian_updater.lmbda
            if self.config.use_lagrangian_reward:
                info["reward_before_lagrangian"] = reward
                reward -= self.lagrangian_updater.lmbda * info["cost_voltage"]
            if self.t == self.horizon:
                self.lagrangian_updater.update(float(np.mean(self.episode_costs)))
        return reward, info

    def _task_observation_names(self):
        return (
            ["lagrangian_lambda"]
            if self.config.use_lagrangian_reward and self.lagrangian_updater
            else []
        )
