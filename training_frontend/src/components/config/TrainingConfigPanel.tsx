import {
	Card,
	Form,
	InputNumber,
	Switch,
	Slider,
	Space,
	Typography,
	Tooltip,
	Collapse,
} from 'antd';
import {
	SettingOutlined,
	QuestionCircleOutlined,
	ThunderboltOutlined,
	DesktopOutlined,
} from '@ant-design/icons';
import { useConfigStore } from '@/stores/configStore';

const { Text } = Typography;

export default function TrainingConfigPanel() {
	const { config, setConfig } = useConfigStore();

	return (
		<Card
			title={
				<Space>
					<SettingOutlined />
					<span>训练参数</span>
				</Space>
			}
			className="hover-card"
			style={{ height: '100%' }}
		>
			<Form layout="vertical" size="small">
				{/* 基础训练参数 */}
				<Form.Item
					label={
						<Space>
							<span>并行环境数</span>
							<Tooltip title="同时运行的环境实例数量，增加可以加速数据采集">
								<QuestionCircleOutlined style={{ color: 'rgba(255,255,255,0.45)' }} />
							</Tooltip>
						</Space>
					}
				>
					<Slider
						min={1}
						max={20}
						value={config.n_rollout_threads}
						onChange={(v) => setConfig({ n_rollout_threads: v })}
						marks={{ 1: '1', 4: '4', 8: '8', 16: '16', 20: '20' }}
					/>
				</Form.Item>

				<Form.Item
					label={
						<Space>
							<span>总训练步数</span>
							<Tooltip title="训练的总环境步数">
								<QuestionCircleOutlined style={{ color: 'rgba(255,255,255,0.45)' }} />
							</Tooltip>
						</Space>
					}
				>
					<InputNumber
						value={config.num_env_steps}
						onChange={(v) => setConfig({ num_env_steps: v || 2000000 })}
						min={10000}
						max={100000000}
						step={100000}
						style={{ width: '100%' }}
						formatter={(v) => `${v}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
						parser={(v) => Number(v?.replace(/,/g, '') || 0)}
					/>
				</Form.Item>

				<Form.Item
					label={
						<Space>
							<span>Episode长度</span>
							<Tooltip title="每个episode的步数，电力系统通常为360（对应6小时，每步1分钟）">
								<QuestionCircleOutlined style={{ color: 'rgba(255,255,255,0.45)' }} />
							</Tooltip>
						</Space>
					}
				>
					<Slider
						min={24}
						max={720}
						step={24}
						value={config.episode_length}
						onChange={(v) => setConfig({ episode_length: v })}
						marks={{ 24: '24', 96: '96', 360: '360', 720: '720' }}
					/>
				</Form.Item>

				{/* 间隔设置 */}
				<Collapse
					ghost
					items={[{
						key: 'intervals',
						label: <Text type="secondary">间隔设置</Text>,
						children: (
							<>
								<Form.Item label="日志间隔">
									<InputNumber
										value={config.log_interval}
										onChange={(v) => setConfig({ log_interval: v || 1 })}
										min={1}
										max={100}
										style={{ width: '100%' }}
										addonAfter="episodes"
									/>
								</Form.Item>
								<Form.Item label="评估间隔">
									<InputNumber
										value={config.eval_interval}
										onChange={(v) => setConfig({ eval_interval: v || 2 })}
										min={1}
										max={100}
										style={{ width: '100%' }}
										addonAfter="episodes"
									/>
								</Form.Item>
								<Form.Item label="保存间隔">
									<InputNumber
										value={config.save_interval}
										onChange={(v) => setConfig({ save_interval: v || 5 })}
										min={1}
										max={100}
										style={{ width: '100%' }}
										addonAfter="episodes"
									/>
								</Form.Item>
							</>
						),
					}]}
				/>

				{/* 评估设置 */}
				<Collapse
					ghost
					items={[{
						key: 'eval',
						label: <Text type="secondary">评估设置</Text>,
						children: (
							<>
								<Form.Item label="启用评估">
									<Switch
										checked={config.use_eval}
										onChange={(v) => setConfig({ use_eval: v })}
									/>
								</Form.Item>
								{config.use_eval && (
									<>
										<Form.Item label="评估并行数">
											<InputNumber
												value={config.n_eval_rollout_threads}
												onChange={(v) => setConfig({ n_eval_rollout_threads: v || 2 })}
												min={1}
												max={10}
												style={{ width: '100%' }}
											/>
										</Form.Item>
										<Form.Item label="评估Episodes">
											<InputNumber
												value={config.eval_episodes}
												onChange={(v) => setConfig({ eval_episodes: v || 10 })}
												min={1}
												max={100}
												style={{ width: '100%' }}
											/>
										</Form.Item>
									</>
								)}
							</>
						),
					}]}
				/>

				{/* 设备设置 */}
				<Collapse
					ghost
					items={[{
						key: 'device',
						label: (
							<Space>
								<DesktopOutlined />
								<Text type="secondary">设备设置</Text>
							</Space>
						),
						children: (
							<>
								<Form.Item label="使用CUDA">
									<Switch
										checked={config.cuda}
										onChange={(v) => setConfig({ cuda: v })}
									/>
								</Form.Item>
								<Form.Item label="CUDA确定性">
									<Switch
										checked={config.cuda_deterministic}
										onChange={(v) => setConfig({ cuda_deterministic: v })}
									/>
								</Form.Item>
								<Form.Item label="PyTorch线程数">
									<InputNumber
										value={config.torch_threads}
										onChange={(v) => setConfig({ torch_threads: v || 4 })}
										min={1}
										max={32}
										style={{ width: '100%' }}
									/>
								</Form.Item>
							</>
						),
					}]}
				/>

				{/* 种子设置 */}
				<Collapse
					ghost
					items={[{
						key: 'seed',
						label: <Text type="secondary">随机种子</Text>,
						children: (
							<>
								<Form.Item label="指定种子">
									<Switch
										checked={config.seed_specify}
										onChange={(v) => setConfig({ seed_specify: v })}
									/>
								</Form.Item>
								{config.seed_specify && (
									<Form.Item label="种子值">
										<InputNumber
											value={config.seed}
											onChange={(v) => setConfig({ seed: v || 12345 })}
											min={1}
											max={999999999}
											style={{ width: '100%' }}
										/>
									</Form.Item>
								)}
							</>
						),
					}]}
				/>

				{/* 高级选项 */}
				<Collapse
					ghost
					items={[{
						key: 'advanced',
						label: (
							<Space>
								<ThunderboltOutlined />
								<Text type="secondary">高级选项</Text>
							</Space>
						),
						children: (
							<>
								<Form.Item label="值归一化">
									<Switch
										checked={config.use_valuenorm}
										onChange={(v) => setConfig({ use_valuenorm: v })}
									/>
								</Form.Item>
								<Form.Item label="线性学习率衰减">
									<Switch
										checked={config.use_linear_lr_decay}
										onChange={(v) => setConfig({ use_linear_lr_decay: v })}
									/>
								</Form.Item>
							</>
						),
					}]}
				/>
			</Form>
		</Card>
	);
}
