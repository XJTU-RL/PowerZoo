import { useMemo } from 'react';
import {
	Card,
	Form,
	InputNumber,
	Slider,
	Space,
	Typography,
	Tooltip,
	Collapse,
	Tag,
} from 'antd';
import {
	ExperimentOutlined,
	QuestionCircleOutlined,
} from '@ant-design/icons';
import { useConfigStore } from '@/stores/configStore';

const { Text } = Typography;

export default function AlgoConfigPanel() {
	const { algorithms, config, setConfig } = useConfigStore();

	const currentAlgo = useMemo(() => {
		return (
			algorithms.multi_agent.find((a) => a.name === config.algo) ||
			algorithms.single_agent.find((a) => a.name === config.algo)
		);
	}, [algorithms, config.algo]);

	const isOnPolicy = currentAlgo?.type === 'on_policy';
	const isOffPolicy = currentAlgo?.type === 'off_policy';

	return (
		<Card
			title={
				<Space>
					<ExperimentOutlined />
					<span>算法参数</span>
					{currentAlgo && (
						<Tag color={isOnPolicy ? 'green' : isOffPolicy ? 'blue' : 'purple'}>
							{currentAlgo.type.replace('_', '-')}
						</Tag>
					)}
				</Space>
			}
			className="hover-card"
			style={{ height: '100%' }}
		>
			<Form layout="vertical" size="small">
				{/* 通用优化器参数 */}
				<Form.Item
					label={
						<Space>
							<span>Actor学习率</span>
							<Tooltip title="策略网络的学习率">
								<QuestionCircleOutlined style={{ color: 'rgba(255,255,255,0.45)' }} />
							</Tooltip>
						</Space>
					}
				>
					<InputNumber
						value={config.lr}
						onChange={(v) => setConfig({ lr: v || 5e-4 })}
						min={1e-7}
						max={1e-1}
						step={1e-5}
						style={{ width: '100%' }}
						formatter={(v) => v?.toExponential(2) || '5e-4'}
					/>
				</Form.Item>

				<Form.Item
					label={
						<Space>
							<span>Critic学习率</span>
							<Tooltip title="价值网络的学习率，通常与Actor相同或略大">
								<QuestionCircleOutlined style={{ color: 'rgba(255,255,255,0.45)' }} />
							</Tooltip>
						</Space>
					}
				>
					<InputNumber
						value={config.critic_lr}
						onChange={(v) => setConfig({ critic_lr: v || 5e-4 })}
						min={1e-7}
						max={1e-1}
						step={1e-5}
						style={{ width: '100%' }}
						formatter={(v) => v?.toExponential(2) || '5e-4'}
					/>
				</Form.Item>

				<Form.Item
					label={
						<Space>
							<span>折扣因子 (Gamma)</span>
							<Tooltip title="未来奖励的折扣系数，越接近1越看重长期回报">
								<QuestionCircleOutlined style={{ color: 'rgba(255,255,255,0.45)' }} />
							</Tooltip>
						</Space>
					}
				>
					<Slider
						min={0.9}
						max={0.999}
						step={0.001}
						value={config.gamma}
						onChange={(v) => setConfig({ gamma: v })}
						marks={{ 0.9: '0.9', 0.95: '0.95', 0.99: '0.99', 0.999: '0.999' }}
					/>
				</Form.Item>

				{/* On-Policy 特定参数 */}
				{isOnPolicy && (
					<Collapse
						ghost
						defaultActiveKey={['on_policy']}
						items={[{
							key: 'on_policy',
							label: <Text type="secondary">On-Policy 参数</Text>,
							children: (
								<>
									<Form.Item
										label={
											<Space>
												<span>GAE Lambda</span>
												<Tooltip title="广义优势估计的平滑参数">
													<QuestionCircleOutlined style={{ color: 'rgba(255,255,255,0.45)' }} />
												</Tooltip>
											</Space>
										}
									>
										<Slider
											min={0.9}
											max={1.0}
											step={0.01}
											value={config.gae_lambda}
											onChange={(v) => setConfig({ gae_lambda: v })}
											marks={{ 0.9: '0.9', 0.95: '0.95', 1.0: '1.0' }}
										/>
									</Form.Item>

									<Form.Item
										label={
											<Space>
												<span>PPO Epoch</span>
												<Tooltip title="每批数据的训练轮数">
													<QuestionCircleOutlined style={{ color: 'rgba(255,255,255,0.45)' }} />
												</Tooltip>
											</Space>
										}
									>
										<InputNumber
											value={config.ppo_epoch}
											onChange={(v) => setConfig({ ppo_epoch: v || 5 })}
											min={1}
											max={20}
											style={{ width: '100%' }}
										/>
									</Form.Item>

									<Form.Item
										label={
											<Space>
												<span>Clip参数</span>
												<Tooltip title="PPO裁剪范围，限制策略更新幅度">
													<QuestionCircleOutlined style={{ color: 'rgba(255,255,255,0.45)' }} />
												</Tooltip>
											</Space>
										}
									>
										<Slider
											min={0.1}
											max={0.5}
											step={0.05}
											value={config.clip_param}
											onChange={(v) => setConfig({ clip_param: v })}
											marks={{ 0.1: '0.1', 0.2: '0.2', 0.3: '0.3', 0.5: '0.5' }}
										/>
									</Form.Item>

									<Form.Item
										label={
											<Space>
												<span>熵系数</span>
												<Tooltip title="熵正则化系数，增加探索性">
													<QuestionCircleOutlined style={{ color: 'rgba(255,255,255,0.45)' }} />
												</Tooltip>
											</Space>
										}
									>
										<Slider
											min={0}
											max={0.2}
											step={0.01}
											value={config.entropy_coef}
											onChange={(v) => setConfig({ entropy_coef: v })}
											marks={{ 0: '0', 0.05: '0.05', 0.1: '0.1', 0.2: '0.2' }}
										/>
									</Form.Item>

									<Form.Item label="Actor Mini-batch数">
										<InputNumber
											value={config.actor_num_mini_batch}
											onChange={(v) => setConfig({ actor_num_mini_batch: v || 1 })}
											min={1}
											max={10}
											style={{ width: '100%' }}
										/>
									</Form.Item>

									<Form.Item label="Critic Mini-batch数">
										<InputNumber
											value={config.critic_num_mini_batch}
											onChange={(v) => setConfig({ critic_num_mini_batch: v || 1 })}
											min={1}
											max={10}
											style={{ width: '100%' }}
										/>
									</Form.Item>
								</>
							),
						}]}
					/>
				)}

				{/* Off-Policy 特定参数 */}
				{isOffPolicy && (
					<Collapse
						ghost
						defaultActiveKey={['off_policy']}
						items={[{
							key: 'off_policy',
							label: <Text type="secondary">Off-Policy 参数</Text>,
							children: (
								<>
									<Form.Item
										label={
											<Space>
												<span>经验池大小</span>
												<Tooltip title="经验回放缓冲区的容量">
													<QuestionCircleOutlined style={{ color: 'rgba(255,255,255,0.45)' }} />
												</Tooltip>
											</Space>
										}
									>
										<InputNumber
											value={config.buffer_size}
											onChange={(v) => setConfig({ buffer_size: v || 100000 })}
											min={10000}
											max={10000000}
											step={10000}
											style={{ width: '100%' }}
											formatter={(v) => `${v}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
											parser={(v) => Number(v?.replace(/,/g, '') || 0)}
										/>
									</Form.Item>

									<Form.Item label="批量大小">
										<InputNumber
											value={config.batch_size}
											onChange={(v) => setConfig({ batch_size: v || 256 })}
											min={32}
											max={2048}
											step={32}
											style={{ width: '100%' }}
										/>
									</Form.Item>

									<Form.Item
										label={
											<Space>
												<span>Polyak系数</span>
												<Tooltip title="目标网络软更新系数">
													<QuestionCircleOutlined style={{ color: 'rgba(255,255,255,0.45)' }} />
												</Tooltip>
											</Space>
										}
									>
										<InputNumber
											value={config.polyak}
											onChange={(v) => setConfig({ polyak: v || 0.005 })}
											min={0.001}
											max={0.1}
											step={0.001}
											style={{ width: '100%' }}
										/>
									</Form.Item>
								</>
							),
						}]}
					/>
				)}

				{/* 通用高级参数 */}
				<Collapse
					ghost
					items={[{
						key: 'advanced',
						label: <Text type="secondary">高级参数</Text>,
						children: (
							<>
								<Form.Item
									label={
										<Space>
											<span>最大梯度范数</span>
											<Tooltip title="梯度裁剪的最大范数">
												<QuestionCircleOutlined style={{ color: 'rgba(255,255,255,0.45)' }} />
											</Tooltip>
										</Space>
									}
								>
									<InputNumber
										value={config.max_grad_norm}
										onChange={(v) => setConfig({ max_grad_norm: v || 10.0 })}
										min={0.1}
										max={100}
										step={0.1}
										style={{ width: '100%' }}
									/>
								</Form.Item>
							</>
						),
					}]}
				/>

				{/* 参数摘要 */}
				<div style={{
					marginTop: 16,
					padding: 12,
					background: 'rgba(0,0,0,0.2)',
					borderRadius: 8,
				}}>
					<Text type="secondary" style={{ fontSize: 11 }}>
						<strong>参数摘要:</strong>
						<br />
						<br />
						LR: {config.lr.toExponential(2)} / {config.critic_lr.toExponential(2)}
						<br />
						Gamma: {config.gamma} | Lambda: {config.gae_lambda}
						<br />
						{isOnPolicy && <>PPO: epoch={config.ppo_epoch}, clip={config.clip_param}</>}
						{isOffPolicy && <>Buffer: {(config.buffer_size/1000).toFixed(0)}K, batch={config.batch_size}</>}
					</Text>
				</div>
			</Form>
		</Card>
	);
}
