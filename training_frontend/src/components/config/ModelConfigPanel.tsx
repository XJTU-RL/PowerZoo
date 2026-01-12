import {
	Card,
	Form,
	InputNumber,
	Switch,
	Select,
	Space,
	Typography,
	Tooltip,
	Collapse,
	Tag,
} from 'antd';
import {
	ApartmentOutlined,
	QuestionCircleOutlined,
} from '@ant-design/icons';
import { useConfigStore } from '@/stores/configStore';

const { Text } = Typography;
const { Option } = Select;

export default function ModelConfigPanel() {
	const { config, setConfig } = useConfigStore();

	const handleHiddenSizesChange = (sizes: string) => {
		const parsed = sizes.split(',').map((s) => parseInt(s.trim())).filter((n) => !isNaN(n));
		if (parsed.length > 0) {
			setConfig({ hidden_sizes: parsed });
		}
	};

	const hiddenSizePresets = [
		{ label: '小型 [64, 64]', value: '64,64' },
		{ label: '中型 [128, 128]', value: '128,128' },
		{ label: '大型 [256, 256]', value: '256,256' },
		{ label: '超大型 [512, 512]', value: '512,512' },
		{ label: '三层 [128, 128, 128]', value: '128,128,128' },
		{ label: '金字塔 [256, 128, 64]', value: '256,128,64' },
	];

	return (
		<Card
			title={
				<Space>
					<ApartmentOutlined />
					<span>模型结构</span>
				</Space>
			}
			className="hover-card"
			style={{ height: '100%' }}
		>
			<Form layout="vertical" size="small">
				{/* 隐藏层配置 */}
				<Form.Item
					label={
						<Space>
							<span>隐藏层大小</span>
							<Tooltip title="MLP网络的隐藏层维度，如[128,128]表示两层各128个神经元">
								<QuestionCircleOutlined style={{ color: 'rgba(255,255,255,0.45)' }} />
							</Tooltip>
						</Space>
					}
				>
					<Select
						value={config.hidden_sizes.join(',')}
						onChange={handleHiddenSizesChange}
						style={{ width: '100%' }}
					>
						{hiddenSizePresets.map((preset) => (
							<Option key={preset.value} value={preset.value}>
								{preset.label}
							</Option>
						))}
					</Select>
					<div style={{ marginTop: 8 }}>
						<Text type="secondary" style={{ fontSize: 11 }}>
							当前: {config.hidden_sizes.map((s, i) => (
								<Tag key={i} color="blue" style={{ marginRight: 4 }}>{s}</Tag>
							))}
						</Text>
					</div>
				</Form.Item>

				{/* 激活函数 */}
				<Form.Item label="激活函数">
					<Select
						value={config.activation_func}
						onChange={(v) => setConfig({ activation_func: v })}
						style={{ width: '100%' }}
					>
						<Option value="relu">ReLU</Option>
						<Option value="tanh">Tanh</Option>
						<Option value="sigmoid">Sigmoid</Option>
						<Option value="leaky_relu">Leaky ReLU</Option>
						<Option value="elu">ELU</Option>
						<Option value="gelu">GELU</Option>
					</Select>
				</Form.Item>

				{/* 特征归一化 */}
				<Form.Item label="特征归一化">
					<Switch
						checked={config.use_feature_normalization}
						onChange={(v) => setConfig({ use_feature_normalization: v })}
					/>
				</Form.Item>

				{/* RNN配置 */}
				<Collapse
					ghost
					defaultActiveKey={config.use_recurrent_policy ? ['rnn'] : []}
					items={[{
						key: 'rnn',
						label: (
							<Space>
								<span>循环网络 (RNN)</span>
								<Tag color={config.use_recurrent_policy ? 'green' : 'default'}>
									{config.use_recurrent_policy ? 'ON' : 'OFF'}
								</Tag>
							</Space>
						),
						children: (
							<>
								<Form.Item label="使用循环策略">
									<Switch
										checked={config.use_recurrent_policy}
										onChange={(v) => setConfig({ use_recurrent_policy: v })}
									/>
								</Form.Item>
								{config.use_recurrent_policy && (
									<>
										<Form.Item
											label={
												<Space>
													<span>RNN层数</span>
													<Tooltip title="GRU/LSTM的层数">
														<QuestionCircleOutlined style={{ color: 'rgba(255,255,255,0.45)' }} />
													</Tooltip>
												</Space>
											}
										>
											<InputNumber
												value={config.recurrent_n}
												onChange={(v) => setConfig({ recurrent_n: v || 1 })}
												min={1}
												max={4}
												style={{ width: '100%' }}
											/>
										</Form.Item>
										<Form.Item
											label={
												<Space>
													<span>数据块长度</span>
													<Tooltip title="用于训练RNN的序列长度，建议为episode_length的因子">
														<QuestionCircleOutlined style={{ color: 'rgba(255,255,255,0.45)' }} />
													</Tooltip>
												</Space>
											}
										>
											<InputNumber
												value={config.data_chunk_length}
												onChange={(v) => setConfig({ data_chunk_length: v || 60 })}
												min={1}
												max={360}
												step={10}
												style={{ width: '100%' }}
											/>
										</Form.Item>
										{config.episode_length % config.data_chunk_length !== 0 && (
											<Text type="warning" style={{ fontSize: 11 }}>
												警告: episode_length ({config.episode_length}) 应为 data_chunk_length ({config.data_chunk_length}) 的整数倍
											</Text>
										)}
									</>
								)}
							</>
						),
					}]}
				/>

				{/* 网络结构说明 */}
				<div style={{
					marginTop: 16,
					padding: 12,
					background: 'rgba(0,0,0,0.2)',
					borderRadius: 8,
				}}>
					<Text type="secondary" style={{ fontSize: 11 }}>
						<strong>网络结构摘要:</strong>
						<br />
						<br />
						Actor: Input → {config.hidden_sizes.join(' → ')} → Output
						<br />
						Critic: Input → {config.hidden_sizes.join(' → ')} → Value
						<br />
						{config.use_recurrent_policy && (
							<>RNN: {config.recurrent_n}层 GRU, chunk={config.data_chunk_length}</>
						)}
					</Text>
				</div>
			</Form>
		</Card>
	);
}
