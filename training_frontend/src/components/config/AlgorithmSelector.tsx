import { useMemo } from 'react';
import {
	Card,
	Select,
	Typography,
	Tag,
	Space,
	Tooltip,
	Divider,
} from 'antd';
import {
	ThunderboltOutlined,
	ApiOutlined,
	BranchesOutlined,
} from '@ant-design/icons';
import { useConfigStore } from '@/stores/configStore';

const { Text } = Typography;
const { Option, OptGroup } = Select;

export default function AlgorithmSelector() {
	const { algorithms, config, setAlgoAndEnv } = useConfigStore();

	const currentAlgo = useMemo(() => {
		return (
			algorithms.multi_agent.find((a) => a.name === config.algo) ||
			algorithms.single_agent.find((a) => a.name === config.algo)
		);
	}, [algorithms, config.algo]);

	const handleChange = (value: string) => {
		setAlgoAndEnv(value, config.env);
	};

	const getTypeColor = (type: string) => {
		switch (type) {
			case 'on_policy':
				return 'green';
			case 'off_policy':
				return 'blue';
			case 'two_timescale':
				return 'purple';
			default:
				return 'default';
		}
	};

	const getCategoryIcon = (category: string) => {
		switch (category) {
			case 'ha':
				return <BranchesOutlined />;
			case 'ma':
				return <ApiOutlined />;
			default:
				return <ThunderboltOutlined />;
		}
	};

	return (
		<Card
			title={
				<Space>
					<ThunderboltOutlined />
					<span>算法选择</span>
				</Space>
			}
			className="hover-card"
		>
			<Select
				value={config.algo}
				onChange={handleChange}
				style={{ width: '100%', marginBottom: 16 }}
				size="large"
				showSearch
				filterOption={(input, option) =>
					(option?.label as string)?.toLowerCase().includes(input.toLowerCase())
				}
			>
				<OptGroup label="HA系列 (Heterogeneous-Agent)">
					{algorithms.multi_agent
						.filter((a) => a.category === 'ha')
						.map((algo) => (
							<Option key={algo.name} value={algo.name} label={algo.display_name}>
								<Space>
									{getCategoryIcon(algo.category)}
									<span>{algo.display_name}</span>
									<Tag color={getTypeColor(algo.type)} style={{ marginLeft: 'auto' }}>
										{algo.type.replace('_', '-')}
									</Tag>
								</Space>
							</Option>
						))}
				</OptGroup>
				<OptGroup label="MA系列 (Multi-Agent)">
					{algorithms.multi_agent
						.filter((a) => a.category === 'ma')
						.map((algo) => (
							<Option key={algo.name} value={algo.name} label={algo.display_name}>
								<Space>
									{getCategoryIcon(algo.category)}
									<span>{algo.display_name}</span>
									<Tag color={getTypeColor(algo.type)} style={{ marginLeft: 'auto' }}>
										{algo.type.replace('_', '-')}
									</Tag>
								</Space>
							</Option>
						))}
				</OptGroup>
				<OptGroup label="其他">
					{algorithms.multi_agent
						.filter((a) => !['ha', 'ma'].includes(a.category))
						.map((algo) => (
							<Option key={algo.name} value={algo.name} label={algo.display_name}>
								<Space>
									{getCategoryIcon(algo.category)}
									<span>{algo.display_name}</span>
									<Tag color={getTypeColor(algo.type)} style={{ marginLeft: 'auto' }}>
										{algo.type.replace('_', '-')}
									</Tag>
								</Space>
							</Option>
						))}
				</OptGroup>
				<OptGroup label="单智能体">
					{algorithms.single_agent.map((algo) => (
						<Option key={algo.name} value={algo.name} label={algo.display_name}>
							<Space>
								<ThunderboltOutlined />
								<span>{algo.display_name}</span>
								<Tag color={getTypeColor(algo.type)} style={{ marginLeft: 'auto' }}>
									{algo.type.replace('_', '-')}
								</Tag>
							</Space>
						</Option>
					))}
				</OptGroup>
			</Select>

			{currentAlgo && (
				<div style={{ padding: '12px', background: 'rgba(0,0,0,0.2)', borderRadius: 8 }}>
					<Space direction="vertical" size="small" style={{ width: '100%' }}>
						<div>
							<Text strong style={{ fontSize: 16 }}>{currentAlgo.display_name}</Text>
							<Tag color={getTypeColor(currentAlgo.type)} style={{ marginLeft: 8 }}>
								{currentAlgo.type.replace('_', '-')}
							</Tag>
							<Tag color="default" style={{ marginLeft: 4 }}>
								{currentAlgo.category.toUpperCase()}
							</Tag>
						</div>
						<Text type="secondary" style={{ fontSize: 12 }}>
							{currentAlgo.description}
						</Text>
						<Divider style={{ margin: '8px 0' }} />
						<Space>
							<Text type="secondary" style={{ fontSize: 11 }}>
								Runner: <Text code style={{ fontSize: 11 }}>{currentAlgo.runner}</Text>
							</Text>
							<Text type="secondary" style={{ fontSize: 11 }}>
								Config: <Text code style={{ fontSize: 11 }}>{currentAlgo.config_file}</Text>
							</Text>
						</Space>
					</Space>
				</div>
			)}
		</Card>
	);
}
