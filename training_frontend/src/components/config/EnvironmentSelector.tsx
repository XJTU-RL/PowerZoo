import { useMemo } from 'react';
import {
	Card,
	Select,
	Typography,
	Tag,
	Space,
	Divider,
	Input,
} from 'antd';
import {
	AppstoreOutlined,
	TeamOutlined,
	EnvironmentOutlined,
} from '@ant-design/icons';
import { useConfigStore } from '@/stores/configStore';

const { Text } = Typography;
const { Option, OptGroup } = Select;

export default function EnvironmentSelector() {
	const { environments, config, setConfig, setAlgoAndEnv } = useConfigStore();

	const currentEnv = useMemo(() => {
		return environments.find((e) => e.name === config.env);
	}, [environments, config.env]);

	const handleChange = (value: string) => {
		setAlgoAndEnv(config.algo, value);
	};

	const getTypeColor = (type: string) => {
		switch (type) {
			case 'vvc':
				return 'green';
			case 'microgrid':
				return 'blue';
			case 'game':
				return 'purple';
			case 'restoration':
				return 'orange';
			case 'standard':
				return 'default';
			default:
				return 'cyan';
		}
	};

	const getTypeLabel = (type: string) => {
		switch (type) {
			case 'vvc':
				return 'Volt-VAR Control';
			case 'microgrid':
				return 'Microgrid';
			case 'game':
				return 'Game Theory';
			case 'restoration':
				return 'Restoration';
			case 'standard':
				return 'Standard';
			default:
				return type;
		}
	};

	// 按类型分组环境
	const groupedEnvs = useMemo(() => {
		const groups: Record<string, typeof environments> = {};
		environments.forEach((env) => {
			if (!groups[env.type]) {
				groups[env.type] = [];
			}
			groups[env.type].push(env);
		});
		return groups;
	}, [environments]);

	return (
		<Card
			title={
				<Space>
					<AppstoreOutlined />
					<span>环境选择</span>
				</Space>
			}
			className="hover-card"
		>
			<Select
				value={config.env}
				onChange={handleChange}
				style={{ width: '100%', marginBottom: 16 }}
				size="large"
				showSearch
				filterOption={(input, option) =>
					(option?.label as string)?.toLowerCase().includes(input.toLowerCase())
				}
			>
				{Object.entries(groupedEnvs).map(([type, envs]) => (
					<OptGroup key={type} label={getTypeLabel(type)}>
						{envs.map((env) => (
							<Option key={env.name} value={env.name} label={env.display_name}>
								<Space style={{ width: '100%', justifyContent: 'space-between' }}>
									<Space>
										<EnvironmentOutlined />
										<span>{env.display_name}</span>
									</Space>
									<Space>
										<Tag color={getTypeColor(env.type)}>{getTypeLabel(env.type)}</Tag>
										<Tag icon={<TeamOutlined />}>{env.default_agents}</Tag>
									</Space>
								</Space>
							</Option>
						))}
					</OptGroup>
				))}
			</Select>

			{currentEnv && (
				<div style={{ padding: '12px', background: 'rgba(0,0,0,0.2)', borderRadius: 8 }}>
					<Space direction="vertical" size="small" style={{ width: '100%' }}>
						<div>
							<Text strong style={{ fontSize: 16 }}>{currentEnv.display_name}</Text>
							<Tag color={getTypeColor(currentEnv.type)} style={{ marginLeft: 8 }}>
								{getTypeLabel(currentEnv.type)}
							</Tag>
							<Tag icon={<TeamOutlined />} style={{ marginLeft: 4 }}>
								{currentEnv.default_agents} Agents
							</Tag>
						</div>
						<Text type="secondary" style={{ fontSize: 12 }}>
							{currentEnv.description}
						</Text>
						<Divider style={{ margin: '8px 0' }} />
						<Text type="secondary" style={{ fontSize: 11 }}>
							Config: <Text code style={{ fontSize: 11 }}>{currentEnv.config_file}</Text>
						</Text>
					</Space>
				</div>
			)}

			<Divider style={{ margin: '16px 0 12px' }} />

			<div>
				<Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 8 }}>
					实验名称
				</Text>
				<Input
					value={config.exp_name}
					onChange={(e) => setConfig({ exp_name: e.target.value })}
					placeholder="输入实验名称"
					style={{ width: '100%' }}
				/>
			</div>
		</Card>
	);
}
