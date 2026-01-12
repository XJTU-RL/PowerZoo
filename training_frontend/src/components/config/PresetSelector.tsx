import {
	Dropdown,
	Button,
	Space,
	Typography,
	Tag,
} from 'antd';
import type { MenuProps } from 'antd';
import {
	DownOutlined,
	ThunderboltOutlined,
	RocketOutlined,
	ExperimentOutlined,
	BugOutlined,
	FireOutlined,
} from '@ant-design/icons';
import { useConfigStore } from '@/stores/configStore';

const { Text } = Typography;

export default function PresetSelector() {
	const { presets, loadPreset } = useConfigStore();

	const getPresetIcon = (name: string) => {
		switch (name) {
			case 'quick_test':
				return <BugOutlined />;
			case 'standard_training':
				return <ThunderboltOutlined />;
			case 'high_performance':
				return <FireOutlined />;
			case 'stackelberg_game':
				return <ExperimentOutlined />;
			default:
				return <RocketOutlined />;
		}
	};

	const getPresetColor = (name: string) => {
		switch (name) {
			case 'quick_test':
				return 'default';
			case 'standard_training':
				return 'blue';
			case 'high_performance':
				return 'red';
			case 'stackelberg_game':
				return 'purple';
			case 'dsr_restoration':
				return 'orange';
			case 'off_policy_training':
				return 'cyan';
			default:
				return 'default';
		}
	};

	const menuItems: MenuProps['items'] = presets.map((preset) => ({
		key: preset.name,
		label: (
			<div style={{ padding: '4px 0', minWidth: 250 }}>
				<Space style={{ width: '100%', justifyContent: 'space-between' }}>
					<Space>
						{getPresetIcon(preset.name)}
						<span>{preset.description}</span>
					</Space>
					<Space>
						<Tag color="blue">{preset.config.algo.toUpperCase()}</Tag>
						<Tag color="cyan">{preset.config.env}</Tag>
					</Space>
				</Space>
				<div style={{ marginTop: 4 }}>
					<Text type="secondary" style={{ fontSize: 11 }}>
						{preset.config.n_rollout_threads} threads | {(preset.config.num_env_steps / 1000000).toFixed(1)}M steps | ep={preset.config.episode_length}
					</Text>
				</div>
			</div>
		),
		onClick: () => loadPreset(preset.name),
	}));

	return (
		<Dropdown menu={{ items: menuItems }} trigger={['click']}>
			<Button icon={<RocketOutlined />}>
				加载预设 <DownOutlined />
			</Button>
		</Dropdown>
	);
}
