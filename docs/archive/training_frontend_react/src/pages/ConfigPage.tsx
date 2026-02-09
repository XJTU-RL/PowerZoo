import { useState } from 'react';
import {
	Row,
	Col,
	Card,
	Button,
	Space,
	Typography,
	Divider,
	message,
	Modal,
	Tooltip,
	Badge,
} from 'antd';
import {
	PlayCircleOutlined,
	SaveOutlined,
	ReloadOutlined,
	CheckCircleOutlined,
	WarningOutlined,
	CloseCircleOutlined,
	RocketOutlined,
	CodeOutlined,
} from '@ant-design/icons';
import { useConfigStore } from '@/stores/configStore';
import AlgorithmSelector from '@/components/config/AlgorithmSelector';
import EnvironmentSelector from '@/components/config/EnvironmentSelector';
import TrainingConfigPanel from '@/components/config/TrainingConfigPanel';
import ModelConfigPanel from '@/components/config/ModelConfigPanel';
import AlgoConfigPanel from '@/components/config/AlgoConfigPanel';
import PresetSelector from '@/components/config/PresetSelector';

const { Title, Text } = Typography;

export default function ConfigPage() {
	const {
		config,
		validation,
		validateConfig,
		startTraining,
		resetConfig,
	} = useConfigStore();

	const [validating, setValidating] = useState(false);
	const [starting, setStarting] = useState(false);
	const [showPreview, setShowPreview] = useState(false);

	const handleValidate = async () => {
		setValidating(true);
		try {
			const result = await validateConfig();
			if (result.valid) {
				message.success('配置验证通过');
			} else {
				message.error('配置验证失败，请检查错误信息');
			}
		} catch (error) {
			message.error(`验证失败: ${error}`);
		} finally {
			setValidating(false);
		}
	};

	const handleStart = async () => {
		// 先验证
		const result = await validateConfig();
		if (!result.valid) {
			message.error('配置验证失败，请先修复错误');
			return;
		}

		setStarting(true);
		try {
			const taskId = await startTraining();
			message.success(`训练任务已启动，任务ID: ${taskId}`);
		} catch (error) {
			message.error(`启动失败: ${error}`);
		} finally {
			setStarting(false);
		}
	};

	const handleReset = () => {
		Modal.confirm({
			title: '确认重置',
			content: '确定要重置所有配置为默认值吗？',
			onOk: () => {
				resetConfig();
				message.info('配置已重置');
			},
		});
	};

	const renderValidationStatus = () => {
		if (!validation) return null;

		if (validation.valid && validation.warnings.length === 0) {
			return (
				<Badge status="success" text="配置有效" />
			);
		}

		if (validation.valid && validation.warnings.length > 0) {
			return (
				<Tooltip title={validation.warnings.join('\n')}>
					<Badge status="warning" text={`${validation.warnings.length} 个警告`} />
				</Tooltip>
			);
		}

		return (
			<Tooltip title={validation.errors.join('\n')}>
				<Badge status="error" text={`${validation.errors.length} 个错误`} />
			</Tooltip>
		);
	};

	return (
		<div>
			{/* 顶部操作栏 */}
			<Card
				style={{ marginBottom: 16 }}
				bodyStyle={{ padding: '16px 24px' }}
			>
				<Row justify="space-between" align="middle">
					<Col>
						<Space size="large">
							<Title level={4} style={{ margin: 0 }}>
								<RocketOutlined style={{ marginRight: 8 }} />
								训练配置
							</Title>
							{renderValidationStatus()}
						</Space>
					</Col>
					<Col>
						<Space>
							<PresetSelector />
							<Button
								icon={<CodeOutlined />}
								onClick={() => setShowPreview(true)}
							>
								预览配置
							</Button>
							<Button
								icon={<ReloadOutlined />}
								onClick={handleReset}
							>
								重置
							</Button>
							<Button
								icon={<CheckCircleOutlined />}
								loading={validating}
								onClick={handleValidate}
							>
								验证
							</Button>
							<Button
								type="primary"
								icon={<PlayCircleOutlined />}
								loading={starting}
								onClick={handleStart}
								size="large"
							>
								启动训练
							</Button>
						</Space>
					</Col>
				</Row>
			</Card>

			{/* 验证结果显示 */}
			{validation && !validation.valid && (
				<Card
					style={{ marginBottom: 16, borderColor: '#ff4d4f' }}
					size="small"
				>
					<Space direction="vertical" style={{ width: '100%' }}>
						{validation.errors.map((err, i) => (
							<Text key={i} type="danger">
								<CloseCircleOutlined style={{ marginRight: 8 }} />
								{err}
							</Text>
						))}
						{validation.warnings.map((warn, i) => (
							<Text key={i} type="warning">
								<WarningOutlined style={{ marginRight: 8 }} />
								{warn}
							</Text>
						))}
					</Space>
				</Card>
			)}

			{/* 基础选择区域 */}
			<Row gutter={16} style={{ marginBottom: 16 }}>
				<Col span={12}>
					<AlgorithmSelector />
				</Col>
				<Col span={12}>
					<EnvironmentSelector />
				</Col>
			</Row>

			{/* 配置面板区域 */}
			<Row gutter={16}>
				<Col span={8}>
					<TrainingConfigPanel />
				</Col>
				<Col span={8}>
					<ModelConfigPanel />
				</Col>
				<Col span={8}>
					<AlgoConfigPanel />
				</Col>
			</Row>

			{/* 配置预览弹窗 */}
			<Modal
				title="配置预览 (YAML)"
				open={showPreview}
				onCancel={() => setShowPreview(false)}
				footer={[
					<Button key="close" onClick={() => setShowPreview(false)}>
						关闭
					</Button>,
					<Button
						key="copy"
						type="primary"
						onClick={() => {
							navigator.clipboard.writeText(JSON.stringify(config, null, 2));
							message.success('配置已复制到剪贴板');
						}}
					>
						复制配置
					</Button>,
				]}
				width={700}
			>
				<pre className="config-preview" style={{ color: '#e6e6e6' }}>
					{JSON.stringify(config, null, 2)}
				</pre>
			</Modal>
		</div>
	);
}
