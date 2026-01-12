import { useEffect, useState } from 'react';
import {
	Card,
	Table,
	Button,
	Space,
	Tag,
	Modal,
	Typography,
	Popconfirm,
	message,
	Empty,
	Tooltip,
	Row,
	Col,
	Statistic,
} from 'antd';
import {
	PlayCircleOutlined,
	PauseCircleOutlined,
	DeleteOutlined,
	FileTextOutlined,
	ReloadOutlined,
	SyncOutlined,
	CheckCircleOutlined,
	CloseCircleOutlined,
	ClockCircleOutlined,
	ExperimentOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { useConfigStore } from '@/stores/configStore';
import type { TrainingTask } from '@/types';
import * as api from '@/services/api';

const { Title, Text } = Typography;

export default function TasksPage() {
	const { tasks, loadTasks, stopTask, deleteTask } = useConfigStore();
	const [loading, setLoading] = useState(false);
	const [logModal, setLogModal] = useState<{ visible: boolean; taskId: string; logs: string }>({
		visible: false,
		taskId: '',
		logs: '',
	});
	const [logLoading, setLogLoading] = useState(false);

	useEffect(() => {
		loadTasks();
		// 每10秒刷新一次
		const interval = setInterval(() => {
			loadTasks();
		}, 10000);
		return () => clearInterval(interval);
	}, [loadTasks]);

	const handleRefresh = async () => {
		setLoading(true);
		try {
			await loadTasks();
			message.success('刷新成功');
		} catch (error) {
			message.error(`刷新失败: ${error}`);
		} finally {
			setLoading(false);
		}
	};

	const handleStop = async (taskId: string) => {
		try {
			await stopTask(taskId);
			message.success('任务已停止');
		} catch (error) {
			message.error(`停止失败: ${error}`);
		}
	};

	const handleDelete = async (taskId: string) => {
		try {
			await deleteTask(taskId);
			message.success('任务已删除');
		} catch (error) {
			message.error(`删除失败: ${error}`);
		}
	};

	const handleViewLogs = async (taskId: string) => {
		setLogLoading(true);
		setLogModal({ visible: true, taskId, logs: '' });
		try {
			const result = await api.getTaskLogs(taskId, 200);
			setLogModal({ visible: true, taskId, logs: result.logs || '暂无日志' });
		} catch (error) {
			setLogModal({ visible: true, taskId, logs: `获取日志失败: ${error}` });
		} finally {
			setLogLoading(false);
		}
	};

	const refreshLogs = async () => {
		if (!logModal.taskId) return;
		setLogLoading(true);
		try {
			const result = await api.getTaskLogs(logModal.taskId, 200);
			setLogModal((prev) => ({ ...prev, logs: result.logs || '暂无日志' }));
		} catch (error) {
			message.error(`刷新日志失败: ${error}`);
		} finally {
			setLogLoading(false);
		}
	};

	const getStatusTag = (status: string) => {
		const statusConfig: Record<string, { color: string; icon: React.ReactNode }> = {
			pending: { color: 'default', icon: <ClockCircleOutlined /> },
			running: { color: 'processing', icon: <SyncOutlined spin /> },
			completed: { color: 'success', icon: <CheckCircleOutlined /> },
			failed: { color: 'error', icon: <CloseCircleOutlined /> },
			stopped: { color: 'warning', icon: <PauseCircleOutlined /> },
		};
		const config = statusConfig[status] || statusConfig.pending;
		return (
			<Tag color={config.color} icon={config.icon}>
				{status.toUpperCase()}
			</Tag>
		);
	};

	const columns: ColumnsType<TrainingTask> = [
		{
			title: '任务ID',
			dataIndex: 'task_id',
			key: 'task_id',
			width: 100,
			render: (id) => <Text code>{id}</Text>,
		},
		{
			title: '算法',
			key: 'algo',
			width: 100,
			render: (_, record) => (
				<Tag color="blue">{record.config.algo.toUpperCase()}</Tag>
			),
		},
		{
			title: '环境',
			key: 'env',
			width: 120,
			render: (_, record) => (
				<Tag color="cyan">{record.config.env}</Tag>
			),
		},
		{
			title: '实验名称',
			key: 'exp_name',
			ellipsis: true,
			render: (_, record) => record.config.exp_name,
		},
		{
			title: '状态',
			dataIndex: 'status',
			key: 'status',
			width: 120,
			render: getStatusTag,
		},
		{
			title: '创建时间',
			dataIndex: 'created_at',
			key: 'created_at',
			width: 180,
			render: (time) => new Date(time).toLocaleString(),
		},
		{
			title: '训练步数',
			key: 'steps',
			width: 120,
			render: (_, record) => (
				<Tooltip title="总训练步数">
					{record.config.num_env_steps.toLocaleString()}
				</Tooltip>
			),
		},
		{
			title: '操作',
			key: 'actions',
			width: 160,
			render: (_, record) => (
				<Space>
					<Tooltip title="查看日志">
						<Button
							type="text"
							size="small"
							icon={<FileTextOutlined />}
							onClick={() => handleViewLogs(record.task_id)}
						/>
					</Tooltip>
					{record.status === 'running' && (
						<Tooltip title="停止任务">
							<Popconfirm
								title="确定要停止这个任务吗？"
								onConfirm={() => handleStop(record.task_id)}
							>
								<Button
									type="text"
									size="small"
									danger
									icon={<PauseCircleOutlined />}
								/>
							</Popconfirm>
						</Tooltip>
					)}
					{record.status !== 'running' && (
						<Tooltip title="删除任务">
							<Popconfirm
								title="确定要删除这个任务吗？"
								onConfirm={() => handleDelete(record.task_id)}
							>
								<Button
									type="text"
									size="small"
									danger
									icon={<DeleteOutlined />}
								/>
							</Popconfirm>
						</Tooltip>
					)}
				</Space>
			),
		},
	];

	// 统计数据
	const runningCount = tasks.filter((t) => t.status === 'running').length;
	const completedCount = tasks.filter((t) => t.status === 'completed').length;
	const failedCount = tasks.filter((t) => t.status === 'failed').length;

	return (
		<div>
			{/* 统计卡片 */}
			<Row gutter={16} style={{ marginBottom: 16 }}>
				<Col span={6}>
					<Card>
						<Statistic
							title="总任务数"
							value={tasks.length}
							prefix={<ExperimentOutlined />}
						/>
					</Card>
				</Col>
				<Col span={6}>
					<Card>
						<Statistic
							title="运行中"
							value={runningCount}
							prefix={<SyncOutlined spin={runningCount > 0} />}
							valueStyle={{ color: runningCount > 0 ? '#1890ff' : undefined }}
						/>
					</Card>
				</Col>
				<Col span={6}>
					<Card>
						<Statistic
							title="已完成"
							value={completedCount}
							prefix={<CheckCircleOutlined />}
							valueStyle={{ color: '#52c41a' }}
						/>
					</Card>
				</Col>
				<Col span={6}>
					<Card>
						<Statistic
							title="失败"
							value={failedCount}
							prefix={<CloseCircleOutlined />}
							valueStyle={{ color: failedCount > 0 ? '#ff4d4f' : undefined }}
						/>
					</Card>
				</Col>
			</Row>

			{/* 任务列表 */}
			<Card
				title={
					<Space>
						<ExperimentOutlined />
						<span>训练任务</span>
					</Space>
				}
				extra={
					<Button
						icon={<ReloadOutlined />}
						loading={loading}
						onClick={handleRefresh}
					>
						刷新
					</Button>
				}
			>
				<Table
					columns={columns}
					dataSource={tasks}
					rowKey="task_id"
					loading={loading}
					locale={{
						emptyText: (
							<Empty
								image={Empty.PRESENTED_IMAGE_SIMPLE}
								description="暂无训练任务"
							>
								<Text type="secondary">
									前往「训练配置」页面创建新任务
								</Text>
							</Empty>
						),
					}}
					pagination={{
						pageSize: 10,
						showSizeChanger: true,
						showQuickJumper: true,
						showTotal: (total) => `共 ${total} 个任务`,
					}}
				/>
			</Card>

			{/* 日志弹窗 */}
			<Modal
				title={
					<Space>
						<FileTextOutlined />
						<span>任务日志 - {logModal.taskId}</span>
					</Space>
				}
				open={logModal.visible}
				onCancel={() => setLogModal({ visible: false, taskId: '', logs: '' })}
				width={900}
				footer={[
					<Button key="refresh" icon={<ReloadOutlined />} loading={logLoading} onClick={refreshLogs}>
						刷新日志
					</Button>,
					<Button key="close" type="primary" onClick={() => setLogModal({ visible: false, taskId: '', logs: '' })}>
						关闭
					</Button>,
				]}
			>
				<div className="log-viewer" style={{ color: '#e6e6e6' }}>
					{logLoading ? '加载中...' : logModal.logs}
				</div>
			</Modal>
		</div>
	);
}
