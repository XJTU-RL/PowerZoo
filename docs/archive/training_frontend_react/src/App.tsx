import { useEffect, useState } from 'react';
import {
	Layout,
	Menu,
	Typography,
	Spin,
	message,
} from 'antd';
import {
	DashboardOutlined,
	SettingOutlined,
	ExperimentOutlined,
	ThunderboltOutlined,
} from '@ant-design/icons';
import { useConfigStore } from '@/stores/configStore';
import ConfigPage from '@/pages/ConfigPage';
import TasksPage from '@/pages/TasksPage';

const { Header, Content, Sider } = Layout;
const { Title } = Typography;

type PageKey = 'config' | 'tasks';

function App() {
	const [currentPage, setCurrentPage] = useState<PageKey>('config');
	const { loading, error, loadInitialData } = useConfigStore();

	useEffect(() => {
		loadInitialData();
	}, [loadInitialData]);

	useEffect(() => {
		if (error) {
			message.error(`加载失败: ${error}`);
		}
	}, [error]);

	const menuItems = [
		{
			key: 'config',
			icon: <SettingOutlined />,
			label: '训练配置',
		},
		{
			key: 'tasks',
			icon: <ExperimentOutlined />,
			label: '任务管理',
		},
	];

	const renderPage = () => {
		switch (currentPage) {
			case 'config':
				return <ConfigPage />;
			case 'tasks':
				return <TasksPage />;
			default:
				return <ConfigPage />;
		}
	};

	if (loading) {
		return (
			<div style={{
				height: '100vh',
				display: 'flex',
				justifyContent: 'center',
				alignItems: 'center',
				background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
			}}>
				<Spin size="large" tip="正在加载..." />
			</div>
		);
	}

	return (
		<Layout style={{ minHeight: '100vh' }}>
			<Header style={{
				display: 'flex',
				alignItems: 'center',
				padding: '0 24px',
				position: 'fixed',
				width: '100%',
				zIndex: 100,
			}}>
				<div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
					<ThunderboltOutlined style={{ fontSize: 28, color: '#faad14' }} />
					<Title level={4} style={{ margin: 0, color: '#fff' }}>
						PowerZoo Training Console
					</Title>
				</div>
				<div style={{ flex: 1 }} />
				<div style={{ color: 'rgba(255,255,255,0.65)', fontSize: 12 }}>
					Multi-Agent RL Training Platform
				</div>
			</Header>

			<Layout style={{ marginTop: 64 }}>
				<Sider
					width={200}
					style={{
						background: 'rgba(0, 21, 41, 0.9)',
						backdropFilter: 'blur(10px)',
						position: 'fixed',
						height: 'calc(100vh - 64px)',
						left: 0,
						top: 64,
						borderRight: '1px solid rgba(255,255,255,0.1)',
					}}
				>
					<Menu
						mode="inline"
						selectedKeys={[currentPage]}
						style={{ height: '100%', borderRight: 0, background: 'transparent' }}
						items={menuItems}
						onClick={({ key }) => setCurrentPage(key as PageKey)}
					/>
				</Sider>

				<Layout style={{ marginLeft: 200, padding: '24px' }}>
					<Content
						style={{
							padding: 0,
							margin: 0,
							minHeight: 'calc(100vh - 112px)',
						}}
					>
						<div className="fade-in">
							{renderPage()}
						</div>
					</Content>
				</Layout>
			</Layout>
		</Layout>
	);
}

export default App;
