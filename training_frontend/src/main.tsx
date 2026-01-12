import React from 'react';
import ReactDOM from 'react-dom/client';
import { ConfigProvider, theme } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import App from './App';
import './styles/global.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
	<React.StrictMode>
		<ConfigProvider
			locale={zhCN}
			theme={{
				algorithm: theme.darkAlgorithm,
				token: {
					colorPrimary: '#1890ff',
					colorSuccess: '#52c41a',
					colorWarning: '#faad14',
					colorError: '#ff4d4f',
					borderRadius: 8,
				},
			}}
		>
			<App />
		</ConfigProvider>
	</React.StrictMode>
);
