import { useState } from 'react';
import { authStore } from '../stores/authStore';

interface Props {
  onLogin: () => void;
}

export default function LoginPage({ onLogin }: Props) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    setError('');
    setLoading(true);
    try {
      await authStore.login(username, password);
      onLogin();
    } catch (e: any) {
      setError(e.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleLogin();
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="bg-white rounded-lg shadow-md p-8 w-96">
        <h1 className="text-xl font-bold text-center mb-2">OpsMind RAG</h1>
        <p className="text-sm text-gray-500 text-center mb-6">登录以继续</p>

        {error && (
          <div className="bg-red-50 text-red-600 text-sm px-4 py-2 rounded mb-4">{error}</div>
        )}

        <div className="space-y-4">
          <input
            type="text"
            placeholder="用户名"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            onKeyDown={handleKeyDown}
            className="w-full border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
          <input
            type="password"
            placeholder="密码"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={handleKeyDown}
            className="w-full border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
          />

          <button
            onClick={handleLogin}
            disabled={loading || !username || !password}
            className="w-full py-2 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600 disabled:opacity-50 transition-colors"
          >
            {loading ? '登录中...' : '登录'}
          </button>
        </div>

        <div className="mt-6 text-xs text-gray-400 text-center space-y-1">
          <p>Demo 用户：<code>alice</code> / <code>bob</code></p>
          <p>密码：<code>opsmind123</code></p>
        </div>
      </div>
    </div>
  );
}
