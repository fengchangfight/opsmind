import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import { authStore } from '../stores/authStore';
export default function LoginPage({ onLogin }) {
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
        }
        catch (e) {
            setError(e.message || 'Login failed');
        }
        finally {
            setLoading(false);
        }
    };
    const handleKeyDown = (e) => {
        if (e.key === 'Enter')
            handleLogin();
    };
    return (_jsx("div", { className: "min-h-screen flex items-center justify-center bg-gray-100", children: _jsxs("div", { className: "bg-white rounded-lg shadow-md p-8 w-96", children: [_jsx("h1", { className: "text-xl font-bold text-center mb-2", children: "OpsMind RAG" }), _jsx("p", { className: "text-sm text-gray-500 text-center mb-6", children: "\u767B\u5F55\u4EE5\u7EE7\u7EED" }), error && (_jsx("div", { className: "bg-red-50 text-red-600 text-sm px-4 py-2 rounded mb-4", children: error })), _jsxs("div", { className: "space-y-4", children: [_jsx("input", { type: "text", placeholder: "\u7528\u6237\u540D", value: username, onChange: (e) => setUsername(e.target.value), onKeyDown: handleKeyDown, className: "w-full border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400" }), _jsx("input", { type: "password", placeholder: "\u5BC6\u7801", value: password, onChange: (e) => setPassword(e.target.value), onKeyDown: handleKeyDown, className: "w-full border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400" }), _jsx("button", { onClick: handleLogin, disabled: loading || !username || !password, className: "w-full py-2 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600 disabled:opacity-50 transition-colors", children: loading ? '登录中...' : '登录' })] }), _jsxs("div", { className: "mt-6 text-xs text-gray-400 text-center space-y-1", children: [_jsxs("p", { children: ["Demo \u7528\u6237\uFF1A", _jsx("code", { children: "alice" }), " / ", _jsx("code", { children: "bob" })] }), _jsxs("p", { children: ["\u5BC6\u7801\uFF1A", _jsx("code", { children: "opsmind123" })] })] })] }) }));
}
