import { jsx as _jsx } from "react/jsx-runtime";
import { useState } from 'react';
import LoginPage from './components/LoginPage';
import Chat from './components/Chat';
export default function App() {
    const [loggedIn, setLoggedIn] = useState(!!localStorage.getItem('opsmind_token'));
    if (!loggedIn) {
        return _jsx(LoginPage, { onLogin: () => setLoggedIn(true) });
    }
    return _jsx(Chat, { onLogout: () => { localStorage.removeItem('opsmind_token'); localStorage.removeItem('opsmind_user'); setLoggedIn(false); } });
}
