export interface User {
  user_id: string;
  username: string;
  display_name: string;
  role: string;
}

interface AuthState {
  token: string | null;
  user: User | null;
  login: (username: string, password: string) => Promise<User>;
  logout: () => void;
  loadToken: () => string | null;
}

const TOKEN_KEY = 'opsmind_token';
const USER_KEY = 'opsmind_user';

function loadFromStorage(): { token: string | null; user: User | null } {
  try {
    const token = localStorage.getItem(TOKEN_KEY);
    const user = JSON.parse(localStorage.getItem(USER_KEY) || 'null');
    return { token, user };
  } catch {
    return { token: null, user: null };
  }
}

export const authStore: AuthState = {
  token: loadFromStorage().token,
  user: loadFromStorage().user,

  async login(username: string, password: string): Promise<User> {
    const res = await fetch('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Login failed');
    }
    const data = await res.json();
    localStorage.setItem(TOKEN_KEY, data.token);
    localStorage.setItem(USER_KEY, JSON.stringify(data.user));
    authStore.token = data.token;
    authStore.user = data.user;
    return data.user;
  },

  logout() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    localStorage.removeItem('opsmind_session_id');
    authStore.token = null;
    authStore.user = null;
  },

  loadToken(): string | null {
    if (!authStore.token) {
      authStore.token = localStorage.getItem(TOKEN_KEY);
    }
    return authStore.token;
  },
};
