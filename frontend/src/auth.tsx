import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { api, setCsrfToken, type SessionInfo } from "./api";

interface AuthState {
  authenticated: boolean;
  loading: boolean;
  login: (password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [authenticated, setAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    api
      .get<SessionInfo>("/api/v1/session")
      .then((session) => {
        if (cancelled) return;
        setCsrfToken(session.csrf_token);
        setAuthenticated(true);
      })
      .catch(() => {
        if (!cancelled) setAuthenticated(false);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (password: string) => {
    await api.post<void>("/api/v1/session", { password });
    const session = await api.get<SessionInfo>("/api/v1/session");
    setCsrfToken(session.csrf_token);
    setAuthenticated(true);
  }, []);

  const logout = useCallback(async () => {
    try {
      await api.delete<void>("/api/v1/session");
    } finally {
      setCsrfToken("");
      setAuthenticated(false);
    }
  }, []);

  const value = useMemo(
    () => ({ authenticated, loading, login, logout }),
    [authenticated, loading, login, logout],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const context = useContext(AuthContext);
  if (context === null) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return context;
}
