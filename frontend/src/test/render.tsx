import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import App from "../App";
import { AuthProvider } from "../auth";

export function renderApp(route: string) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <AuthProvider>
        <App />
      </AuthProvider>
    </MemoryRouter>,
  );
}

export function renderPlain(element: ReactElement, route = "/") {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <AuthProvider>{element}</AuthProvider>
    </MemoryRouter>,
  );
}
