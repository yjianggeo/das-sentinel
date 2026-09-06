import { useCallback, useEffect, useRef, useState } from "react";
import { api, type ScanDirectoryListing, type Site } from "../api";

type Props = {
  site: Site;
  onCancel: () => void;
  onSelect: (path: string) => void;
};

function isAbortError(cause: unknown): boolean {
  return cause instanceof Error && cause.name === "AbortError";
}

export default function ScanDirectoryDialog({ site, onCancel, onSelect }: Props) {
  const [listing, setListing] = useState<ScanDirectoryListing | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const activeController = useRef<AbortController | null>(null);

  const load = useCallback((path: string) => {
    activeController.current?.abort();
    const controller = new AbortController();
    activeController.current = controller;
    setListing(null);
    setLoading(true);
    setError("");
    api
      .get<ScanDirectoryListing>(
        `/api/v1/sites/${site.id}/scan-directories`,
        { path },
        controller.signal,
      )
      .then((response) => {
        if (activeController.current === controller && !controller.signal.aborted) {
          setListing(response);
        }
      })
      .catch((cause: unknown) => {
        if (
          activeController.current !== controller
          || controller.signal.aborted
          || isAbortError(cause)
        ) {
          return;
        }
        setError("无法加载目录");
      })
      .finally(() => {
        if (activeController.current === controller && !controller.signal.aborted) {
          setLoading(false);
        }
      });
  }, [site.id]);

  useEffect(() => {
    load("");
    return () => activeController.current?.abort();
  }, [load]);

  const breadcrumb = listing
    ? listing.path
      ? `${listing.site_name} / ${listing.path}`
      : listing.site_name
    : site.name;

  return (
    <div className="dialog-backdrop">
      <div
        className="dialog scan-directory-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={`选择 ${site.name} 扫描目录`}
      >
        <h2>选择扫描目录</h2>
        <p className="scan-directory-breadcrumb">{breadcrumb}</p>
        {loading && <p className="loading">正在加载目录…</p>}
        {error && <p className="error" role="alert">{error}</p>}
        {listing !== null && listing.parent_path !== null && (
          <button
            className="subtle"
            type="button"
            onClick={() => {
              const parentPath = listing?.parent_path;
              if (parentPath !== null && parentPath !== undefined) load(parentPath);
            }}
          >
            返回上级目录
          </button>
        )}
        {listing && (
          <div className="scan-directory-list" aria-label="目录列表">
            {listing.directories.map((entry) => (
              <button
                key={entry.path}
                className="subtle"
                type="button"
                onClick={() => load(entry.path)}
              >
                {entry.name}
              </button>
            ))}
          </div>
        )}
        <div className="scan-directory-actions">
          <button className="subtle" type="button" onClick={onCancel}>
            取消
          </button>
          <button
            className="primary"
            type="button"
            disabled={!listing?.path}
            onClick={() => {
              if (listing?.path) onSelect(listing.path);
            }}
          >
            选择当前目录
          </button>
        </div>
      </div>
    </div>
  );
}
