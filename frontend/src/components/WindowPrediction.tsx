import { useEffect, useState } from 'react';
import { api } from '../api';

interface Prediction {
  status: 'pending' | 'succeeded' | 'failed' | 'inactive' | 'unavailable' | 'stale';
  window_id?: string;
  analysis_id?: string;
  input_sha256?: string;
  model_name?: string;
  label?: 'present' | 'absent';
  score?: number;
  threshold?: number;
  distance_start_m?: number;
}

export default function WindowPrediction({ windowId, analysisId, inputSha256 }: {
  windowId: string; analysisId: string; inputSha256: string;
}) {
  const [value, setValue] = useState<Prediction | null>(null);
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    setValue(null);
    setError('');
    const path = `/api/v1/event-windows/${windowId}/prediction`;
    const binding = { analysis_id: analysisId, input_sha256: inputSha256 };
    async function load(retry = false) {
      try {
        const next = retry
          ? await api.post<Prediction>(path, binding, controller.signal)
          : await api.get<Prediction>(path, binding, controller.signal);
        if (controller.signal.aborted) return;
        if (next.status === 'succeeded' && (next.window_id !== windowId || next.analysis_id !== analysisId || next.input_sha256 !== inputSha256)) {
          setValue({ status: 'stale' });
          return;
        }
        setValue(next);
        if (next.status === 'pending') timer = setTimeout(() => void load(), 1500);
      } catch {
        if (!controller.signal.aborted) setError('暂时无法获取模型判断');
      }
    }
    void load(attempt > 0);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [windowId, analysisId, inputSha256, attempt]);

  return <section aria-label="整窗模型判断" className="window-prediction">
    <h3>整窗模型判断</h3>
    {error ? <p role="alert">{error}</p> : !value || value.status === 'pending' ? <p role="status">模型识别中…</p> : <>
      {value.status === 'succeeded' && <><strong>机器判断：{value.label === 'present' ? '地震类（含爆破）' : '噪声类（含车辆）'}</strong>
        <p>{value.model_name} · 输入 {value.distance_start_m} m 至末道 · 30 秒 / 100 Hz</p>
        <p>模型得分 {value.score?.toFixed(4)} · 判定阈值 {value.threshold?.toFixed(4)}。仅供复核参考，人工标签由你保存。</p></>}
      {value.status === 'failed' && <p role="alert">模型识别失败，尚无机器判断。</p>}
      {value.status === 'stale' && <p>窗口输入已更新，请刷新窗口后查看模型判断。</p>}
      {value.status === 'unavailable' && <p>当前窗口尚不具备有效的模型输入。</p>}
      {value.status === 'inactive' && <p>本站点尚未启用整窗模型</p>}
    </>}
    {(error || value?.status === 'failed') && <button onClick={() => setAttempt(v => v + 1)}>重试识别</button>}
  </section>;
}

interface ActiveModel { name: string; distance_start_m: number; threshold: number; selected_epoch: number }

export function WindowActiveModel({ siteId }: { siteId: string }) {
  const [value, setValue] = useState<ActiveModel | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    setLoaded(false); setValue(null); setError('');
    if (siteId) void api.get<ActiveModel | null>('/api/v1/window-models/active', { site_id: siteId }, controller.signal)
      .then(model => { if (!controller.signal.aborted) { setValue(model); setLoaded(true); } })
      .catch(() => { if (!controller.signal.aborted) setError('无法加载当前上线模型'); });
    return () => controller.abort();
  }, [siteId]);
  if (!siteId) return null;
  return <section aria-label="当前上线模型"><h2>当前上线模型</h2>
    {error ? <p role="alert">{error}</p> : !loaded ? <p>正在加载…</p> : value ? <>
      <strong>{value.name}</strong><p>已上线 · 输入 {value.distance_start_m} m 至末道 · 30 秒 / 100 Hz · 第 {value.selected_epoch} 轮权重</p>
      <p>地震类（含爆破） / 噪声类（含车辆） · 判定阈值 {value.threshold.toFixed(6)}</p>
      <p>新完成的扫描窗口自动识别，已有窗口打开时识别；机器判断不改写人工标签。</p>
    </> : <p>本站点尚未启用整窗模型</p>}
  </section>;
}
