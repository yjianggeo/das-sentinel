import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { api } from '../api';
import WindowPrediction, { WindowActiveModel } from './WindowPrediction';

vi.mock('../api', () => ({ api: { get: vi.fn(), post: vi.fn() } }));
const binding = { windowId: 'w1', analysisId: 'a1', inputSha256: 'hash' };
const result = { status: 'succeeded', window_id: 'w1', analysis_id: 'a1', input_sha256: 'hash',
  label: 'present', score: .1, threshold: .0382, distance_start_m: 300, model_name: '上一轮 300 m CNN' };
beforeEach(() => vi.clearAllMocks());

it('shows the model prediction without creating or selecting a human label', async () => {
  vi.mocked(api.get).mockResolvedValue(result);
  render(<WindowPrediction {...binding} />);
  expect(await screen.findByText('机器判断：地震类（含爆破）')).toBeVisible();
  expect(screen.getByText(/输入 300 m 至末道/)).toBeVisible();
  expect(screen.queryByRole('radio')).not.toBeInTheDocument();
  expect(api.post).not.toHaveBeenCalled();
  expect(api.get).toHaveBeenCalledWith('/api/v1/event-windows/w1/prediction',
    { analysis_id: 'a1', input_sha256: 'hash' }, expect.any(AbortSignal));
});

it('retries failed inference explicitly, without calling it noise', async () => {
  vi.mocked(api.get).mockResolvedValue({ status: 'failed' });
  vi.mocked(api.post).mockResolvedValue({ ...result, label: 'absent', score: .01 });
  render(<WindowPrediction {...binding} />);
  fireEvent.click(await screen.findByRole('button', { name: '重试识别' }));
  expect(await screen.findByText('机器判断：噪声类（含车辆）')).toBeVisible();
  expect(api.post).toHaveBeenCalledWith('/api/v1/event-windows/w1/prediction',
    { analysis_id: 'a1', input_sha256: 'hash' }, expect.any(AbortSignal));
});

it('ignores a delayed prediction after switching windows', async () => {
  let resolve!: (value: unknown) => void;
  vi.mocked(api.get).mockImplementationOnce(() => new Promise(r => { resolve = r; }))
    .mockResolvedValue({ status: 'inactive' });
  const { rerender } = render(<WindowPrediction {...binding} />);
  rerender(<WindowPrediction {...binding} windowId='w2' />);
  await screen.findByText('本站点尚未启用整窗模型');
  resolve(result);
  await waitFor(() => expect(screen.queryByText(/机器判断：地震类/)).not.toBeInTheDocument());
});

it('shows stale input separately and exposes the active model on the models page', async () => {
  vi.mocked(api.get).mockResolvedValueOnce({ status: 'stale' });
  const { unmount } = render(<WindowPrediction {...binding} />);
  expect(await screen.findByText(/窗口输入已更新/)).toBeVisible();
  unmount();
  vi.mocked(api.get).mockResolvedValue({ name: '上一轮 300 m CNN', distance_start_m: 300,
    threshold: .03820185, selected_epoch: 7 });
  render(<WindowActiveModel siteId='site' />);
  expect(await screen.findByText('上一轮 300 m CNN')).toBeVisible();
  expect(screen.getByText(/已上线/)).toBeVisible();
});
