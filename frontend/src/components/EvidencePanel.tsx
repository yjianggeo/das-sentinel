import type { EventDetail } from "../api";

const DETECTOR_NAMES: Record<string, string> = {
  stalta: "STA/LTA 多道符合",
  energy: "多频带能量统计",
  coherence: "去偏相干性",
  trajectory: "车辆轨迹",
  template: "模板匹配",
};

const METHOD_NAMES: Record<string, string> = {
  rules: "规则判断",
  model: "模型判断",
  fusion: "融合判断",
};

const LABEL_NAMES: Record<string, string> = {
  earthquake: "地震",
  blast: "爆破",
  traffic: "车辆",
  noise: "噪声",
  unknown: "待定",
};

export default function EvidencePanel({ event }: { event: EventDetail }) {
  const latestMachine = event.machine_history[event.machine_history.length - 1];
  return (
    <div className="evidence-panel">
      <section>
        <h3>检测证据</h3>
        <ul>
          {event.evidence.map((item, index) => (
            <li key={`${item.detector}-${index}`}>
              {DETECTOR_NAMES[item.detector] ?? item.detector}：
              得分 {item.score.toFixed(2)}
            </li>
          ))}
          {event.evidence.length === 0 && <li>无检测证据</li>}
        </ul>
      </section>
      <section>
        <h3>{latestMachine ? METHOD_NAMES[latestMachine.method] ?? latestMachine.method : "规则判断"}</h3>
        {latestMachine ? (
          <>
            <p>
              机器标签：
              {LABEL_NAMES[latestMachine.label] ?? latestMachine.label}（置信{" "}
              {latestMachine.decision_score.toFixed(2)}）
            </p>
            <ul>
              {latestMachine.reasons.map((reason, index) => (
                <li key={index}>{reason}</li>
              ))}
            </ul>
          </>
        ) : (
          <p>暂无机器分类</p>
        )}
      </section>
      <section>
        <h3>事件特征</h3>
        <table className="data">
          <tbody>
            {Object.entries(event.features).map(([name, value]) => (
              <tr key={name}>
                <td>{name}</td>
                <td>{typeof value === "number" ? value.toPrecision(4) : String(value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
