import type { ModelInfo } from "../api";

const LABEL_ORDER = ["earthquake", "blast", "traffic", "noise"] as const;
const LABEL_NAMES: Record<string, string> = {
  earthquake: "地震",
  blast: "爆破",
  traffic: "车辆",
  noise: "噪声",
};

export default function ValidationReport({ model }: { model: ModelInfo }) {
  const metrics = model.metrics;
  const matrix = metrics.confusion_matrix ?? [];
  return (
    <div className="validation-report">
      <h3>验证报告（v{model.version}）</h3>
      <p>
        宏平均 F1：{(metrics.macro_f1 ?? 0).toFixed(3)}　规则基线 F1：
        {(metrics.rule_baseline_macro_f1 ?? 0).toFixed(3)}
      </p>
      <table className="data" aria-label="分类指标">
        <thead>
          <tr>
            <th>类别</th>
            <th>复核样本数</th>
            <th>召回率</th>
            <th>精确率</th>
          </tr>
        </thead>
        <tbody>
          {LABEL_ORDER.map((label) => (
            <tr key={label}>
              <td>{LABEL_NAMES[label]}</td>
              <td>{metrics.reviewed_count_by_label?.[label] ?? 0}</td>
              <td>{(metrics.recall_by_label?.[label] ?? 0).toFixed(3)}</td>
              <td>{(metrics.precision_by_label?.[label] ?? 0).toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {matrix.length === 4 && (
        <table className="data" aria-label="混淆矩阵">
          <thead>
            <tr>
              <th>真实 \ 预测</th>
              {LABEL_ORDER.map((label) => (
                <th key={label}>{LABEL_NAMES[label]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rowIndex) => (
              <tr key={LABEL_ORDER[rowIndex]}>
                <td>{LABEL_NAMES[LABEL_ORDER[rowIndex]]}</td>
                {row.map((value, columnIndex) => (
                  <td key={columnIndex}>{value}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
