from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from das_sentinel.classifiers.group_folds import make_group_folds
from das_sentinel.classifiers.similarity import FeatureVectorizer
from das_sentinel.classifiers.supervised import (
    _SEED,
    TrainingGateError,
    _canonical_rule_profile,
    _finite_number,
)
from das_sentinel.persistence.models import (
    Event,
    ProcessingRun,
    Review,
    Site,
    SourceFile,
)
from das_sentinel.services.training import (
    LABEL_GATES,
    SUPERVISED_LABELS,
    TrainingDataError,
    TrainingRow,
    TrainingService,
)

_LABELS = tuple(str(label) for label in SUPERVISED_LABELS)
_LABEL_NAMES = {
    "earthquake": "地震",
    "blast": "爆破",
    "traffic": "交通",
    "noise": "噪声",
}
_FOLDS = 5
_GATE_MESSAGES = {
    "site has no current processing profile": "站点尚未设置当前处理配置。",
    "rule profile version must be non-empty": "当前处理配置缺少版本号。",
    "rule profile must be a mapping": "当前处理配置格式无效，应为字段对象。",
    "rule profile must be canonical JSON data": "当前处理配置含无效值，无法用于训练。",
    "reviewed events must share one feature schema": "已复核事件的特征字段不一致，需使用相同特征结构。",
    "training features do not match the approved schema": "训练特征字段不符合允许的数值特征结构。",
    "training features must be finite numeric values": "训练特征必须为有限数值，当前数据含无效数值或非数值。",
}


class TrainingReadinessService:
    """Inspect the existing training structure without fitting or writing artifacts."""

    def __init__(self, session: Session, artifact_root: Path) -> None:
        self.session = session
        self.training = TrainingService(session, artifact_root)

    def get(self, site_id: str) -> dict[str, Any]:
        if self.session.get(Site, site_id) is None:
            raise LookupError(f"site not found: {site_id}")

        # Keep the production review tie-breaker, including unsupported latest labels.
        latest_label = (
            select(Review.label)
            .where(Review.event_id == Event.id)
            .order_by(Review.created_at.desc(), Review.id.desc())
            .limit(1)
            .correlate(Event)
            .scalar_subquery()
        )
        events = self.session.execute(
            select(
                Event.occurred_at,
                ProcessingRun.pipeline_version,
                ProcessingRun.profile_version,
                latest_label.label("review_label"),
            )
            .join(ProcessingRun, ProcessingRun.id == Event.processing_run_id)
            .join(SourceFile, SourceFile.id == ProcessingRun.source_file_id)
            .where(SourceFile.site_id == site_id)
            .order_by(Event.id)
        ).all()
        reviewed = [event for event in events if event.review_label is not None]
        counts = Counter(event.review_label for event in reviewed)
        days = {
            label: {event.occurred_at.date() for event in reviewed
                    if event.review_label == label}
            for label in _LABELS
        }
        per_class = LABEL_GATES["per_class_reviewed"]
        classes = [
            {
                "label": label,
                "reviewed_count": counts[label],
                "utc_day_count": len(days[label]),
                "missing_count": max(0, per_class - counts[label]),
                "minimum_days_missing": max(0, _FOLDS - len(days[label])),
            }
            for label in _LABELS
        ]
        unsupported = sum(count for label, count in counts.items() if label not in _LABELS)
        blockers = []
        warnings = [
            "准备度只检查训练数据的结构门槛，通过不代表模型精度或发布指标达标。",
            "UTC 事件日期用于分组，不代表独立物理事件；每类覆盖五天只是必要条件，仍需检查每折类别覆盖。",
            "低频候选需要单独人工复核，不会自动计入四类训练标签。",
        ]
        if unsupported:
            blockers.append(
                f"有 {unsupported} 条事件的最新人工标签不支持；仅接受地震、爆破、交通和噪声。"
            )
        if not reviewed:
            blockers.append("尚无已复核事件可用于训练。")
        for item in classes:
            label_name = _LABEL_NAMES[item["label"]]
            if item["missing_count"]:
                blockers.append(
                    f"{label_name}还需复核 {item['missing_count']} 条，达到每类 {per_class} 条的门槛。"
                )
            if item["minimum_days_missing"]:
                blockers.append(
                    f"{label_name}至少还需覆盖 {item['minimum_days_missing']} 个 UTC 事件日期，"
                    "满足五折覆盖的必要条件。"
                )

        try:
            profile = self.training._current_profile(site_id)
            _canonical_rule_profile(dict(profile.payload), str(profile.version))
        except (TrainingDataError, TrainingGateError) as error:
            blockers.append(_GATE_MESSAGES.get(str(error), "当前处理配置不满足训练要求。"))
        except (ValueError, TypeError):
            blockers.append("当前处理配置无效，无法检查训练准备度。")

        rows = []
        features = None
        try:
            # This supplies precisely the row order and UTC dates used by train().
            rows = self.training._reviewed_rows(site_id)
            if rows:
                features = _training_features(rows)
        except (json.JSONDecodeError, AttributeError, TypeError):
            blockers.append("已复核事件的特征数据格式无效，应为有效 JSON 字段对象。")
        except TrainingGateError as error:
            blockers.append(_GATE_MESSAGES.get(str(error), "训练特征不满足结构检查要求。"))

        groups = [row.group for row in rows]
        if rows and len(set(groups)) < _FOLDS:
            blockers.append("训练至少需要五个不同的 UTC 事件日期。")
        folds = []
        if (
            features is not None
            and not unsupported
            and all(counts[label] >= per_class for label in _LABELS)
            and len(set(groups)) >= _FOLDS
        ):
            labels = np.asarray([row.label for row in rows])
            try:
                splits = make_group_folds(labels, groups, n_splits=_FOLDS, seed=_SEED)
            except ValueError:
                splits = []
                blockers.append("未能在限定搜索内找到每折均覆盖四类标签、且日期组互斥的五折划分。")
            for number, (train_index, test_index) in enumerate(
                splits, start=1
            ):
                train_counts = Counter(labels[train_index])
                test_counts = Counter(labels[test_index])
                missing_train = [label for label in _LABELS if not train_counts[label]]
                missing_test = [label for label in _LABELS if not test_counts[label]]
                folds.append({
                    "fold": number,
                    "train_counts": {label: train_counts[label] for label in _LABELS},
                    "test_counts": {label: test_counts[label] for label in _LABELS},
                    "missing_train_labels": missing_train,
                    "missing_test_labels": missing_test,
                })
                if missing_train:
                    blockers.append(
                        f"第 {number} 折训练集缺少："
                        + "、".join(_LABEL_NAMES[label] for label in missing_train) + "。"
                    )
                if missing_test:
                    blockers.append(
                        f"第 {number} 折验证集缺少："
                        + "、".join(_LABEL_NAMES[label] for label in missing_test) + "。"
                    )
                if set(np.asarray(groups)[train_index]) & set(np.asarray(groups)[test_index]):
                    blockers.append(
                        f"第 {number} 折的训练集与验证集存在日期分组重叠。"
                    )

        pipeline_counts = dict(sorted(Counter(
            event.pipeline_version for event in reviewed
        ).items()))
        profile_counts = dict(sorted(Counter(
            str(event.profile_version) for event in reviewed
        ).items()))
        if len(pipeline_counts) > 1:
            warnings.append(
                "已复核事件混有不同流水线版本；特征字段相同不代表信号处理方式相同。"
            )
        if len(profile_counts) > 1:
            warnings.append(
                "已复核事件混有不同处理配置版本；训练时的规则基线使用当前配置。"
            )
        models = self.training.list_for_site(site_id)
        active_model = self.training.active_model(site_id)
        return {
            "site_id": site_id,
            "reviewed_events": len(reviewed),
            "unreviewed_events": len(events) - len(reviewed),
            "unsupported_label_count": unsupported,
            "classes": classes,
            "ready": not blockers,
            "blockers": blockers,
            "warnings": warnings,
            "folds": folds,
            "pipeline_counts": pipeline_counts,
            "profile_counts": profile_counts,
            "model_count": len(models),
            "active_model_version": active_model.version if active_model else None,
        }


def _training_features(rows: list[TrainingRow]) -> np.ndarray:
    """Apply train_model's feature gates without invoking any estimator."""
    feature_sets = [
        {name for name in row.features if name not in {"site_id", "cable_type", "device_config"}}
        for row in rows
    ]
    if any(names != feature_sets[0] for names in feature_sets[1:]):
        raise TrainingGateError("reviewed events must share one feature schema")
    feature_names = tuple(sorted(feature_sets[0]))
    try:
        vectorizer = FeatureVectorizer(feature_names)
    except ValueError as error:
        raise TrainingGateError("training features do not match the approved schema") from error
    for row in rows:
        for name in feature_names:
            try:
                _finite_number(row.features.get(name))
            except OverflowError as error:
                raise TrainingGateError(
                    "training features must be finite numeric values"
                ) from error
    try:
        return vectorizer.transform(row.features for row in rows)
    except (TypeError, ValueError) as error:
        raise TrainingGateError("training features do not match the approved schema") from error
