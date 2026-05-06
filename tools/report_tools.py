"""Report tools — generate Markdown model evaluation report."""

from datetime import datetime


def generate_markdown_report(
    project_name: str,
    data_summary: dict | None = None,
    quality_summary: dict | None = None,
    cleaning_summary: dict | None = None,
    model_metrics: dict | None = None,
    threshold_table: list[dict] | None = None,
    feature_importance: list[dict] | None = None,
    risk_warnings: list[str] | None = None,
    recommendations: list[str] | None = None,
) -> str:
    """Generate a comprehensive Markdown model evaluation report.

    Args:
        project_name: Project name.
        data_summary: Data overview (rows, cols, label distribution).
        quality_summary: Quality analysis summary.
        cleaning_summary: Cleaning operations summary.
        model_metrics: Model evaluation metrics (AUC, KS, etc.).
        threshold_table: Threshold strategy table rows.
        feature_importance: Top feature importance list.
        risk_warnings: Risk warnings from RiskGuard.
        recommendations: Optimization recommendations.

    Returns:
        Markdown string.
    """
    lines = []
    lines.append(f"# 模型评估报告 — {project_name}")
    lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Data overview
    lines.append("## 1. 数据概况\n")
    if data_summary:
        lines.append(f"- 样本量: {data_summary.get('n_rows', 'N/A')}")
        lines.append(f"- 特征数: {data_summary.get('n_features', 'N/A')}")
        lines.append(f"- 正样本率: {data_summary.get('positive_rate', 'N/A')}")
        lines.append(f"- 训练集大小: {data_summary.get('train_size', 'N/A')}")
        lines.append(f"- 测试集大小: {data_summary.get('test_size', 'N/A')}")
    else:
        lines.append("暂无数据概况信息。")
    lines.append("")

    # 2. Quality analysis
    lines.append("## 2. 数据质量分析摘要\n")
    if quality_summary:
        lines.append(f"- 质量评分: {quality_summary.get('overall_quality_score', 'N/A')}/100")
        lines.append(f"- 重复行数: {quality_summary.get('duplicate_rows', 0)}")
        lines.append(f"- 高缺失列数: {quality_summary.get('high_missing_count', 0)}")
        lines.append(f"- 异常值列数: {quality_summary.get('outlier_count', 0)}")
    else:
        lines.append("暂无质量分析信息。")
    lines.append("")

    # 3. Cleaning record
    lines.append("## 3. 数据清洗记录\n")
    if cleaning_summary:
        lines.append(f"- 清洗前: {cleaning_summary.get('before_shape', 'N/A')}")
        lines.append(f"- 清洗后: {cleaning_summary.get('after_shape', 'N/A')}")
        lines.append(f"- 执行步骤数: {cleaning_summary.get('n_steps', 0)}")
        steps = cleaning_summary.get("steps", [])
        if steps:
            lines.append("\n| 步骤 | 动作 | 影响行数 |")
            lines.append("|------|------|----------|")
            for s in steps:
                lines.append(f"| {s.get('priority', '-')} | {s.get('action', '-')} | {s.get('rows_affected', '-')} |")
    else:
        lines.append("暂无清洗记录。")
    lines.append("")

    # 4. Model metrics
    lines.append("## 4. 模型评估指标\n")
    if model_metrics:
        lines.append("| 指标 | 值 |")
        lines.append("|------|------|")
        lines.append(f"| AUC | {model_metrics.get('auc', 'N/A')} |")
        lines.append(f"| KS | {model_metrics.get('ks', 'N/A')} |")
        lines.append(f"| Accuracy | {model_metrics.get('accuracy', 'N/A')} |")
        lines.append(f"| Precision | {model_metrics.get('precision', 'N/A')} |")
        lines.append(f"| Recall | {model_metrics.get('recall', 'N/A')} |")
        lines.append(f"| F1 | {model_metrics.get('f1', 'N/A')} |")

        cm = model_metrics.get("confusion_matrix")
        if cm:
            lines.append(f"\n**混淆矩阵:** TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
    else:
        lines.append("暂无模型指标。")
    lines.append("")

    # 5. Threshold strategy
    lines.append("## 5. 阈值策略分析\n")
    if threshold_table:
        lines.append("| 阈值 | 通过率 | 拒绝率 | 通过坏账率 | 坏样本捕获率 |")
        lines.append("|------|--------|--------|------------|--------------|")
        for row in threshold_table[:10]:  # Top 10 rows
            lines.append(
                f"| {row.get('threshold', '-')} "
                f"| {row.get('pass_rate', '-')} "
                f"| {row.get('reject_rate', '-')} "
                f"| {row.get('pass_bad_rate', '-')} "
                f"| {row.get('capture_rate', '-')} |"
            )
    else:
        lines.append("暂无阈值策略分析。")
    lines.append("")

    # 6. Feature importance
    lines.append("## 6. 特征重要性 (Top 20)\n")
    if feature_importance:
        lines.append("| 排名 | 特征名 | 重要性 |")
        lines.append("|------|--------|--------|")
        for i, feat in enumerate(feature_importance[:20], 1):
            lines.append(
                f"| {i} | {feat.get('feature_name', '-')} "
                f"| {feat.get('importance_score', '-')} |"
            )
    else:
        lines.append("暂无特征重要性信息。")
    lines.append("")

    # 7. Risk warnings
    lines.append("## 7. 风险提示\n")
    if risk_warnings:
        for w in risk_warnings:
            lines.append(f"- ⚠️ {w}")
    else:
        lines.append("无风险提示。")
    lines.append("")

    # 8. Recommendations
    lines.append("## 8. 优化建议\n")
    if recommendations:
        for r in recommendations:
            lines.append(f"- {r}")
    else:
        lines.append(_generate_default_recommendations(model_metrics))
    lines.append("")

    return "\n".join(lines)


def _generate_default_recommendations(metrics: dict | None) -> str:
    """Generate default recommendations based on metrics."""
    if not metrics:
        return "暂无优化建议。"

    recs = []
    auc = metrics.get("auc", 0)
    ks = metrics.get("ks", 0)

    if auc < 0.7:
        recs.append("- AUC 偏低，建议增加特征或尝试更复杂的模型")
    if ks < 0.2:
        recs.append("- KS 偏低，模型区分度不足，建议检查特征质量")
    if auc > 0.95:
        recs.append("- AUC 异常高，请检查是否存在数据泄露")

    if not recs:
        recs.append("- 模型表现良好，可考虑进一步调优超参数")

    return "\n".join(recs)
