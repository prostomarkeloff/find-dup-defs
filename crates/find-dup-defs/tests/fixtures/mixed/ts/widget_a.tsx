export function renderBadge(label: string, count: number) {
    const text = `${label}: ${count}`;
    const cls = count > 0 ? "badge active" : "badge";
    return <span className={cls}>{text}</span>;
}
