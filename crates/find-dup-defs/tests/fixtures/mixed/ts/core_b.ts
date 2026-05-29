export const MAX_RETRIES = 5;
export const DEFAULT_TIMEOUT = 30;

export type UserIds = number[];

export interface Repo {
    get(id: number): number;
    set(id: number, value: number): void;
    has(id: number): boolean;
}

export function computeScore(values: number[], weight: number): number {
    let total = 0;
    for (const v of values) {
        total += v * weight;
    }
    return total / values.length;
}

export function plusNumbers(x: number, y: number): number {
    const result = x + y;
    return result * 2;
}

export class Repository {
    fetchItem(itemId: number): number {
        const record = this.store.get(itemId);
        if (record === undefined) {
            throw new Error("missing");
        }
        return record;
    }
}
