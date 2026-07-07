import type { GameState } from "./game_manager.ts";

interface StorageLike {
    setItem(key: string, value: string): void;
    getItem(key: string): string | null;
    removeItem(key: string): void;
}

// In-memory fallback when localStorage is unavailable
const fakeStorage: StorageLike & { _data: Record<string, string> } = {
    _data: {},

    setItem(id, val) {
        this._data[id] = String(val);
    },

    getItem(id) {
        return Object.prototype.hasOwnProperty.call(this._data, id)
            ? this._data[id]
            : null;
    },

    removeItem(id) {
        delete this._data[id];
    },
};

export class LocalStorageManager {
    private bestScoreKey = "bestScore";
    private gameStateKey = "gameState";
    private storage: StorageLike;

    constructor() {
        this.storage = this.localStorageSupported()
            ? window.localStorage
            : fakeStorage;
    }

    private localStorageSupported(): boolean {
        const testKey = "test";

        try {
            const storage = window.localStorage;
            storage.setItem(testKey, "1");
            storage.removeItem(testKey);
            return true;
        } catch {
            return false;
        }
    }

    // Best score getters/setters
    getBestScore(): number {
        return Number(this.storage.getItem(this.bestScoreKey)) || 0;
    }

    setBestScore(score: number): void {
        this.storage.setItem(this.bestScoreKey, String(score));
    }

    // Game state getters/setters and clearing
    getGameState(): GameState | null {
        const stateJSON = this.storage.getItem(this.gameStateKey);
        return stateJSON ? (JSON.parse(stateJSON) as GameState) : null;
    }

    setGameState(gameState: GameState): void {
        this.storage.setItem(this.gameStateKey, JSON.stringify(gameState));
    }

    clearGameState(): void {
        this.storage.removeItem(this.gameStateKey);
    }
}
