export interface Position {
    x: number;
    y: number;
}

export interface TileState {
    position: Position;
    value: number;
}

export class Tile {
    x: number;
    y: number;
    value: number;
    previousPosition: Position | null = null;
    mergedFrom: [Tile, Tile] | null = null; // Tracks tiles that merged together

    constructor(position: Position, value = 2) {
        this.x = position.x;
        this.y = position.y;
        this.value = value;
    }

    savePosition(): void {
        this.previousPosition = { x: this.x, y: this.y };
    }

    updatePosition(position: Position): void {
        this.x = position.x;
        this.y = position.y;
    }

    serialize(): TileState {
        return {
            position: { x: this.x, y: this.y },
            value: this.value,
        };
    }
}
