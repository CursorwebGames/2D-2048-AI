import { Tile, type Position, type TileState } from "./tile.ts";

export interface GridState {
    size: number;
    cells: (TileState | null)[][];
}

export class Grid {
    size: number;
    cells: (Tile | null)[][];

    constructor(size: number, previousState?: (TileState | null)[][]) {
        this.size = size;
        this.cells = previousState ? this.fromState(previousState) : this.empty();
    }

    // Build a grid of the specified size
    private empty(): (Tile | null)[][] {
        const cells: (Tile | null)[][] = [];

        for (let x = 0; x < this.size; x++) {
            const row: (Tile | null)[] = (cells[x] = []);

            for (let y = 0; y < this.size; y++) {
                row.push(null);
            }
        }

        return cells;
    }

    private fromState(state: (TileState | null)[][]): (Tile | null)[][] {
        const cells: (Tile | null)[][] = [];

        for (let x = 0; x < this.size; x++) {
            const row: (Tile | null)[] = (cells[x] = []);

            for (let y = 0; y < this.size; y++) {
                const tile = state[x][y];
                row.push(tile ? new Tile(tile.position, tile.value) : null);
            }
        }

        return cells;
    }

    // Find the first available random position
    randomAvailableCell(): Position | undefined {
        const cells = this.availableCells();

        if (cells.length) {
            return cells[Math.floor(Math.random() * cells.length)];
        }
    }

    availableCells(): Position[] {
        const cells: Position[] = [];

        this.eachCell((x, y, tile) => {
            if (!tile) {
                cells.push({ x, y });
            }
        });

        return cells;
    }

    // Call callback for every cell
    eachCell(callback: (x: number, y: number, tile: Tile | null) => void): void {
        for (let x = 0; x < this.size; x++) {
            for (let y = 0; y < this.size; y++) {
                callback(x, y, this.cells[x][y]);
            }
        }
    }

    // Check if there are any cells available
    cellsAvailable(): boolean {
        return !!this.availableCells().length;
    }

    // Check if the specified cell is taken
    cellAvailable(cell: Position): boolean {
        return !this.cellOccupied(cell);
    }

    cellOccupied(cell: Position): boolean {
        return !!this.cellContent(cell);
    }

    cellContent(cell: Position): Tile | null {
        if (this.withinBounds(cell)) {
            return this.cells[cell.x][cell.y];
        } else {
            return null;
        }
    }

    // Inserts a tile at its position
    insertTile(tile: Tile): void {
        this.cells[tile.x][tile.y] = tile;
    }

    removeTile(tile: Tile): void {
        this.cells[tile.x][tile.y] = null;
    }

    withinBounds(position: Position): boolean {
        return (
            position.x >= 0 &&
            position.x < this.size &&
            position.y >= 0 &&
            position.y < this.size
        );
    }

    serialize(): GridState {
        const cellState: (TileState | null)[][] = [];

        for (let x = 0; x < this.size; x++) {
            const row: (TileState | null)[] = (cellState[x] = []);

            for (let y = 0; y < this.size; y++) {
                const tile = this.cells[x][y];
                row.push(tile ? tile.serialize() : null);
            }
        }

        return {
            size: this.size,
            cells: cellState,
        };
    }
}
