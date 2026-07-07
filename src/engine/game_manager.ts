import { Grid, type GridState } from "./grid.ts";
import { Tile, type Position } from "./tile.ts";
import type { KeyboardInputManager, Direction } from "./keyboard_input_manager.ts";
import type { HTMLActuator } from "./html_actuator.ts";
import type { LocalStorageManager } from "./local_storage_manager.ts";

export interface GameState {
    grid: GridState;
    score: number;
    over: boolean;
    won: boolean;
    keepPlaying: boolean;
}

interface Vector {
    x: number;
    y: number;
}

export class GameManager {
    size: number;
    private inputManager: KeyboardInputManager;
    private storageManager: LocalStorageManager;
    private actuator: HTMLActuator;

    private startTiles = 2;

    private grid!: Grid;
    private score = 0;
    private over = false;
    private won = false;
    private keepPlaying = false;

    constructor(
        size: number,
        inputManager: KeyboardInputManager,
        actuator: HTMLActuator,
        storageManager: LocalStorageManager
    ) {
        this.size = size;
        this.inputManager = inputManager;
        this.storageManager = storageManager;
        this.actuator = actuator;

        this.inputManager.on("move", this.move.bind(this));
        this.inputManager.on("restart", this.restart.bind(this));
        this.inputManager.on("keepPlaying", this.continuePlaying.bind(this));

        this.setup();
    }

    // Restart the game
    private restart(): void {
        this.storageManager.clearGameState();
        this.actuator.continueGame(); // Clear the game won/lost message
        this.setup();
    }

    // Keep playing after winning (allows going over 2048)
    private continuePlaying(): void {
        this.keepPlaying = true;
        this.actuator.continueGame(); // Clear the game won/lost message
    }

    // Return true if the game is lost, or has won and the user hasn't kept playing
    private isGameTerminated(): boolean {
        return this.over || (this.won && !this.keepPlaying);
    }

    // Set up the game
    private setup(): void {
        const previousState = this.storageManager.getGameState();

        // Reload the game from a previous game if present
        if (previousState) {
            this.grid = new Grid(previousState.grid.size, previousState.grid.cells); // Reload grid
            this.score = previousState.score;
            this.over = previousState.over;
            this.won = previousState.won;
            this.keepPlaying = previousState.keepPlaying;
        } else {
            this.grid = new Grid(this.size);
            this.score = 0;
            this.over = false;
            this.won = false;
            this.keepPlaying = false;

            // Add the initial tiles
            this.addStartTiles();
        }

        // Update the actuator
        this.actuate();
    }

    // Set up the initial tiles to start the game with
    private addStartTiles(): void {
        for (let i = 0; i < this.startTiles; i++) {
            this.addRandomTile();
        }
    }

    // Adds a tile in a random position
    private addRandomTile(): void {
        if (this.grid.cellsAvailable()) {
            const value = Math.random() < 0.9 ? 2 : 4;
            const tile = new Tile(this.grid.randomAvailableCell()!, value);

            this.grid.insertTile(tile);
        }
    }

    // Sends the updated grid to the actuator
    private actuate(): void {
        if (this.storageManager.getBestScore() < this.score) {
            this.storageManager.setBestScore(this.score);
        }

        // Clear the state when the game is over (game over only, not win)
        if (this.over) {
            this.storageManager.clearGameState();
        } else {
            this.storageManager.setGameState(this.serialize());
        }

        this.actuator.actuate(this.grid, {
            score: this.score,
            over: this.over,
            won: this.won,
            bestScore: this.storageManager.getBestScore(),
            terminated: this.isGameTerminated(),
        });
    }

    // Represent the current game as an object
    private serialize(): GameState {
        return {
            grid: this.grid.serialize(),
            score: this.score,
            over: this.over,
            won: this.won,
            keepPlaying: this.keepPlaying,
        };
    }

    // Save all tile positions and remove merger info
    private prepareTiles(): void {
        this.grid.eachCell((_x, _y, tile) => {
            if (tile) {
                tile.mergedFrom = null;
                tile.savePosition();
            }
        });
    }

    // Move a tile and its representation
    private moveTile(tile: Tile, cell: Position): void {
        this.grid.cells[tile.x][tile.y] = null;
        this.grid.cells[cell.x][cell.y] = tile;
        tile.updatePosition(cell);
    }

    // Move tiles on the grid in the specified direction
    private move(direction: Direction): void {
        if (this.isGameTerminated()) return; // Don't do anything if the game's over

        const vector = this.getVector(direction);
        const traversals = this.buildTraversals(vector);
        let moved = false;

        // Save the current tile positions and remove merger information
        this.prepareTiles();

        // Traverse the grid in the right direction and move tiles
        traversals.x.forEach((x) => {
            traversals.y.forEach((y) => {
                const cell = { x, y };
                const tile = this.grid.cellContent(cell);

                if (tile) {
                    const positions = this.findFarthestPosition(cell, vector);
                    const next = this.grid.cellContent(positions.next);

                    // Only one merger per row traversal?
                    if (next && next.value === tile.value && !next.mergedFrom) {
                        const merged = new Tile(positions.next, tile.value * 2);
                        merged.mergedFrom = [tile, next];

                        this.grid.insertTile(merged);
                        this.grid.removeTile(tile);

                        // Converge the two tiles' positions
                        tile.updatePosition(positions.next);

                        // Update the score
                        this.score += merged.value;

                        // The mighty 2048 tile
                        if (merged.value === 2048) this.won = true;
                    } else {
                        this.moveTile(tile, positions.farthest);
                    }

                    if (!this.positionsEqual(cell, tile)) {
                        moved = true; // The tile moved from its original cell!
                    }
                }
            });
        });

        if (moved) {
            this.addRandomTile();

            if (!this.movesAvailable()) {
                this.over = true; // Game over!
            }

            this.actuate();
        }
    }

    // Get the vector representing the chosen direction
    private getVector(direction: Direction): Vector {
        // Vectors representing tile movement
        const map: Record<Direction, Vector> = {
            0: { x: 0, y: -1 }, // Up
            1: { x: 1, y: 0 }, // Right
            2: { x: 0, y: 1 }, // Down
            3: { x: -1, y: 0 }, // Left
        };

        return map[direction];
    }

    // Build a list of positions to traverse in the right order
    private buildTraversals(vector: Vector): { x: number[]; y: number[] } {
        const traversals: { x: number[]; y: number[] } = { x: [], y: [] };

        for (let pos = 0; pos < this.size; pos++) {
            traversals.x.push(pos);
            traversals.y.push(pos);
        }

        // Always traverse from the farthest cell in the chosen direction
        if (vector.x === 1) traversals.x = traversals.x.reverse();
        if (vector.y === 1) traversals.y = traversals.y.reverse();

        return traversals;
    }

    private findFarthestPosition(
        cell: Position,
        vector: Vector
    ): { farthest: Position; next: Position } {
        let previous: Position;

        // Progress towards the vector direction until an obstacle is found
        do {
            previous = cell;
            cell = { x: previous.x + vector.x, y: previous.y + vector.y };
        } while (this.grid.withinBounds(cell) && this.grid.cellAvailable(cell));

        return {
            farthest: previous,
            next: cell, // Used to check if a merge is required
        };
    }

    private movesAvailable(): boolean {
        return this.grid.cellsAvailable() || this.tileMatchesAvailable();
    }

    // Check for available matches between tiles (more expensive check)
    private tileMatchesAvailable(): boolean {
        for (let x = 0; x < this.size; x++) {
            for (let y = 0; y < this.size; y++) {
                const tile = this.grid.cellContent({ x, y });

                if (tile) {
                    for (let direction = 0; direction < 4; direction++) {
                        const vector = this.getVector(direction as Direction);
                        const cell = { x: x + vector.x, y: y + vector.y };

                        const other = this.grid.cellContent(cell);

                        if (other && other.value === tile.value) {
                            return true; // These two tiles can be merged
                        }
                    }
                }
            }
        }

        return false;
    }

    private positionsEqual(first: Position, second: Position): boolean {
        return first.x === second.x && first.y === second.y;
    }
}
