import type { Grid } from "./grid.ts";
import type { Tile, Position } from "./tile.ts";
import type { RewardBreakdown } from "./reward.ts";

const SHOW_REWARD = true;

export interface ActuateMetadata {
    score: number;
    over: boolean;
    won: boolean;
    bestScore: number;
    terminated: boolean;
    reward: RewardBreakdown;
}

export class HTMLActuator {
    private tileContainer = document.querySelector(".tile-container")!;
    private scoreContainer = document.querySelector(".score-container")!;
    private bestContainer = document.querySelector(".best-container")!;
    private messageContainer = document.querySelector(".game-message")!;
    private rewardContainer = document.querySelector(".reward")!;

    private score = 0;
    private lastReward: RewardBreakdown = { total: 0, terms: [] };

    constructor() {
        this.renderReward();
    }

    actuate(grid: Grid, metadata: ActuateMetadata): void {
        window.requestAnimationFrame(() => {
            this.clearContainer(this.tileContainer);

            grid.cells.forEach((column) => {
                column.forEach((cell) => {
                    if (cell) {
                        this.addTile(cell);
                    }
                });
            });

            this.updateScore(metadata.score);
            this.updateBestScore(metadata.bestScore);
            this.updateReward(metadata.reward);

            if (metadata.terminated) {
                if (metadata.over) {
                    this.message(false); // You lose
                } else if (metadata.won) {
                    this.message(true); // You win!
                }
            }
        });
    }

    // Updates and (if enabled) displays the RL reward breakdown for the last move
    updateReward(reward: RewardBreakdown): void {
        this.lastReward = reward;
        this.renderReward();
    }

    private renderReward(): void {
        if (!SHOW_REWARD) {
            this.rewardContainer.textContent = "";
            return;
        }

        const breakdown = this.lastReward.terms
            .map((term) => `${term.label}: ${term.value > 0 ? "+" : ""}${term.value}`)
            .join(", ");

        this.rewardContainer.textContent = breakdown
            ? `Reward: ${this.lastReward.total} (${breakdown})`
            : `Reward: ${this.lastReward.total}`;
    }

    // Continues the game (both restart and keep playing)
    continueGame(): void {
        this.clearMessage();
    }

    private clearContainer(container: Element): void {
        while (container.firstChild) {
            container.removeChild(container.firstChild);
        }
    }

    private addTile(tile: Tile): void {
        const wrapper = document.createElement("div");
        const inner = document.createElement("div");
        const position = tile.previousPosition || { x: tile.x, y: tile.y };
        const positionClass = this.positionClass(position);

        // We can't use classlist because it somehow glitches when replacing classes
        const classes = ["tile", "tile-" + tile.value, positionClass];

        if (tile.value > 2048) classes.push("tile-super");

        this.applyClasses(wrapper, classes);

        inner.classList.add("tile-inner");
        inner.textContent = String(tile.value);

        if (tile.previousPosition) {
            // Make sure that the tile gets rendered in the previous position first
            window.requestAnimationFrame(() => {
                classes[2] = this.positionClass({ x: tile.x, y: tile.y });
                this.applyClasses(wrapper, classes); // Update the position
            });
        } else if (tile.mergedFrom) {
            classes.push("tile-merged");
            this.applyClasses(wrapper, classes);

            // Render the tiles that merged
            tile.mergedFrom.forEach((merged) => {
                this.addTile(merged);
            });
        } else {
            classes.push("tile-new");
            this.applyClasses(wrapper, classes);
        }

        // Add the inner part of the tile to the wrapper
        wrapper.appendChild(inner);

        // Put the tile on the board
        this.tileContainer.appendChild(wrapper);
    }

    private applyClasses(element: Element, classes: string[]): void {
        element.setAttribute("class", classes.join(" "));
    }

    private normalizePosition(position: Position): Position {
        return { x: position.x + 1, y: position.y + 1 };
    }

    private positionClass(position: Position): string {
        position = this.normalizePosition(position);
        return "tile-position-" + position.x + "-" + position.y;
    }

    private updateScore(score: number): void {
        this.clearContainer(this.scoreContainer);

        const difference = score - this.score;
        this.score = score;

        this.scoreContainer.textContent = String(this.score);

        if (difference > 0) {
            const addition = document.createElement("div");
            addition.classList.add("score-addition");
            addition.textContent = "+" + difference;

            this.scoreContainer.appendChild(addition);
        }
    }

    private updateBestScore(bestScore: number): void {
        this.bestContainer.textContent = String(bestScore);
    }

    private message(won: boolean): void {
        const type = won ? "game-won" : "game-over";
        const message = won ? "You win!" : "Game over!";

        this.messageContainer.classList.add(type);
        this.messageContainer.getElementsByTagName("p")[0].textContent = message;
    }

    private clearMessage(): void {
        this.messageContainer.classList.remove("game-won", "game-over");
    }
}
