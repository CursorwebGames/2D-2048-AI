export type Direction = 0 | 1 | 2 | 3; // 0: up, 1: right, 2: down, 3: left

interface InputEvents {
    move: Direction;
    restart: void;
    keepPlaying: void;
}

type EventCallback<K extends keyof InputEvents> = (data: InputEvents[K]) => void;

export class KeyboardInputManager {
    private events: { [K in keyof InputEvents]?: EventCallback<K>[] } = {};

    constructor() {
        this.listen();
    }

    on<K extends keyof InputEvents>(event: K, callback: EventCallback<K>): void {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event]!.push(callback);
    }

    private emit<K extends keyof InputEvents>(event: K, data: InputEvents[K]): void {
        const callbacks = this.events[event];
        if (callbacks) {
            callbacks.forEach((callback) => callback(data));
        }
    }

    private listen(): void {
        const map: Record<string, Direction> = {
            ArrowUp: 0,
            ArrowRight: 1,
            ArrowDown: 2,
            ArrowLeft: 3,
            KeyK: 0, // Vim up
            KeyL: 1, // Vim right
            KeyJ: 2, // Vim down
            KeyH: 3, // Vim left
            KeyW: 0,
            KeyD: 1,
            KeyS: 2,
            KeyA: 3,
        };

        // Respond to direction keys
        document.addEventListener("keydown", (event) => {
            const modifiers =
                event.altKey || event.ctrlKey || event.metaKey || event.shiftKey;
            const mapped = map[event.code];

            if (!modifiers) {
                if (mapped !== undefined) {
                    event.preventDefault();
                    this.emit("move", mapped);
                }

                // R key restarts the game
                if (event.code === "KeyR") {
                    this.restart(event);
                }
            }
        });

        // Respond to button presses
        this.bindButtonPress(".retry-button", this.restart);
        this.bindButtonPress(".restart-button", this.restart);
        this.bindButtonPress(".keep-playing-button", this.keepPlaying);

        // Respond to swipe events
        let touchStartClientX = 0;
        let touchStartClientY = 0;
        const gameContainer = document.querySelector(".game-container")!;

        gameContainer.addEventListener("touchstart", (event) => {
            const touchEvent = event as TouchEvent;
            if (touchEvent.touches.length > 1 || touchEvent.targetTouches.length > 1) {
                return; // Ignore if touching with more than 1 finger
            }

            touchStartClientX = touchEvent.touches[0].clientX;
            touchStartClientY = touchEvent.touches[0].clientY;

            event.preventDefault();
        });

        gameContainer.addEventListener("touchmove", (event) => {
            event.preventDefault();
        });

        gameContainer.addEventListener("touchend", (event) => {
            const touchEvent = event as TouchEvent;
            if (touchEvent.touches.length > 0 || touchEvent.targetTouches.length > 0) {
                return; // Ignore if still touching with one or more fingers
            }

            const touchEndClientX = touchEvent.changedTouches[0].clientX;
            const touchEndClientY = touchEvent.changedTouches[0].clientY;

            const dx = touchEndClientX - touchStartClientX;
            const absDx = Math.abs(dx);

            const dy = touchEndClientY - touchStartClientY;
            const absDy = Math.abs(dy);

            if (Math.max(absDx, absDy) > 10) {
                // (right : left) : (down : up)
                this.emit("move", absDx > absDy ? (dx > 0 ? 1 : 3) : dy > 0 ? 2 : 0);
            }
        });
    }

    private restart(event: Event): void {
        event.preventDefault();
        this.emit("restart", undefined);
    }

    private keepPlaying(event: Event): void {
        event.preventDefault();
        this.emit("keepPlaying", undefined);
    }

    private bindButtonPress(selector: string, fn: (event: Event) => void): void {
        const button = document.querySelector(selector)!;
        button.addEventListener("click", fn.bind(this));
        button.addEventListener("touchend", fn.bind(this));
    }
}
