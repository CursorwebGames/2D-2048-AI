import "./style/main.scss";
import { GameManager } from "./game_manager.ts";
import { KeyboardInputManager } from "./keyboard_input_manager.ts";
import { HTMLActuator } from "./html_actuator.ts";
import { LocalStorageManager } from "./local_storage_manager.ts";

// Wait till the browser is ready to render the game (avoids glitches)
window.requestAnimationFrame(() => {
    new GameManager(
        4,
        new KeyboardInputManager(),
        new HTMLActuator(),
        new LocalStorageManager()
    );
});
