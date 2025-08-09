from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def snake_game():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Snake Game</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background-color: #c8e6c9; /* Light green background */
            font-family: 'Arial', sans-serif;
            color: #333;
        }
        
        .game-container {
            text-align: center;
        }
        
        h1 {
            margin-bottom: 20px;
            color: #388e3c; /* Dark green */
            font-weight: bold;
            font-size: 3em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        canvas {
            border: 5px solid #388e3c; /* Dark green border */
            background-color: #a5d6a7; /* Lighter green background for canvas */
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.2);
        }
        
        .score {
            font-size: 24px;
            margin: 20px 0;
            color: #388e3c; /* Dark green */
            font-weight: bold;
        }
        
        .controls {
            margin-top: 20px;
            font-size: 16px;
            color: #555;
        }
        
        .game-over {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: rgba(255, 255, 255, 0.9);
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            display: none;
            animation: fadeIn 0.5s ease-in-out;
            border: 5px solid #388e3c;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translate(-50%, -60%); }
            to { opacity: 1; transform: translate(-50%, -50%); }
        }
        
        .game-over h2 {
            color: #d32f2f; /* Red for game over */
            margin-bottom: 15px;
            font-size: 2.5em;
        }

        .restart-btn {
            background-color: #4CAF50; /* Green */
            color: white;
            border: none;
            padding: 12px 25px;
            font-size: 18px;
            border-radius: 8px;
            cursor: pointer;
            margin-top: 20px;
            transition: background-color 0.3s, transform 0.2s;
            font-family: 'Arial', sans-serif;
            font-weight: bold;
        }
        
        .restart-btn:hover {
            background-color: #66BB6A; /* Lighter green */
            transform: scale(1.05);
        }

        .paused {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: rgba(255, 255, 255, 0.9);
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            display: none;
            border: 5px solid #388e3c;
        }

        .paused h2 {
            color: #388e3c;
            margin-bottom: 15px;
            font-size: 2.5em;
        }

    </style>
</head>
<body>
    <div class="game-container">
        <h1>SNAKE</h1>
        <div class="score">SCORE: <span id="score">0</span> | HIGH SCORE: <span id="highScore">0</span></div>
        <canvas id="gameCanvas" width="400" height="400"></canvas>
        <div class="controls">
            Use Arrow Keys or WASD to move | P to Pause/Resume
        </div>
        
        <div class="game-over" id="gameOver">
            <h2>GAME OVER</h2>
            <p>Final Score: <span id="finalScore">0</span></p>
            <button class="restart-btn" onclick="restartGame()">Play Again</button>
        </div>
        <div class="paused" id="paused">
            <h2>PAUSED</h2>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');
        const scoreElement = document.getElementById('score');
        const highScoreElement = document.getElementById('highScore');
        const gameOverElement = document.getElementById('gameOver');
        const finalScoreElement = document.getElementById('finalScore');
        const pausedElement = document.getElementById('paused');

        // Game variables
        const gridSize = 20;
        const tileCount = canvas.width / gridSize;

        let snake = [
            {x: 10, y: 10}
        ];
        let food = {};
        let dx = 0;
        let dy = 0;
        let score = 0;
        let highScore = 0;
        let gameRunning = false;
        let isPaused = false;
        let gameSpeed = 120;

        function generateFood() {
            food = {
                x: Math.floor(Math.random() * tileCount),
                y: Math.floor(Math.random() * tileCount)
            };
            
            for (let segment of snake) {
                if (segment.x === food.x && segment.y === food.y) {
                    generateFood();
                    return;
                }
            }
        }

        function drawSnake() {
            for (let i = 0; i < snake.length; i++) {
                const segment = snake[i];
                const segmentX = segment.x * gridSize;
                const segmentY = segment.y * gridSize;

                // Body segment color
                const gradient = ctx.createLinearGradient(segmentX, segmentY, segmentX + gridSize, segmentY + gridSize);
                gradient.addColorStop(0, '#8BC34A');
                gradient.addColorStop(1, '#4CAF50');
                ctx.fillStyle = gradient;

                ctx.fillRect(segmentX, segmentY, gridSize, gridSize);
                ctx.strokeStyle = '#388E3C';
                ctx.strokeRect(segmentX, segmentY, gridSize, gridSize);

                // Head
                if (i === 0) {
                    const headX = segment.x * gridSize;
                    const headY = segment.y * gridSize;
                    
                    ctx.fillStyle = '#689F38';
                    ctx.fillRect(headX, headY, gridSize, gridSize);
                    ctx.strokeStyle = '#33691E';
                    ctx.strokeRect(headX, headY, gridSize, gridSize);

                    // Eyes
                    ctx.fillStyle = 'white';
                    const eyeSize = gridSize / 5;
                    
                    if (dx === 1) { // Right
                        ctx.fillRect(headX + gridSize - eyeSize * 1.5, headY + eyeSize, eyeSize, eyeSize);
                        ctx.fillRect(headX + gridSize - eyeSize * 1.5, headY + gridSize - eyeSize * 2, eyeSize, eyeSize);
                    } else if (dx === -1) { // Left
                        ctx.fillRect(headX + eyeSize / 2, headY + eyeSize, eyeSize, eyeSize);
                        ctx.fillRect(headX + eyeSize / 2, headY + gridSize - eyeSize * 2, eyeSize, eyeSize);
                    } else if (dy === 1) { // Down
                        ctx.fillRect(headX + eyeSize, headY + gridSize - eyeSize * 1.5, eyeSize, eyeSize);
                        ctx.fillRect(headX + gridSize - eyeSize * 2, headY + gridSize - eyeSize * 1.5, eyeSize, eyeSize);
                    } else if (dy === -1) { // Up
                        ctx.fillRect(headX + eyeSize, headY + eyeSize / 2, eyeSize, eyeSize);
                        ctx.fillRect(headX + gridSize - eyeSize * 2, headY + eyeSize / 2, eyeSize, eyeSize);
                    }
                }
            }
        }

        function drawFood() {
            const foodX = food.x * gridSize;
            const foodY = food.y * gridSize;

            // Apple body
            ctx.fillStyle = 'red';
            ctx.beginPath();
            ctx.arc(foodX + gridSize / 2, foodY + gridSize / 2, gridSize / 2 - 2, 0, 2 * Math.PI);
            ctx.fill();

            // Apple stem
            ctx.fillStyle = '#8B4513';
            ctx.fillRect(foodX + gridSize / 2 - 2, foodY - 2, 4, 6);
            
            // Leaf
            ctx.fillStyle = 'green';
            ctx.beginPath();
            ctx.moveTo(foodX + gridSize / 2, foodY);
            ctx.lineTo(foodX + gridSize / 2 + 6, foodY - 6);
            ctx.lineTo(foodX + gridSize / 2, foodY - 4);
            ctx.fill();
        }

        function moveSnake() {
            const head = {x: snake[0].x + dx, y: snake[0].y + dy};
            snake.unshift(head);

            if (head.x === food.x && head.y === food.y) {
                score += 10;
                scoreElement.textContent = score;
                generateFood();
                if (gameSpeed > 50) {
                    gameSpeed -= 5;
                }
            } else {
                snake.pop();
            }
        }

        function checkCollision() {
            const head = snake[0];

            if (head.x < 0 || head.x >= tileCount || head.y < 0 || head.y >= tileCount) {
                return true;
            }

            for (let i = 1; i < snake.length; i++) {
                if (head.x === snake[i].x && head.y === snake[i].y) {
                    return true;
                }
            }

            return false;
        }

        function gameLoop() {
            if (!gameRunning || isPaused) return;

            ctx.shadowBlur = 0;
            ctx.fillStyle = '#a5d6a7';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            moveSnake();

            if (checkCollision()) {
                gameOver();
                return;
            }

            drawFood();
            drawSnake();

            setTimeout(gameLoop, gameSpeed);
        }

        function gameOver() {
            gameRunning = false;
            finalScoreElement.textContent = score;
            if (score > highScore) {
                highScore = score;
                localStorage.setItem('snakeHighScore', highScore);
                highScoreElement.textContent = highScore;
            }
            gameOverElement.style.display = 'block';
        }

        function togglePause() {
            isPaused = !isPaused;
            pausedElement.style.display = isPaused ? 'block' : 'none';
            if (!isPaused) {
                gameLoop();
            }
        }

        function restartGame() {
            snake = [{x: 10, y: 10}];
            dx = 0;
            dy = 0;
            score = 0;
            gameSpeed = 120;
            scoreElement.textContent = score;
            gameOverElement.style.display = 'none';
            pausedElement.style.display = 'none';
            isPaused = false;
            generateFood();
            gameRunning = true;
            gameLoop();
        }

        function changeDirection(event) {
            const P_KEY = 80;
            if (event.keyCode === P_KEY) {
                togglePause();
                return;
            }

            if (isPaused) return;

            if (!gameRunning && event.keyCode !== 32) return; // Allow space to start

            const LEFT_KEY = 37;
            const RIGHT_KEY = 39;
            const UP_KEY = 38;
            const DOWN_KEY = 40;
            
            const A_KEY = 65;
            const D_KEY = 68;
            const W_KEY = 87;
            const S_KEY = 83;

            const keyPressed = event.keyCode;
            const goingUp = dy === -1;
            const goingDown = dy === 1;
            const goingRight = dx === 1;
            const goingLeft = dx === -1;

            if ((keyPressed === LEFT_KEY || keyPressed === A_KEY) && !goingRight) {
                dx = -1;
                dy = 0;
            }
            if ((keyPressed === UP_KEY || keyPressed === W_KEY) && !goingDown) {
                dx = 0;
                dy = -1;
            }
            if ((keyPressed === RIGHT_KEY || keyPressed === D_KEY) && !goingLeft) {
                dx = 1;
                dy = 0;
            }
            if ((keyPressed === DOWN_KEY || keyPressed === S_KEY) && !goingUp) {
                dx = 0;
                dy = 1;
            }
        }

        document.addEventListener('keydown', changeDirection);

        document.addEventListener('DOMContentLoaded', function() {
            highScore = localStorage.getItem('snakeHighScore') || 0;
            highScoreElement.textContent = highScore;
            generateFood();
            gameRunning = true;
            gameLoop();
        });
    </script>
</body>
</html>
    ''')

if __name__ == '__main__':
    app.run(debug=True)
