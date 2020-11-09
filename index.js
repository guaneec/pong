"use strict";

const paddleWidth = 0.05
const paddleLength = 0.1
const paddleX = 0.1
const ballRadius = 0.0125
const paddleV = 0.35
const vYMax = 1
let human1Action = 1, human2Action = 1;

let player1, player2;
let allWeights;

const dims = [5, 10, 10, 3];

const actions = {
    p1: 1,
    p2: 1,
}

function resizeCanvas(canvas) {
    const w = canvas.parentElement.parentElement.clientWidth;
    const h = canvas.parentElement.parentElement.clientHeight;
    canvas.height = canvas.width = Math.min(w, h);
}

function drawState(ctx, state) {
    const canvas = ctx.canvas;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'black';

    // player 1
    ctx.fillRect((paddleX - paddleWidth) * w, state.p1Y * h, w * paddleWidth, h * paddleLength);

    // player 2
    ctx.fillRect((1 - paddleX) * w, state.p2Y * h, w * paddleWidth, h * paddleLength);

    // ball
    ctx.fillRect(w * (state.ballX - ballRadius), h * (state.ballY - ballRadius), ballRadius * 2 * w, ballRadius * 2 * h);
}

function mirrorMod(a, b) {
    let c = a % (2 * b);
    c = c < 0 ? c + 2 * b : c;
    return c < b ? [c, 1] : [2 * b - c, -1];
}

class NNViz {
    constructor(elem) {
        this.elem = elem;
    }

    read(values) {
        this.elem.innerHTML = values.map(layer => '<div class="layer">' + layer.map(cell => `<div class="cell" style="opacity: ${0.1 + 0.9 * cell};"></div>`).join('') + '</div>').join('');
    }
}

let nnvizs;

class Matrix {
    constructor(data, l1, l2) {
        if (l1 * l2 != data.length) throw new Error('wrong dimensions');
        this.data = data;
        this.l1 = l1;
        this.l2 = l2;
    }

    at(i1, i2) {
        return this.data[i1 * this.l2 + i2];
    }

    activate() {
        this.data = this.data.map(x => Math.tanh(x));
    }

    softmax() {
        const d = Math.max(...this.data);
        this.data.forEach((_v, i, a) => a[i] -= d);
        this.data = this.data.map(Math.exp);
        const s = this.data.reduce((a, b) => a + b, 0);
        this.data = this.data.map(x => x / s);
    }
}

function matmul(m1, m2) {
    if (m1.l2 !== m2.l1) throw new Error('wrong dimensions');
    const data = new Array(m1.l1 * m2.l2).fill(0);
    const l = m1.l2;
    for (let i1 = 0; i1 < m1.l1; ++i1) {
        for (let i2 = 0; i2 < m2.l2; ++i2) {
            for (let k = 0; k < l; ++k) {
                data[i1 * m2.l2 + i2] += m1.at(i1, k) * m2.at(k, i2);
            }
        }
    }
    return new Matrix(data, m1.l1, m2.l2);
}

const argmax = (a) => a.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);

function matadd(m1, m2) {
    if (m1.l1 !== m2.l1 || m2.l2 !== m1.l2) throw new Error('wrong dimensions');
    const s = m1.l1 * m1.l2;
    const data = Array(s).fill(0);
    for (let i = 0; i < s; ++i) data[i] = m1.data[i] + m2.data[i];
    return new Matrix(data, m1.l1, m2.l2);
}

class PongState {
    constructor() {
        this.serve();
    }

    serve(player) {
        this.ballX = 0.5;
        this.ballY = 0.5;
        const direction = player == 1 ? -1 : player == 2 ? 1 : Math.floor(Math.random() * 2) * 2 - 1;
        this.ballVX = 0.4 * direction;
        this.ballVY = -0.2;
        this.p1Y = 0.5;
        this.p2Y = 0.5;
    }

    step(dt, action1, action2, on1Hit, on2Hit, onScore) {
        this.p1Y += dt * paddleV * (action1 - 1);
        this.p2Y += dt * paddleV * (action2 - 1);
        this.p1Y = Math.min(1 - paddleLength, Math.max(0, this.p1Y));
        this.p2Y = Math.min(1 - paddleLength, Math.max(0, this.p2Y));
        const prevX = this.ballX;
        const prevY = this.ballY;
        this.ballX += this.ballVX * dt;
        this.ballY += this.ballVY * dt;
        let flipY;
        [this.ballY, flipY] = mirrorMod(this.ballY - ballRadius, 1 - 2 * ballRadius);
        this.ballY += ballRadius;
        this.ballVY *= flipY;

        if (prevX != this.ballX) {
            let contactY;
            // p1 collision
            contactY = (prevY - this.ballY) / (prevX - this.ballX) * (paddleX - this.ballX + ballRadius) + this.ballY - this.p1Y;
            if (prevX > paddleX + ballRadius && paddleX + ballRadius > this.ballX && -ballRadius <= contactY && contactY <= paddleLength + ballRadius) {
                this.ballX = 2 * (paddleX + ballRadius) - this.ballX;
                this.ballVX *= -1;
                this.ballVY = Math.abs(this.ballVX) * (contactY / (paddleLength + ballRadius * 2) - 0.5 + 0.05 * Math.random()) / 0.5 * vYMax;
                if (on1Hit) on1Hit();
            }

            contactY = (prevY - this.ballY) / (prevX - this.ballX) * (1 - paddleX - this.ballX - ballRadius) + this.ballY - this.p2Y;
            if (prevX < 1 - paddleX - ballRadius && 1 - paddleX - ballRadius < this.ballX && -ballRadius <= contactY && contactY <= paddleLength + ballRadius) {
                this.ballX = 2 * (1 - paddleX - ballRadius) - this.ballX;
                this.ballVX *= -1;
                this.ballVY = Math.abs(this.ballVX) * (contactY / (paddleLength + ballRadius * 2) - 0.5 + 0.05 * Math.random()) / 0.5 * vYMax;
                if (on2Hit) on2Hit();
            }
        }

        if (this.ballX < 0) {
            if (onScore) onScore(2);
            this.serve();
        }
        else if (this.ballX > 1) {
            if (onScore) onScore(1);
            this.serve();
        }
    }

    matrix(side) {
        const data = side === 2 ? [this.p2Y, this.ballX, this.ballY, this.ballVX, this.ballVY] : [this.p1Y, 1 - this.ballX, this.ballY, -this.ballVX, this.ballVY];
        return new Matrix(data, 1, 5);
    }
}

class Policy {
    reset() {

    }
}

class CPUPolicy extends Policy {
    constructor(side) {
        super();
        this.side = side;
        this.reset();
    }

    reset() {
        this.bias = (Math.random() - 0.5) * paddleLength;
    }

    getAction(state) {
        const pY = this.side == 1 ? state.p1Y : state.p2Y;
        return pY + paddleLength / 2 + this.bias > state.ballY ? 0 : 2;
    }
}

class NNPolicy extends Policy {
    constructor(side, id) {
        super();
        this.side = side;
        const wFlat = allWeights[id][0];
        const ws = Array(dims.length - 1);
        const bs = Array(dims.length - 1);
        for (let i = 0, j = 0; i < dims.length - 1; ++i) {
            const [l1, l2] = dims.slice(i, i + 2);
            ws[i] = new Matrix(wFlat.slice(j, j + l1 * l2), l1, l2);
            bs[i] = new Matrix(wFlat.slice(j + l1 * l2, j + l1 * l2 + l2), 1, l2);
            j += (l1 + 1) * l2;
        }
        this.ws = ws;
        this.bs = bs;
    }
    getAction(state) {
        let y = state.matrix(this.side);
        const values = []
        for (let i = 0; i < this.ws.length; ++i) {
            y = matmul(y, this.ws[i]);
            y = matadd(y, this.bs[i]);
            if (i !== this.ws.length - 1) {
                y.activate();
                values.push(y.data.map(x => x * 0.5 + 0.5));
            }
            else {
                y.softmax();
                values.push([...y.data]);
            }
        }
        nnvizs[this.side - 1].read(values);
        return argmax(y.data);
    }
}

class HumanPolicy extends Policy {
    constructor(side) {
        super();
        this.side = side;
    }
    getAction() {
        return this.side === 1 ? human1Action : human2Action;
    }
}

function runGame(ctx) {
    const state = new PongState();
    let t0;
    const onScore = (player) => {
        const label = document.getElementById(`score${player}`);
        label.innerText = 1 + parseInt(label.innerText);
    }
    function tick(t) {
        requestAnimationFrame(tick);
        const dt = t - t0;
        t0 = t;
        drawState(ctx, state);
        actions.p1 = player1.getAction(state);
        actions.p2 = player2.getAction(state);
        state.step(dt / 1000, actions.p1, actions.p2, () => player1.reset(), () => player2.reset(), onScore);
    }
    t0 = performance.now();
    tick(t0);
}


document.addEventListener('DOMContentLoaded', async () => {
    allWeights = await (await fetch('weights.json')).json();
    console.log(Object.keys(allWeights));
    const canvas = document.getElementsByTagName('canvas')[0];
    resizeCanvas(canvas);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'red';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.addEventListener('keydown', (e) => {
        if (e.key == 'w') human1Action = 0;
        if (e.key == 'ArrowUp') human2Action = 0;
        if (e.key == 's') human1Action = 2;
        if (e.key == 'ArrowDown') human2Action = 2;
    });
    document.addEventListener('keyup', (e) => {
        if (['w', 's'].includes(e.key)) human1Action = 1;
        if (['ArrowUp', 'ArrowDown'].includes(e.key)) human2Action = 1;
    })
    addEventListener('resize', () => {
        console.log('resize');
        resizeCanvas(canvas);
    });
    const tmp = Object.keys(allWeights).sort().map(x => `<option>${x}</option>`).join('');
    const idSel1 = document.querySelector('select#id1');
    const idSel2 = document.querySelector('select#id2');
    idSel1.innerHTML = tmp;
    idSel2.innerHTML = tmp;
    const sel1 = document.querySelector('select#p1');
    const sel2 = document.querySelector('select#p2');
    const setP1 = () => {
        switch (sel1.value) {
            case 'chaser':
                player1 = new CPUPolicy(1);
                break;
            case 'nn':
                player1 = new NNPolicy(1, idSel1.value);
                break;
            case 'human':
                player1 = new HumanPolicy(1);
                break;
        }
        document.getElementById('nn1').innerHTML = '';
        idSel1.style.display = sel1.value === 'nn' ? '' : 'none';
        document.getElementById('info1').style.display = sel1.value === 'nn' ? '' : 'none';
    }
    const setP2 = () => {
        switch (sel2.value) {
            case 'chaser':
                player2 = new CPUPolicy(2);
                break;
            case 'nn':
                player2 = new NNPolicy(2, idSel2.value);
                break;
            case 'human':
                player2 = new HumanPolicy(2);
                break;
        }
        document.getElementById('nn2').innerHTML = '';
        idSel2.style.display = sel2.value === 'nn' ? '' : 'none';
        document.getElementById('info2').style.display = sel2.value === 'nn' ? '' : 'none';
    }
    setP1();
    setP2();
    sel1.addEventListener('change', setP1);
    sel2.addEventListener('change', setP2);
    idSel1.addEventListener('change', setP1);
    idSel2.addEventListener('change', setP2);
    document.querySelector('button').addEventListener('click', () => {
        document.querySelectorAll('#score1, #score2').forEach(el => el.innerHTML = '0');
    });
    nnvizs = [new NNViz(document.getElementById('nn1')), new NNViz(document.getElementById('nn2'))];
    runGame(ctx);
})
