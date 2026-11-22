// Sokoban solver: every level is a hallway with independent rooms, and a box can never be
// pushed back out of a room, so the level's push-optimal count is the sum of its rooms'
// optima. Each room is at most 6x6 with 3-4 boxes: a breadth-first search over pushes
// finds its exact optimum. The player is then walked between the pushes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const DIRS: [number, number, string][] = [[0, -1, 'u'], [0, 1, 'd'], [-1, 0, 'l'], [1, 0, 'r']];

interface Level {
  w: number;
  h: number;
  floor: Uint8Array;
  goal: Uint8Array;
  box: Uint8Array;
  player: number;
}

function parse(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const floor = new Uint8Array(w * h);
  const goal = new Uint8Array(w * h);
  const box = new Uint8Array(w * h);
  let player = -1;
  for (let y = 0; y < h; y++) {
    const row = rows[y];
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      if (ch === '#') continue;
      const i = y * w + x;
      floor[i] = 1;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') box[i] = 1;
      if (ch === '@' || ch === '+') player = i;
    }
  }
  return { w, h, floor, goal, box, player };
}

/** Breadth-first steps from `from` to `to` over floor cells, treating boxes as walls. */
function walk(lv: Level, boxes: Uint8Array, from: number, to: number): string | null {
  const n = lv.w * lv.h;
  if (from === to) return '';
  const prev = new Int32Array(n).fill(-1);
  const seen = new Uint8Array(n);
  seen[from] = 1;
  const queue = [from];
  for (let q = 0; q < queue.length; q++) {
    const cur = queue[q];
    const x = cur % lv.w;
    for (const [dx, dy, ch] of DIRS) {
      const nx = x + dx;
      const ny = ((cur - x) / lv.w | 0) + dy;
      if (nx < 0 || nx >= lv.w || ny < 0 || ny >= lv.h) continue;
      const next = ny * lv.w + nx;
      if (!lv.floor[next] || boxes[next] || seen[next]) continue;
      seen[next] = 1;
      prev[next] = cur;
      if (next === to) {
        const out: string[] = [];
        let c = next;
        while (c !== from) {
          const p = prev[c];
          const px = p % lv.w;
          out.push(DIRS.find(([ddx, ddy]) => p + ddy * lv.w + ddx === c)![2]);
          c = p;
        }
        return out.reverse().join('');
      }
      queue.push(next);
    }
  }
  return null;
}

interface Push {
  from: number;
  dir: number;
}

/** The room's push-optimal plan: pushes that put every box on a goal, starting at `start`. */
function solveRoom(lv: Level, cells: Uint8Array, extra: Uint8Array, start: number): Push[] | null {
  const w = lv.w;
  const n = lv.w * lv.h;
  const walkable = new Uint8Array(n);
  const cellList: number[] = [];
  for (let i = 0; i < n; i++) {
    if (cells[i] || extra[i]) walkable[i] = 1;
    if (cells[i]) cellList.push(i);
  }
  const boxes0 = cellList.filter((i) => lv.box[i]).sort((a, b) => a - b);
  const goals = cellList.filter((i) => lv.goal[i]);
  const m = boxes0.length;
  if (goals.length !== m) return null;

  // Each goal's lower bound for a box: the floor distance of the box's moves, the player ignored.
  const FAR = 1e9;
  const gdist: Int32Array[] = goals.map((g) => {
    const d = new Int32Array(n).fill(FAR);
    d[g] = 0;
    const queue = [g];
    for (let q = 0; q < queue.length; q++) {
      const cur = queue[q];
      const x = cur % w;
      const y = (cur - x) / w;
      for (const [dx, dy] of DIRS) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= w || ny < 0 || ny >= lv.h) continue;
        const next = ny * w + nx;
        if (!cells[next] || d[next] !== FAR) continue;
        d[next] = d[cur] + 1;
        queue.push(next);
      }
    }
    return d;
  });

  const used = new Uint8Array(m);
  const lower = (boxes: number[]): number => {
    let best = FAR;
    const rec = (i: number, sum: number) => {
      if (sum >= best) return;
      if (i === m) { best = sum; return; }
      for (let j = 0; j < m; j++) {
        if (used[j]) continue;
        used[j] = 1;
        rec(i + 1, sum + gdist[j][boxes[i]]);
        used[j] = 0;
      }
    };
    rec(0, 0);
    return best;
  };

  const reach = (boxes: number[], from: number): Uint8Array => {
    const set = new Set(boxes);
    const seen = new Uint8Array(n);
    if (set.has(from) || !walkable[from]) return seen;
    seen[from] = 1;
    const queue = [from];
    for (let q = 0; q < queue.length; q++) {
      const cur = queue[q];
      const x = cur % w;
      const y = (cur - x) / w;
      for (const [dx, dy] of DIRS) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= w || ny < 0 || ny >= lv.h) continue;
        const next = ny * w + nx;
        if (!walkable[next] || set.has(next) || seen[next]) continue;
        seen[next] = 1;
        queue.push(next);
      }
    }
    return seen;
  };

  interface State { boxes: number[]; player: number; g: number; key: string; prev?: State; push?: Push }
  const closed = new Set<string>();
  const startState: State = { boxes: boxes0, player: start, g: 0, key: '' };
  startState.key = startState.boxes.join(',') + '|' + start;
  const buckets: State[][] = [];
  const put = (st: State, f: number) => {
    (buckets[f] ??= []).push(st);
  };
  let fmin = lower(boxes0);
  put(startState, fmin);

  const solved = (boxes: number[]): boolean => boxes.every((b) => lv.goal[b]);

  while (fmin < buckets.length) {
    const bucket = buckets[fmin];
    if (!bucket || bucket.length === 0) { fmin++; continue; }
    const state = bucket.pop()!;
    if (closed.has(state.key)) continue;
    closed.add(state.key);
    if (solved(state.boxes)) {
      const plan: Push[] = [];
      let cur: State | undefined = state;
      while (cur && cur.push) {
        plan.push(cur.push);
        cur = cur.prev;
      }
      return plan.reverse();
    }
    const seen = reach(state.boxes, state.player);
    const boxSet = new Set(state.boxes);
    for (let bi = 0; bi < m; bi++) {
      const b = state.boxes[bi];
      const bx = b % w;
      const by = (b - bx) / w;
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = DIRS[di];
        const nx = bx + dx;
        const ny = by + dy;
        const mx = bx - dx;
        const my = by - dy;
        if (nx < 0 || nx >= w || ny < 0 || ny >= lv.h) continue;
        if (mx < 0 || mx >= w || my < 0 || my >= lv.h) continue;
        const to = ny * w + nx;
        const from = my * w + mx;
        if (!cells[to] || !walkable[from]) continue;
        if (boxSet.has(to) || boxSet.has(from)) continue;
        if (!seen[from]) continue;
        const boxes = state.boxes.slice();
        boxes[bi] = to;
        boxes.sort((a, b2) => a - b);
        const next: State = { boxes, player: b, g: state.g + 1, key: boxes.join(',') + '|' + b, prev: state, push: { from: b, dir: di } };
        if (closed.has(next.key)) continue;
        put(next, next.g + lower(boxes));
      }
    }
  }
  return null;
}

function solve(text: string): string {
  const lv = parse(text);
  const w = lv.w;
  const n = lv.w * lv.h;
  const hallway = (lv.player - (lv.player % w)) / w;

  // Rooms: the floor above and below the one-square hallway, as connected components.
  const roomCells = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const y = (i - (i % w)) / w;
    if (lv.floor[i] && y !== hallway) roomCells[i] = 1;
  }
  const comp = new Int32Array(n).fill(-1);
  const rooms: number[][] = [];
  for (let i = 0; i < n; i++) {
    if (!roomCells[i] || comp[i] >= 0) continue;
    const id = rooms.length;
    const cells: number[] = [];
    const queue = [i];
    comp[i] = id;
    for (let q = 0; q < queue.length; q++) {
      const cur = queue[q];
      cells.push(cur);
      const x = cur % w;
      const y = (cur - x) / w;
      for (const [dx, dy] of DIRS) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= w || ny < 0 || ny >= lv.h) continue;
        const next = ny * w + nx;
        if (!roomCells[next] || comp[next] >= 0) continue;
        comp[next] = id;
        queue.push(next);
      }
    }
    rooms.push(cells);
  }

  const boxes = new Uint8Array(lv.box);
  let player = lv.player;
  const moves: string[] = [];

  // Every room in hallway order, each solved on its own from the hallway cell at its door.
  rooms.sort((a, b) => Math.min(...a) - Math.min(...b));
  for (const cells of rooms) {
    const mask = new Uint8Array(n);
    for (const c of cells) mask[c] = 1;
    const doors = cells.filter((c) => {
      const x = c % w;
      const y = (c - x) / w;
      if (Math.abs(y - hallway) !== 1) return false;
      return lv.floor[hallway * w + x] === 1;
    });
    if (doors.length !== 1) continue;
    const door = doors[0];
    const doorX = door % w;
    const hall = hallway * w + doorX;
    const extra = new Uint8Array(n);
    extra[hall] = 1;
    const plan = solveRoom(lv, mask, extra, hall);
    if (!plan) continue;
    for (const push of plan) {
      const [dx, dy] = DIRS[push.dir];
      const stand = push.from - (dy * w + dx);
      const path = walk(lv, boxes, player, stand);
      if (path === null) return '';
      moves.push(path);
      moves.push(DIRS[push.dir][2]);
      const to = push.from + (dy * w + dx);
      boxes[push.from] = 0;
      boxes[to] = 1;
      player = push.from;
    }
  }
  return moves.join('');
}

for (const id of readdirSync(I).sort()) {
  const out = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${out}\n`);
}
