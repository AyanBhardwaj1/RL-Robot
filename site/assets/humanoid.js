// Humanoid push recovery in the browser: a line-for-line port of breakpoint/humanoid.py.
// Linear inverted pendulum + capture-point stepping (biped) or driving under the capture point (wheeled).
// tests/test_humanoid_js.py runs this file under Node and checks it against the Python module.
(function (root) {
  "use strict";
  const G = 9.81;
  const DT = 0.002;
  const GRIPPY = 10; // friction used to ask "would it have recovered with ample grip?" (humanoid.py GRIPPY)
  const DIRECTIONS = { front: 180, "front-left": -135, left: -90, "rear-left": -45, rear: 0, "rear-right": 45, right: 90, "front-right": 135 };

  const BIPED = { kind: "biped", mass_kg: 72, com_height: 0.95, foot_half_length: 0.11, foot_half_width: 0.05,
    stance_half_width: 0.11, max_step: 0.55, max_lateral_step: 0.40, min_swing: 0.32, max_steps: 3,
    max_step_up: 0.18, ankle_torque: 150, swing_load_factor: 0.6 };
  const WHEELED = { kind: "wheeled", mass_kg: 90, com_height: 1.0, half_length: 0.30, half_width: 0.24,
    max_accel: 2.5, max_speed: 1.5, reaction_s: 0.05, max_step_up: 0.025 };
  const CONDITIONS = { payload_kg: 0, payload_reach: 0.35, payload_height: 1.05, friction: 0.8, latency_s: 0,
    actuator_strength: 1, floor_slope_deg: 0, floor_step_m: 0 };

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const ticks = (s) => Math.max(0, Math.floor(s / DT + 0.5)); // same rounding as humanoid.py _ticks
  const norm = (a, b) => Math.hypot(a, b);

  function effective(robot, c) {
    const m = robot.mass_kg + c.payload_kg;
    const z = (robot.mass_kg * robot.com_height + c.payload_kg * c.payload_height) / m;
    return { z, offset: (c.payload_kg * c.payload_reach) / m, m };
  }

  function result(recovered, steps, reason, t, rec) {
    return Object.assign({ recovered, steps, reason, time_s: t }, rec || {});
  }

  function recoverBiped(impulse, dirDeg, robot, cond, record, horizon) {
    robot = Object.assign({}, BIPED, robot); cond = Object.assign({}, CONDITIONS, cond); horizon = horizon || 5;
    const rec = record ? { com: [], capture: [], footsteps: [] } : null;
    if (cond.floor_step_m > robot.max_step_up) return result(false, 0, `blocked: a ${Math.round(cond.floor_step_m * 100)} cm step is taller than it can climb`, 0, rec);
    const { z, offset, m } = effective(robot, cond);
    const w = Math.sqrt(G / z);
    const swing = robot.min_swing * (1 + robot.swing_load_factor * cond.payload_kg / robot.mass_kg) / Math.max(cond.actuator_strength, 0.05);
    const slopeAcc = G * Math.sin(cond.floor_slope_deg * Math.PI / 180);
    const biasX = slopeAcc / (w * w);
    const copReach = robot.ankle_torque * Math.max(cond.actuator_strength, 0.05) / (m * G);
    let sx = 0, sy = 0;
    const supportHalfW = robot.stance_half_width + robot.foot_half_width;
    let x = offset, y = 0;
    const prePush = [offset + biasX, 0];
    const a = dirDeg * Math.PI / 180;
    let vx = (impulse / m) * Math.cos(a), vy = (impulse / m) * Math.sin(a);
    let t = 0, steps = 0, dbl = true, swingUntil = -1, planned = null;
    const lat = ticks(cond.latency_s);
    const hist = [];
    const maxOff = cond.friction * z;
    const hx = Math.min(robot.foot_half_length, copReach);
    // Smith predictor (see humanoid.py): roll the late measurement forward with the CoP commanded since.
    const eDt = Math.exp(w * DT), eN = Math.pow(eDt, lat);
    const p0x = clamp(prePush[0], -hx, hx), p0y = clamp(prePush[1], -supportHalfW, supportHalfW);
    const ringX = new Array(lat).fill(p0x), ringY = new Array(lat).fill(p0y);
    let sumX = (eN - 1) * p0x, sumY = (eN - 1) * p0y;
    let tick = 0, swX = 0, swY = 0; // swX/swY: the loaded foot during the first swing
    let reason = "";
    let i = 0;
    let finished = false;
    while (t < horizon) {
      const xix = x + vx / w + biasX, xiy = y + vy / w;
      hist.push([xix, xiy]);
      const seen = hist.length > lat ? hist[hist.length - 1 - lat] : prePush; // stale view is the pre-push state
      const estX = lat ? eN * seen[0] - sumX : seen[0], estY = lat ? eN * seen[1] - sumY : seen[1];
      const hw = dbl ? supportHalfW : robot.foot_half_width;
      const lox = sx - hx, hix = sx + hx, loy = sy - hw, hiy = sy + hw;
      let bx0 = lox, bx1 = hix, by0 = loy, by1 = hiy;
      if (planned !== null && dbl) { // first swing: only the other foot can hold the CoP
        bx0 = swX - hx; bx1 = swX + hx; by0 = swY - robot.foot_half_width; by1 = swY + robot.foot_half_width;
      }
      let px = clamp(estX, bx0, bx1), py = clamp(estY, by0, by1);
      // friction cone: pull the CoP toward the CoM until |x - p| <= mu z; slip only if no support point can
      const qx = clamp(x, bx0, bx1), qy = clamp(y, by0, by1), need = norm(x - qx, y - qy);
      if (need > maxOff) {
        reason = `slipped: needed friction ${(need / z).toFixed(2)} > ${cond.friction.toFixed(2)}`;
        if (cond.friction < GRIPPY) { // a slip only if ample grip would have saved it (see humanoid.py)
          const g = recoverBiped(impulse, dirDeg, robot, Object.assign({}, cond, { friction: GRIPPY }), false, horizon);
          if (!g.recovered) reason = `${g.reason}: the foot slipped first, at ${t.toFixed(2)} s; ample grip would not have saved it`;
        }
        finished = true; break;
      }
      if (norm(x - px, y - py) > maxOff) {
        const dx = px - qx, dy = py - qy, fx = qx - x, fy = qy - y;
        const a2 = dx * dx + dy * dy, b2 = 2 * (fx * dx + fy * dy), c2 = fx * fx + fy * fy - maxOff * maxOff;
        const sr = Math.min(Math.max((-b2 + Math.sqrt(Math.max(b2 * b2 - 4 * a2 * c2, 0))) / (2 * a2), 0), 1);
        px = qx + sr * dx; py = qy + sr * dy;
      }
      if (lat) {
        const j = tick % lat;
        sumX = eDt * sumX + (eDt - 1) * px - eN * (eDt - 1) * ringX[j];
        sumY = eDt * sumY + (eDt - 1) * py - eN * (eDt - 1) * ringY[j];
        ringX[j] = px; ringY[j] = py;
      }
      tick += 1;
      const inside = estX >= lox - 1e-9 && estX <= hix + 1e-9 && estY >= loy - 1e-9 && estY <= hiy + 1e-9;
      if (planned === null && !inside) {
        if (steps >= robot.max_steps) { reason = `needed more than ${robot.max_steps} steps`; finished = true; break; }
        let qx = px, qy = py; // CoP during the coming swing
        if (dbl) {
          const side = estY - sy > 1e-9 ? 1 : estY - sy < -1e-9 ? -1 : 1;
          swX = sx; swY = sy - side * robot.stance_half_width;
          qx = clamp(estX, swX - hx, swX + hx); qy = clamp(estY, swY - robot.foot_half_width, swY + robot.foot_half_width);
        }
        const e = Math.exp(w * swing);
        let dx = qx + (estX - qx) * e - sx, dy = qy + (estY - qy) * e - sy;
        const reach = norm(dx, dy);
        if (reach > robot.max_step) { dx *= robot.max_step / reach; dy *= robot.max_step / reach; }
        dy = clamp(dy, -robot.max_lateral_step, robot.max_lateral_step);
        planned = [sx + dx, sy + dy];
        swingUntil = t + swing;
      }
      if (planned !== null && t >= swingUntil) {
        sx = planned[0]; sy = planned[1]; planned = null; dbl = false; steps += 1;
        if (rec) rec.footsteps.push([+t.toFixed(3), +sx.toFixed(4), +sy.toFixed(4), steps % 2]);
      }
      const ax = w * w * (x - px) + slopeAcc, ay = w * w * (y - py);
      vx += ax * DT; vy += ay * DT; x += vx * DT; y += vy * DT; t += DT; i += 1;
      if (rec && i % 10 === 0) { rec.com.push([+t.toFixed(3), +x.toFixed(4), +y.toFixed(4)]); rec.capture.push([+t.toFixed(3), +xix.toFixed(4), +xiy.toFixed(4)]); }
      const nxix = x + vx / w + biasX, nxiy = y + vy / w;
      const rw = dbl ? supportHalfW : robot.foot_half_width;
      if (planned === null && norm(vx, vy) < 0.02 && Math.abs(nxix - sx) <= hx && Math.abs(nxiy - sy) <= rw && t > 0.3) {
        return result(true, steps, "came to rest", t, rec);
      }
      if (norm(nxix - sx, nxiy - sy) > robot.max_step * 3) { reason = "capture point ran away beyond leg reach"; finished = true; break; }
    }
    if (finished) return result(false, steps, reason, t, rec);
    const hw = dbl ? supportHalfW : robot.foot_half_width;
    const ex = x + vx / w + biasX, ey = y + vy / w;
    const ok = ex >= sx - hx && ex <= sx + hx && ey >= sy - hw && ey <= sy + hw;
    return result(ok, steps, ok ? "came to rest" : "still moving at the end of the horizon", t, rec);
  }

  function recoverWheeled(impulse, dirDeg, base, cond, record, horizon) {
    base = Object.assign({}, WHEELED, base); cond = Object.assign({}, CONDITIONS, cond); horizon = horizon || 5;
    const rec = record ? { com: [], capture: [], footsteps: [] } : null;
    if (cond.floor_step_m > base.max_step_up) return result(false, 0, `blocked: a ${Math.round(cond.floor_step_m * 100)} cm step is taller than it can climb`, 0, rec);
    const { z, offset, m } = effective(base, cond);
    const w = Math.sqrt(G / z);
    const accel = base.max_accel * Math.max(cond.actuator_strength, 0.05) * Math.min(1, 2 * cond.friction);
    const slopeAcc = G * Math.sin(cond.floor_slope_deg * Math.PI / 180);
    const biasX = slopeAcc / (w * w);
    let x = offset, y = 0, bx = 0, by = 0, bvx = 0, bvy = 0;
    const prePush = [offset + biasX, 0];
    const a = dirDeg * Math.PI / 180;
    let vx = (impulse / m) * Math.cos(a), vy = (impulse / m) * Math.sin(a);
    const hx = base.half_length, hy = base.half_width;
    const delay = ticks(cond.latency_s + base.reaction_s);
    const hist = [];
    let t = 0, i = 0;
    const tipX = hx + base.max_speed ** 2 / (2 * accel) + 0.25, tipY = hy + base.max_speed ** 2 / (2 * accel) + 0.25;
    while (t < horizon) {
      const xix = x + vx / w + biasX, xiy = y + vy / w;
      hist.push([xix - bx, xiy - by]);
      const seen = hist.length > delay ? hist[hist.length - 1 - delay] : prePush;
      const px = clamp(xix, bx - hx, bx + hx), py = clamp(xiy, by - hy, by + hy);
      const dn = norm(x - px, y - py);
      if (dn > cond.friction * z) return result(false, 0, `slipped: needed friction ${(dn / z).toFixed(2)} > ${cond.friction.toFixed(2)}`, t, rec);
      const ex = seen[0] - clamp(seen[0], -hx * 0.5, hx * 0.5), ey = seen[1] - clamp(seen[1], -hy * 0.5, hy * 0.5);
      let dvx, dvy;
      if (Math.abs(ex) > 1e-6 || Math.abs(ey) > 1e-6) {
        dvx = clamp(ex * 4, -base.max_speed, base.max_speed) - bvx; dvy = clamp(ey * 4, -base.max_speed, base.max_speed) - bvy;
      } else { dvx = -bvx; dvy = -bvy; }
      const n = norm(dvx, dvy);
      if (n > accel * DT) { dvx *= accel * DT / n; dvy *= accel * DT / n; }
      bvx += dvx; bvy += dvy; bx += bvx * DT; by += bvy * DT;
      const ax = w * w * (x - px) + slopeAcc, ay = w * w * (y - py);
      vx += ax * DT; vy += ay * DT; x += vx * DT; y += vy * DT; t += DT; i += 1;
      if (rec && i % 10 === 0) {
        rec.com.push([+t.toFixed(3), +x.toFixed(4), +y.toFixed(4)]);
        rec.capture.push([+t.toFixed(3), +xix.toFixed(4), +xiy.toFixed(4)]);
        rec.footsteps.push([+t.toFixed(3), +bx.toFixed(4), +by.toFixed(4), 2]);
      }
      const rx = x + vx / w + biasX - bx, ry = y + vy / w - by;
      if (Math.abs(rx) > tipX || Math.abs(ry) > tipY) return result(false, 0, "tipped over: the base could not drive under the capture point", t, rec);
      if (norm(vx, vy) < 0.02 && norm(bvx, bvy) < 0.02 && Math.abs(rx) <= hx && Math.abs(ry) <= hy && t > 0.3) return result(true, 0, "came to rest", t, rec);
    }
    const rx = x + vx / w + biasX - bx, ry = y + vy / w - by;
    const ok = Math.abs(rx) <= hx && Math.abs(ry) <= hy;
    return result(ok, 0, ok ? "came to rest" : "still moving at the end of the horizon", t, rec);
  }

  function recover(body, impulse, dirDeg, cond, record) {
    return body === "wheeled" ? recoverWheeled(impulse, dirDeg, {}, cond, record) : recoverBiped(impulse, dirDeg, {}, cond, record);
  }

  function breakingPoint(body, dirDeg, cond, maxImpulse, iters) {
    maxImpulse = maxImpulse || 200; iters = iters || 12;
    const ok = (j) => recover(body, j, dirDeg, cond, false).recovered;
    if (!ok(0)) return 0;
    if (ok(maxImpulse)) return maxImpulse;
    let lo = 0, hi = maxImpulse;
    for (let k = 0; k < iters; k++) { const mid = 0.5 * (lo + hi); if (ok(mid)) lo = mid; else hi = mid; }
    return Math.round(lo * 100) / 100;
  }

  function envelope(body, cond, iters) {
    const out = {};
    for (const [name, deg] of Object.entries(DIRECTIONS)) out[name] = breakingPoint(body, deg, cond, 200, iters || 12);
    return out;
  }

  // Steps needed to recover from an initial CoM velocity on an n x n grid; -1 = fell or blocked.
  // Row 0 = +vMax forward, column 0 = +vMax toward the robot's left (mirrors capturability_map in Python).
  function capturabilityMap(body, cond, n, vMax) {
    n = n || 41; vMax = vMax || 2;
    const robot = body === "wheeled" ? WHEELED : BIPED;
    const m = robot.mass_kg + Object.assign({}, CONDITIONS, cond).payload_kg;
    const grid = [];
    for (let i = 0; i < n; i++) {
      const vx = vMax - 2 * vMax * i / (n - 1);
      const row = [];
      for (let j = 0; j < n; j++) {
        const vy = vMax - 2 * vMax * j / (n - 1);
        const speed = Math.hypot(vx, vy);
        const r = recover(body, speed * m, Math.atan2(vy, vx) * 180 / Math.PI, cond, false);
        row.push(r.recovered ? r.steps : -1);
      }
      grid.push(row);
    }
    return grid;
  }

  const api = { capturabilityMap, DIRECTIONS, BIPED, WHEELED, CONDITIONS, recover, recoverBiped, recoverWheeled, breakingPoint, envelope, effective };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.Humanoid = api;
})(typeof window !== "undefined" ? window : globalThis);
