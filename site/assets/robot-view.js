// Breakpoint robot viewer: renders MuJoCo geom poses (Z-up) with three.js r128 (global THREE).
// Frames are arrays of [x, y, z, qw, qx, qy, qz] per geom, optionally quantised (x1000 ints).
(function () {
  "use strict";

  const COLORS = { ground: 0x2a2830, groundIce: 0x9fc6e0, ghost: 0xf2731a };

  function geometryFor(spec) {
    const s = spec.size;
    if (spec.type === "box") return new THREE.BoxGeometry(s[0] * 2, s[1] * 2, s[2] * 2);
    if (spec.type === "capsule") {
      const g = new THREE.CylinderGeometry(s[0], s[0], s[1] * 2 + s[0], 14);
      g.rotateX(Math.PI / 2);
      return g;
    }
    return new THREE.SphereGeometry(s[0], 18, 14);
  }

  class RobotView {
    constructor(container, opts = {}) {
      this.container = container;
      this.opts = Object.assign({ background: 0x17161b, follow: true, ghostColor: COLORS.ghost }, opts);
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
      this.renderer.domElement.setAttribute("aria-hidden", "true");
      container.appendChild(this.renderer.domElement);
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(this.opts.background);
      this.camera = new THREE.PerspectiveCamera(40, 1, 0.05, 100);
      this.camera.position.set(-1.1, 0.75, 1.35);
      this.scene.add(new THREE.HemisphereLight(0xffffff, 0x3a3440, 0.95));
      const sun = new THREE.DirectionalLight(0xffffff, 0.75);
      sun.position.set(3, 6, 4);
      this.scene.add(sun);
      this.world = null;
      this.meshes = [];
      this.ghostMeshes = [];
      this.track = null;
      this.ghostTrack = null;
      this.frame = 0;
      this.playing = false;
      this.visible = true;
      this.loop = false;
      this.onFrame = null;
      this.onEnd = null;
      this._last = 0;
      this._acc = 0;
      this._resize = this.resize.bind(this);
      window.addEventListener("resize", this._resize);
      if ("IntersectionObserver" in window) {
        new IntersectionObserver((es) => { this.visible = es[0].isIntersecting; }, { threshold: 0.01 }).observe(container);
      }
      this.resize();
      requestAnimationFrame(this._tick.bind(this));
    }

    resize() {
      const w = this.container.clientWidth || 1;
      const h = this.container.clientHeight || 1;
      this.renderer.setSize(w, h, false);
      this.renderer.domElement.style.width = "100%";
      this.renderer.domElement.style.height = "100%";
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.render();
    }

    build(specs, terrain, ghostSpecs) {
      if (this.world) this.scene.remove(this.world);
      this.world = new THREE.Group();
      this.world.rotation.x = -Math.PI / 2; // MuJoCo is Z-up
      this.scene.add(this.world);
      this.world.add(this._ground(terrain));
      this.meshes = specs.map((s) => {
        const m = new THREE.Mesh(geometryFor(s), new THREE.MeshStandardMaterial({
          color: new THREE.Color(s.rgba[0], s.rgba[1], s.rgba[2]), roughness: 0.55, metalness: 0.05 }));
        this.world.add(m);
        return m;
      });
      this.ghostMeshes = (ghostSpecs || []).map((s) => {
        const m = new THREE.Mesh(geometryFor(s), new THREE.MeshStandardMaterial({
          color: this.opts.ghostColor, transparent: true, opacity: 0.3, depthWrite: false }));
        this.world.add(m);
        return m;
      });
    }

    _ground(terrain) {
      if (!terrain) {
        const g = new THREE.PlaneGeometry(40, 12, 1, 1);
        const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: COLORS.ground, roughness: 1 }));
        mesh.position.z = 0.06;
        return mesh;
      }
      const g = new THREE.PlaneGeometry(terrain.half_x * 2, terrain.half_y * 2, terrain.cols - 1, terrain.rows - 1);
      const pos = g.attributes.position;
      for (let r = 0; r < terrain.rows; r++) {
        for (let c = 0; c < terrain.cols; c++) {
          pos.setZ(r * terrain.cols + c, terrain.heights[(terrain.rows - 1 - r) * terrain.cols + c]);
        }
      }
      g.computeVertexNormals();
      const icy = terrain.friction < 0.3;
      return new THREE.Mesh(g, new THREE.MeshStandardMaterial({
        color: icy ? COLORS.groundIce : COLORS.ground, roughness: icy ? 0.2 : 1, flatShading: true }));
    }

    static decode(frames, quantised) {
      if (!quantised) return frames;
      return frames.map((f) => f.map((g) => g.map((v) => v / 1000)));
    }

    play(track, opts = {}) {
      this.track = track;
      this.ghostTrack = opts.ghost || null;
      this.dt = opts.dt || 0.04;
      this.loop = !!opts.loop;
      this.frame = 0;
      this._acc = 0;
      this.playing = true;
      this._apply(0);
    }

    pause() { this.playing = false; }
    resume() { if (this.track) this.playing = true; }

    showFrame(i) {
      if (!this.track) return;
      this.frame = Math.max(0, Math.min(this.track.length - 1, i));
      this._apply(this.frame, true);
    }

    _set(list, poses) {
      for (let i = 0; i < poses.length && i < list.length; i++) {
        const p = poses[i];
        list[i].position.set(p[0], p[1], p[2]);
        list[i].quaternion.set(p[4], p[5], p[6], p[3]);
      }
    }

    _apply(i, snap) {
      const f = this.track[Math.min(i, this.track.length - 1)];
      this._set(this.meshes, f);
      if (this.ghostTrack) this._set(this.ghostMeshes, this.ghostTrack[Math.min(i, this.ghostTrack.length - 1)]);
      if (this.opts.follow) {
        const t = f[0];
        const target = new THREE.Vector3(t[0] - 1.15, t[2] + 0.7, -t[1] + 1.35);
        if (snap) this.camera.position.copy(target); else this.camera.position.lerp(target, 0.12);
        this.camera.lookAt(t[0] + 0.15, t[2] - 0.02, -t[1]);
      }
      if (this.onFrame) this.onFrame(i, f);
      this.render();
    }

    render() { this.renderer.render(this.scene, this.camera); }

    _tick(now) {
      requestAnimationFrame(this._tick.bind(this));
      const elapsed = Math.min(0.1, (now - (this._last || now)) / 1000);
      this._last = now;
      if (!this.playing || !this.track || !this.visible) return;
      this._acc += elapsed;
      let advanced = false;
      while (this._acc >= this.dt) {
        this._acc -= this.dt;
        this.frame += 1;
        advanced = true;
      }
      const n = Math.max(this.track.length, this.ghostTrack ? this.ghostTrack.length : 0);
      if (this.frame >= n) {
        if (this.loop) { this.frame = 0; } else {
          this.frame = n - 1;
          this.playing = false;
          if (this.onEnd) this.onEnd();
        }
      }
      if (advanced) this._apply(this.frame);
    }
  }

  window.RobotView = RobotView;
})();
