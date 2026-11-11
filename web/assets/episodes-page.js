// Humanoid page: tag a teleop episode log in the browser (episodes.js), sample or the visitor's own file.
(function () {
  "use strict";
  const E = window.Episodes;
  const $ = (s) => document.querySelector(s);
  const summary = $("#eps-summary"), table = $("#eps-table"), tbody = $("#eps-table tbody");
  if (!E || !summary) return;
  const ORDER = { fall: 0, "near-fall": 1, lag: 2, clean: 3 };
  const pct = new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 1 }); // 52.5%, not a rounding that disagrees with the CLI

  let request = 0; // only the latest sample/file request may write the result
  function show(text, label, id) {
    if (id !== request) return;
    let report;
    try { report = E.filterEpisodes(E.parseJsonl(text)); } catch (e) {
      summary.textContent = `Could not tag ${label}: ${e.message}`; table.hidden = true; return;
    }
    const c = report.counts;
    summary.textContent = `${label}: ${report.episodes} episode${report.episodes === 1 ? "" : "s"}. ${c.clean} clean, ${c.lag} lag, ${c["near-fall"]} near-fall, ${c.fall} fall. Keep ${pct.format(report.keep_share)} as nominal demonstrations.`;
    tbody.textContent = "";
    report.verdicts.filter((v) => v.tag !== "clean").sort((a, b) => ORDER[a.tag] - ORDER[b.tag]).slice(0, 40).forEach((v) => {
      const tr = document.createElement("tr");
      [v.episode, v.tag, v.reasons.join("; ")].forEach((t, i) => {
        const td = document.createElement("td"); td.textContent = t; if (i === 1) td.className = "num" + (v.tag === "fall" ? " bad" : ""); tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.hidden = !tbody.children.length;
  }

  const fail = (id, msg) => { if (id === request) { summary.textContent = msg; table.hidden = true; } };
  $("#eps-sample").addEventListener("click", () => {
    const id = ++request;
    summary.textContent = "Tagging…";
    fetch("/assets/data/teleop-episodes.jsonl").then((r) => { if (!r.ok) throw new Error(r.status); return r.text(); })
      .then((t) => show(t, "Sample log", id))
      .catch(() => fail(id, "The sample log is missing; run python -m breakpoint episodes --demo 200 --write-demo runs/episodes/teleop-demo.jsonl"));
  });
  $("#eps-file").addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    e.target.value = ""; // so picking the same (edited) file again fires change
    if (!f) return;
    const id = ++request;
    if (f.size > 20e6) { fail(id, "That file is over 20 MB; use the command line for big logs."); return; }
    summary.textContent = `Tagging ${f.name}…`;
    f.text().then((t) => setTimeout(() => show(t, f.name, id), 0)).catch(() => fail(id, `Could not read ${f.name}.`));
  });
})();
