// Humanoid page: tag a teleop episode log in the browser (episodes.js), sample or the visitor's own file.
(function () {
  "use strict";
  const E = window.Episodes;
  const $ = (s) => document.querySelector(s);
  const summary = $("#eps-summary"), table = $("#eps-table"), tbody = $("#eps-table tbody");
  if (!E || !summary) return;
  const ORDER = { fall: 0, "near-fall": 1, lag: 2, clean: 3 };
  const pct = new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 1 }); // 52.5%, not a rounding that disagrees with the CLI

  function show(text, label) {
    let report;
    try { report = E.filterEpisodes(E.parseJsonl(text)); } catch (e) {
      summary.textContent = `Could not tag ${label}: ${e.message}`; table.hidden = true; return;
    }
    const c = report.counts;
    summary.textContent = `${label}: ${report.episodes} episodes. ${c.clean} clean, ${c.lag} lag, ${c["near-fall"]} near-fall, ${c.fall} fall. Keep ${pct.format(report.keep_share)} as nominal demonstrations.`;
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

  $("#eps-sample").addEventListener("click", () => {
    summary.textContent = "Tagging…";
    fetch("/assets/data/teleop-episodes.jsonl").then((r) => { if (!r.ok) throw new Error(r.status); return r.text(); })
      .then((t) => show(t, "Sample log")).catch(() => { summary.textContent = "The sample log is missing; run python -m breakpoint episodes --demo 200 --write-demo runs/episodes/teleop-demo.jsonl"; });
  });
  $("#eps-file").addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    if (f.size > 20e6) { summary.textContent = "That file is over 20 MB; use the command line for big logs."; return; }
    f.text().then((t) => show(t, f.name));
  });
})();
