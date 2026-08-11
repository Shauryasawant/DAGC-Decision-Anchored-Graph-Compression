# DAGC Docs

This folder is the deep-dive companion to the [README](../README.md). The README
gets you installed and running in five minutes; these pages go one level deeper
into each part of the system, for anyone extending, tuning, or evaluating DAGC.

If you haven't run `compress()` yet, start with the [README Quick Start](../README.md#quick-start)
first — everything here assumes you've seen the basic call already.

## Core Concepts

<!-- Headroom-like card grid rendered as an HTML table so GitHub shows a compact grid -->
<table>
  <tr>
    <td align="center" valign="top" width="220">
      <div style="font-size:28px">📘</div>
      <strong><a href="./how-it-works.md">How Compression Works</a></strong>
      <div style="margin-top:6px">The four-stage pipeline: extraction → dependency graph → budget allocation → validation</div>
    </td>
    <td align="center" valign="top" width="220">
      <div style="font-size:28px">⚙️</div>
      <strong><a href="./tuning.md">Tuning</a></strong>
      <div style="margin-top:6px">Every <code>DAGCConfig</code> field, what it trades off, and when to change it</div>
    </td>
    <td align="center" valign="top" width="220">
      <div style="font-size:28px">🧠</div>
      <strong><a href="./decision-rationale.md">Decision Rationale</a></strong>
      <div style="margin-top:6px">Extracting <em>why</em> a decision was made, and making that survive compression</div>
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="220">
      <div style="font-size:28px">🔬</div>
      <strong><a href="./evaluation.md">Evaluation</a></strong>
      <div style="margin-top:6px">DRR, RCI, artifact retention — what each metric means and how to run them</div>
    </td>
    <td align="center" valign="top" width="220">
      <div style="font-size:28px">🚦</div>
      <strong><a href="./proxy.md">Optional Proxy</a></strong>
      <div style="margin-top:6px">Run <code>dagc-server</code>, fail-open behavior, and deployment tradeoffs</div>
    </td>
    <td align="center" valign="top" width="220">
      <div style="font-size:28px">🏗️</div>
      <strong><a href="./architecture.md">Architecture</a></strong>
      <div style="margin-top:6px">Why <code>dagc</code> and <code>dagc_eval</code> are split and offline-by-default guarantees</div>
    </td>
  </tr>
</table>

## Reading order

If you're evaluating DAGC for a new use case, this order tends to answer
questions in the sequence they come up:

1. <a href="./how-it-works.md">How Compression Works</a> — understand what's actually happening to your trace
2. <a href="./tuning.md">Tuning</a> — get the reduction/fidelity tradeoff right for your traces
3. <a href="./evaluation.md">Evaluation</a> — measure it, don't take the defaults on faith
4. <a href="./decision-rationale.md">Decision Rationale</a> — if your downstream task needs *why*, not just *what*
5. <a href="./architecture.md">Architecture</a> / <a href="./proxy.md">Optional Proxy</a> — if you're deploying this, not just calling it in-process

## Something not here?

The README's [Project Layout](../README.md#project-layout) section is the source
of truth for what code exists. If a doc page here describes behavior that doesn't
match `src/dagc/`, the code wins — open an issue.
