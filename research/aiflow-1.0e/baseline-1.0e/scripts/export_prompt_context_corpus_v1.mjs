#!/usr/bin/env node
"use strict";

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

function args(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    if (!key?.startsWith("--") || argv[index + 1] == null) throw new Error("arguments must be --key value pairs");
    result[key.slice(2)] = argv[index + 1];
  }
  return result;
}

function dPath(value, label) {
  const resolved = path.resolve(value);
  if (!/^D:\\/i.test(resolved)) throw new Error(`${label} must remain on D: ${resolved}`);
  return resolved;
}

function sha256(file) {
  return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

function jsonLines(file) {
  return fs.readFileSync(file, "utf8").split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
}

const options = args(process.argv.slice(2));
for (const name of ["prompts", "truth", "output", "audit"]) {
  if (!options[name]) throw new Error(`--${name} is required`);
}
const promptPath = dPath(options.prompts, "prompt catalog");
const truthPath = dPath(options.truth, "evaluation truth");
const outputPath = dPath(options.output, "context corpus");
const auditPath = dPath(options.audit, "context corpus audit");
for (const file of [promptPath, truthPath]) if (!fs.statSync(file).isFile()) throw new Error(`input file missing: ${file}`);
for (const file of [outputPath, auditPath]) if (fs.existsSync(file)) throw new Error(`refusing to overwrite output: ${file}`);

const moduleUrl = `${pathToFileURL(promptPath).href}?sha=${sha256(promptPath)}`;
const { PROMPTS } = await import(moduleUrl);
if (!Array.isArray(PROMPTS) || !PROMPTS.length) throw new Error("prompt catalog export is empty");
const truthRows = jsonLines(truthPath);
const sequenceKey = labels => JSON.stringify(labels);
const truthSequences = new Set(truthRows.map(row => sequenceKey((row.target_cells || []).map(cell => String(cell.token)))));
const grouped = new Map();
let formulas = 0;
let excludedEquations = 0;
let excludedRelations = 0;
let excludedOverlap = 0;
for (const prompt of PROMPTS) {
  if (prompt.kind !== "formula") continue;
  formulas += 1;
  const labels = (prompt.cells || []).map(cell => String(cell.label));
  if (labels.includes("=")) { excludedEquations += 1; continue; }
  if ((prompt.relations || []).length) { excludedRelations += 1; continue; }
  if (truthSequences.has(sequenceKey(labels))) { excludedOverlap += 1; continue; }
  if (labels.length < 2) throw new Error(`formula prompt is too short: ${prompt.id}`);
  const key = sequenceKey(labels);
  const current = grouped.get(key);
  if (current) {
    current.source_prompt_ids.push(String(prompt.id));
    continue;
  }
  grouped.set(key, {
    schema: "aiflow-owned-prompt-context/v1",
    formula_id: `prompt-context-${String(grouped.size).padStart(4, "0")}`,
    labels,
    display: String(prompt.display),
    category: String(prompt.category),
    source_prompt_ids: [String(prompt.id)],
    commercial_training_rights: "project-owned collector prompt catalog",
  });
}
const rows = [...grouped.values()];
if (!rows.length) throw new Error("all prompt formulas were excluded");
const support = {};
for (const row of rows) for (const label of row.labels) support[label] = (support[label] || 0) + 1;
const output = rows.map(row => JSON.stringify(row)).join("\n") + "\n";
const audit = {
  schema: "aiflow-owned-prompt-context-audit/v1",
  prompt_catalog: { path: promptPath, sha256: sha256(promptPath), prompts: PROMPTS.length, formulas },
  excluded: {
    exact_current159_token_sequence: excludedOverlap,
    equation_or_arithmetic_target: excludedEquations,
    two_dimensional_relation: excludedRelations,
  },
  admitted: {
    unique_formula_sequences: rows.length,
    tokens: rows.reduce((total, row) => total + row.labels.length, 0),
    label_support: Object.entries(support)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([label, count]) => ({ label, count })),
  },
  evaluation_truth: { path: truthPath, sha256: sha256(truthPath), formulas: truthRows.length },
  contracts: {
    evaluation_sequence_overlap: 0,
    arithmetic_evaluation_training: false,
    relation_inference_training: false,
    raster_images: false,
    raw_ink: false,
    commercial_training_rights: true,
  },
};
fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, output, { encoding: "utf8", flag: "wx" });
fs.writeFileSync(auditPath, JSON.stringify(audit, null, 2) + "\n", { encoding: "utf8", flag: "wx" });
console.log(JSON.stringify({ event: "prompt_context_export_complete", output: outputPath, audit: auditPath, formulas: rows.length, tokens: audit.admitted.tokens }));
