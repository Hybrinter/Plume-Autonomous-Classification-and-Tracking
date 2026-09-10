export const meta = {
  name: 'sensor-ingest-implementation',
  description: 'Implement the 11-task sensor ingest chain plan: implement -> verify -> quality per task',
  phases: [
    { title: 'Task 1', detail: 'Band enum, MosaicFrame, fault codes' },
    { title: 'Task 2', detail: 'SensorConfig + smear threshold' },
    { title: 'Task 3', detail: 'CFA separation' },
    { title: 'Task 4', detail: 'Mosaic-plane calibration' },
    { title: 'Task 5', detail: 'DN normalization' },
    { title: 'Task 6', detail: 'Calibration artifact IO' },
    { title: 'Task 7', detail: 'Contract switchover', model: 'opus' },
    { title: 'Task 8', detail: 'Remove RawFrameMsg' },
    { title: 'Task 9', detail: 'RealSensor PySpin driver', model: 'opus' },
    { title: 'Task 10', detail: 'Raw-mosaic SIL scene' },
    { title: 'Task 11', detail: 'ADR + context docs' },
    { title: 'Final audit', detail: 'Whole-phase completeness check' },
  ],
}

const PLAN = 'docs/superpowers/plans/2026-06-09-sensor-ingest-chain.md'
const SPEC = 'docs/superpowers/specs/2026-06-09-pact-flight-final-state-design.md'

const TASKS = [
  { n: 1, title: 'Band enum, MosaicFrame type, new FaultCodes', model: 'sonnet' },
  { n: 2, title: 'SensorConfig + new preprocessing threshold', model: 'sonnet' },
  { n: 3, title: 'Demosaic / CFA separation', model: 'sonnet' },
  { n: 4, title: 'Mosaic-plane radiometric calibration (bad-pixel + dark/flat)', model: 'sonnet' },
  { n: 5, title: 'DN normalization', model: 'sonnet' },
  { n: 6, title: 'Calibration artifact loading (checksummed) + identity builder', model: 'sonnet' },
  { n: 7, title: 'The contract switchover (HAL -> mosaic; pipeline rewire)', model: 'opus' },
  { n: 8, title: 'Remove RawFrameMsg from the message contract', model: 'sonnet' },
  { n: 9, title: 'RealSensor -- full PySpin acquisition + control plane', model: 'opus' },
  { n: 10, title: 'Raw-mosaic scene rendering for SIL', model: 'sonnet' },
  { n: 11, title: 'ADR + context docs + final verification', model: 'sonnet' },
]

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    commits: { type: 'array', items: { type: 'string' }, description: 'git commit hashes created' },
    gates_green: { type: 'boolean' },
    summary: { type: 'string' },
    deviations: { type: 'array', items: { type: 'string' }, description: 'any departures from the plan and why' },
  },
  required: ['commits', 'gates_green', 'summary', 'deviations'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    pass: { type: 'boolean' },
    issues: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['blocker', 'minor'] },
          file: { type: 'string' },
          description: { type: 'string' },
        },
        required: ['severity', 'file', 'description'],
      },
    },
  },
  required: ['pass', 'issues'],
}

const QUALITY_SCHEMA = {
  type: 'object',
  properties: {
    changed: { type: 'boolean' },
    commit: { type: 'string' },
    summary: { type: 'string' },
  },
  required: ['changed', 'summary'],
}

const GATES = 'uv run pytest packages; uv run ruff check packages; uv run ruff format --check packages; uv run mypy packages; uv run lint-imports'

const COMMON = [
  'Repo: PACT flight software (Windows, PowerShell; uv workspace under packages/).',
  'MANDATORY first reads: CLAUDE.md, every file in .claude/rules/, the plan header section of ' + PLAN + ', and Section 3 of ' + SPEC + '.',
  'Gates (run from repo root, ALL must be green): ' + GATES,
  'The working tree has PRE-EXISTING unrelated modifications (src/pact/**, .idea/**, .claude/settings.local.json, .coverage, bash.exe.stackdump, tests/integration/**). NEVER stage, commit, revert, or otherwise touch those paths. Stage only files in your task scope.',
  'Commit messages: use the message given in the plan task, ending with a blank line then: Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>',
].join('\n')

function implPrompt(t) {
  return [
    'You are implementing ONE task of an approved TDD implementation plan, at maximum effort and rigor. Think through every step before acting; verify every expected output actually appears.',
    COMMON,
    '',
    'YOUR TASK: "### Task ' + t.n + ': ' + t.title + '" in ' + PLAN + '. Read that full section and execute its checkbox steps IN ORDER (write failing test, see it fail, implement, see it pass, commit).',
    'Rules of engagement:',
    '- Follow the plan code faithfully; the plan abbreviates some docstrings, so write FULL docstrings per .claude/rules/docstrings.md (summary, inputs with types, outputs with types, notes; module header summaries; REQ IDs in module docstrings). 100-char lines, ASCII only.',
    '- Where the plan says "follow the existing pattern in this file", read the file and match it exactly.',
    '- Tick the checkboxes of your task ONLY in ' + PLAN + ' (change "- [ ]" to "- [x]") and include the plan file in your commit.',
    '- Run the FULL gates before committing. If a gate fails, fix the root cause; do not skip, xfail, or loosen anything.',
    '- Do not start, modify, or anticipate any other task.',
    'Return (final message = raw data): the commit hash(es) you created, whether all gates are green, a 2-3 sentence summary, and any deviations from the plan with justification.',
  ].join('\n')
}

function verifyPrompt(t, commits) {
  return [
    'You are an independent verification agent. Your job is to find problems, not to approve. Be skeptical; verify claims by running commands and reading code yourself.',
    COMMON,
    '',
    'VERIFY: Task ' + t.n + ' ("' + t.title + '") of ' + PLAN + ', implemented in commit(s): ' + commits.join(', ') + ' (inspect with `git show <hash>`).',
    'Checks (all required):',
    '1. Plan fidelity: every checkbox step of the task section was actually done; code matches the contracts in the plan (signatures, types, Result usage, fault codes, file paths).',
    '2. Spec fidelity: the change is consistent with Section 3 of ' + SPEC + ' (raw-mosaic contract, demosaic in preprocess, drivers acquire-only, frames never on the bus, identity calibration SIL-only).',
    '3. Tests are real: they assert behavior (not tautologies), they would fail if the implementation were wrong, and `uv run pytest packages` passes NOW in the working tree.',
    '4. Gates: run the full gate set yourself and confirm green.',
    '5. Conventions: .claude/rules compliance (docstrings, typing, ASCII, line length, numpy shape comments, no **kwargs).',
    '6. No scope creep and no damage: nothing outside the task touched; pre-existing unrelated dirty files untouched.',
    'Severity: "blocker" = wrong behavior, failing/weak gates, contract mismatch, missing step. "minor" = style/docstring/naming polish a quality pass can absorb.',
    'Return pass=true only if there are ZERO blockers.',
  ].join('\n')
}

function fixPrompt(t, issues) {
  return [
    'You are fixing verification findings on an already-committed task, at maximum effort. Fix root causes, never suppress symptoms.',
    COMMON,
    '',
    'TASK UNDER REPAIR: "### Task ' + t.n + ': ' + t.title + '" in ' + PLAN + '.',
    'BLOCKER FINDINGS to resolve (fix every one):',
    JSON.stringify(issues.filter(i => i.severity === 'blocker'), null, 2),
    'Also address minors if trivial. Run the full gates green, then commit the fixes as a new commit: "fix(ingest): address task ' + t.n + ' verification findings" (+ the Co-Authored-By footer).',
    'Return: commit hash(es), gates_green, summary, deviations (empty array if none).',
  ].join('\n')
}

function qualityPrompt(t, commits, minors) {
  return [
    'You are a code-quality agent doing a behavior-preserving cleanup pass, at maximum effort and taste. You may refactor, simplify, deduplicate, fix docstrings/comments/naming -- but observable behavior, public contracts, and test semantics must not change.',
    COMMON,
    '',
    'SCOPE: the changes introduced by Task ' + t.n + ' ("' + t.title + '"), commits: ' + commits.join(', ') + '. Inspect with `git show`; only touch files those commits touched.',
    'Minor findings from the verifier to absorb (may be empty):',
    JSON.stringify(minors, null, 2),
    'Hunt for: needless complexity, duplication, dead code, comment noise (comments explaining the obvious or narrating the diff), docstring-rule violations, weak names, missed dataclass/Result idioms, numpy shape comments missing at declaration sites.',
    'If you change anything: run the full gates green, then commit once: "refactor(ingest): quality pass after task ' + t.n + '" (+ the Co-Authored-By footer). If the code is already clean, change nothing.',
    'Return: changed (boolean), commit hash if any, and a summary of what you improved or why nothing was needed.',
  ].join('\n')
}

const results = []
for (const t of TASKS) {
  const phase = 'Task ' + t.n
  log('Task ' + t.n + ': implementing (' + t.model + ')')
  let impl = await agent(implPrompt(t), {
    model: t.model, schema: IMPL_SCHEMA, label: 'implement:task' + t.n, phase,
  })
  if (!impl) return { aborted: 'Task ' + t.n + ' implementer died/skipped', results }

  let commits = impl.commits || []
  let verdict = await agent(verifyPrompt(t, commits), {
    model: 'fable', schema: VERDICT_SCHEMA, label: 'verify:task' + t.n, phase,
  })
  if (!verdict) return { aborted: 'Task ' + t.n + ' verifier died/skipped', results }

  let rounds = 0
  while ((!verdict.pass || !impl.gates_green) && rounds < 2) {
    rounds += 1
    log('Task ' + t.n + ': verification found blockers, fix round ' + rounds)
    const fix = await agent(fixPrompt(t, verdict.issues), {
      model: t.model, schema: IMPL_SCHEMA, label: 'fix:task' + t.n + '.r' + rounds, phase,
    })
    if (!fix) return { aborted: 'Task ' + t.n + ' fixer died/skipped', results }
    impl = fix
    commits = commits.concat(fix.commits || [])
    verdict = await agent(verifyPrompt(t, commits), {
      model: 'fable', schema: VERDICT_SCHEMA, label: 'reverify:task' + t.n + '.r' + rounds, phase,
    })
    if (!verdict) return { aborted: 'Task ' + t.n + ' re-verifier died/skipped', results }
  }
  if (!verdict.pass) {
    return { aborted: 'Task ' + t.n + ' still failing verification after 2 fix rounds', failing_issues: verdict.issues, results }
  }

  const minors = verdict.issues.filter(i => i.severity === 'minor')
  const quality = await agent(qualityPrompt(t, commits, minors), {
    model: 'opus', schema: QUALITY_SCHEMA, label: 'quality:task' + t.n, phase,
  })
  if (quality && quality.changed && quality.commit) commits = commits.concat([quality.commit])

  log('Task ' + t.n + ': done (' + commits.length + ' commit(s))')
  results.push({
    task: t.n, title: t.title, commits,
    fix_rounds: rounds,
    quality: quality ? quality.summary : 'quality agent skipped',
    deviations: impl.deviations,
  })
}

log('All 11 tasks done; running final completeness audit')
const audit = await agent([
  'You are the final completeness auditor for a finished implementation phase. Be adversarial: assume something was missed and try to find it.',
  COMMON,
  '',
  'AUDIT SCOPE: the entire sensor ingest chain phase. Inputs: the plan ' + PLAN + ' (all 11 tasks, including its Self-review notes section) and Section 3 of ' + SPEC + '.',
  'Checks:',
  '1. Run the full gate set; confirm green.',
  '2. For every spec Section 3 requirement, point to the code that implements it; flag any gap.',
  '3. `rg "RawFrameMsg|RAW_FRAME|BAND_INDICES|motion_smear_exposure_us" packages config` must show no live references (docs/history aside).',
  '4. Confirm every plan checkbox is ticked and every task commit exists on the branch (git log).',
  '5. Confirm the SIL closed-loop tests exercise the full ingest path on rendered mosaic signal.',
  'Severity rules as before: pass=true only with zero blockers.',
].join('\n'), { model: 'fable', schema: VERDICT_SCHEMA, label: 'final-audit', phase: 'Final audit' })

return { results, final_audit: audit }