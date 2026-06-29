// Reusable multi-agent review workflow.
// Fans out 5 independent reviewers (identical input), aggregates + de-duplicates
// their findings, then runs one adversarial verifier per finding that re-checks it
// against the actual code (and empirically tries drop-in scenarios).
//
// Run it (from a Claude session in this repo) via scriptPath — the `name` registry
// does NOT resolve files under .claude/workflows/:
//   Workflow({ scriptPath: ".claude/workflows/review-dropin.mjs" })
// or against another repo:
//   Workflow({ scriptPath: ".claude/workflows/review-dropin.mjs", args: { repo: "/abs/path" } })

export const meta = {
  name: 'review-dropin',
  description: 'Fan out 5 reviewers assessing a package as a drop-in xgboost replacement, aggregate, then adversarially push back on each finding',
  whenToUse: 'When you want a full, unbiased, multi-perspective review with adversarial verification of each finding.',
  phases: [
    { title: 'Review', detail: '5 independent reviews, identical input, drop-in-replacement lens' },
    { title: 'Aggregate', detail: 'merge + dedup into canonical findings' },
    { title: 'Pushback', detail: 'adversarially verify each finding against the code' },
  ],
}

const REPO = (args && args.repo) || '/home/zen930/projects/population-based-training-v2026'

const REVIEW_PROMPT = [
  'You are a senior Python ML engineer doing a FULL, UNBIASED, independent review of the genetic_xgb',
  'package. The OVERRIDING QUESTION for this review: is it 100% usable as a DROP-IN REPLACEMENT for',
  'xgboost.XGBClassifier / xgboost.XGBRegressor in real ML workflows? Judge it against what a team',
  'already using XGBoost would need to swap it in with minimal friction.',
  '',
  'Repo root: ' + REPO,
  'READ the actual code before judging. Public API: GeneticXGBClassifier, GeneticXGBRegressor',
  '(src/genetic_xgb/estimators.py), plus search_space/metrics/strategy/trainer/etc. Use Read/Grep/',
  'Bash (read-only). You MAY run "uv run pytest -p no:cacheprovider" and write small throwaway',
  'scripts under /tmp (use "uv run python ...") to EMPIRICALLY test drop-in scenarios. DO NOT modify',
  'any tracked file.',
  '',
  'ACTUALLY TRY these drop-in scenarios and report what breaks (with the real error):',
  '  1. fit signature: XGBoost is fit(X, y); genetic_xgb is fit(X_train, y_train, X_val, y_val).',
  '     Can it be used where code calls estimator.fit(X, y)? What about sample_weight?',
  '  2. sklearn integration: Pipeline, GridSearchCV / RandomizedSearchCV, cross_val_score, clone();',
  '     does it subclass BaseEstimator/ClassifierMixin/RegressorMixin? get_params()/set_params()',
  '     round-trip? Is there a score() method?',
  '  3. pandas DataFrame input: feature names, n_features_in_, feature_names_in_; predict on a',
  '     DataFrame; column-order safety.',
  '  4. model persistence: pickle / joblib.dump of a FITTED estimator + reload + predict;',
  '     save_model/load_model parity; cross-process reuse.',
  '  5. output parity: predict / predict_proba shapes and types vs XGBClassifier; classes_ semantics;',
  '     regressor predict dtype/shape; feature_importances_ availability.',
  '  6. ergonomics for adoption: required validation split (no internal CV), defaults, runtime cost',
  '     vs a single XGBoost fit, and silent behavioral differences that would surprise an XGBoost user.',
  '',
  'Also cover general correctness/ML-soundness/testing/docs issues, but PRIORITIZE concrete drop-in',
  'blockers. For each finding: severity, exact location (file:line), crisp description and WHY it',
  'blocks/limits drop-in use, and a concrete suggested fix. Prefer real reproduced failures.',
].join('\n')

const REVIEW_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['dropin_verdict', 'overall_assessment', 'findings'],
  properties: {
    dropin_verdict: { type: 'string', enum: ['drop-in-ready', 'minor-friction', 'major-friction', 'not-a-drop-in'] },
    overall_assessment: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'severity', 'category', 'location', 'description', 'suggested_fix', 'blocks_dropin'],
        properties: {
          title: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low', 'nit'] },
          category: { type: 'string', enum: ['dropin-compat', 'correctness', 'ml-soundness', 'api-design', 'testing', 'packaging-docs', 'performance'] },
          location: { type: 'string' },
          description: { type: 'string' },
          suggested_fix: { type: 'string' },
          blocks_dropin: { type: 'boolean' },
        },
      },
    },
  },
}

const AGG_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['findings'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'title', 'severity', 'category', 'location', 'description', 'suggested_fix', 'blocks_dropin', 'raised_by_count'],
        properties: {
          id: { type: 'string' },
          title: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low', 'nit'] },
          category: { type: 'string' },
          location: { type: 'string' },
          description: { type: 'string' },
          suggested_fix: { type: 'string' },
          blocks_dropin: { type: 'boolean' },
          raised_by_count: { type: 'integer' },
        },
      },
    },
  },
}

const PUSHBACK_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['id', 'title', 'verdict', 'severity_assessment', 'blocks_dropin', 'evidence', 'rebuttal_or_confirmation', 'recommended_action'],
  properties: {
    id: { type: 'string' },
    title: { type: 'string' },
    verdict: { type: 'string', enum: ['confirmed', 'partially-confirmed', 'refuted', 'subjective-judgment-call'] },
    severity_assessment: { type: 'string', enum: ['critical', 'high', 'medium', 'low', 'nit', 'non-issue'] },
    blocks_dropin: { type: 'boolean' },
    evidence: { type: 'string' },
    rebuttal_or_confirmation: { type: 'string' },
    recommended_action: { type: 'string', enum: ['fix-now', 'fix-later', 'document-only', 'wont-fix', 'needs-user-decision'] },
  },
}

phase('Review')
const reviews = (await parallel(
  [1, 2, 3, 4, 5].map((i) => () =>
    agent(REVIEW_PROMPT, { label: 'reviewer-' + i, phase: 'Review', schema: REVIEW_SCHEMA })
  )
)).filter(Boolean)
const totalRaw = reviews.reduce((n, r) => n + (r.findings ? r.findings.length : 0), 0)
log('Collected ' + reviews.length + ' reviews, ' + totalRaw + ' raw findings; verdicts: ' + reviews.map((r) => r.dropin_verdict).join(', '))

phase('Aggregate')
const aggPrompt = [
  'Aggregate these 5 independent reviews into ONE de-duplicated list. Merge findings describing the',
  'same underlying issue; set raised_by_count to how many of the 5 raised it. Keep every DISTINCT',
  'issue (including minority). Assign stable ids (F1, F2, ...). Preserve the most precise location',
  'and strongest fix, keep blocks_dropin true if any reviewer marked it so. Sort: drop-in blockers',
  'first, then by severity, then by raised_by_count desc.',
  '',
  'The 5 reviews as JSON:',
  JSON.stringify(reviews, null, 1),
].join('\n')
const aggregated = await agent(aggPrompt, { label: 'aggregator', phase: 'Aggregate', schema: AGG_SCHEMA })
const findings = (aggregated && aggregated.findings) || []
log('Aggregated to ' + findings.length + ' distinct findings; pushing back on each')

phase('Pushback')
const pushbacks = (await parallel(
  findings.map((f) => () =>
    agent(
      [
        'You are an ADVERSARIAL verifier. A review of genetic_xgb (judged as a DROP-IN xgboost',
        'replacement) produced the finding below. PUSH BACK: independently check it against the ACTUAL',
        'code and, where it is a usability/drop-in claim, EMPIRICALLY try it (write a tiny script under',
        '/tmp with "uv run python"). Default to skepticism. DO NOT modify tracked files.',
        '',
        'Repo root: ' + REPO,
        '',
        'FINDING ' + f.id + ' (' + f.severity + ', ' + f.category + ', blocks_dropin=' + f.blocks_dropin + ', raised ' + f.raised_by_count + '/5):',
        'Title: ' + f.title,
        'Location: ' + f.location,
        'Description: ' + f.description,
        'Suggested fix: ' + f.suggested_fix,
        '',
        'Return: verdict, independent severity assessment (may be non-issue), whether it truly blocks',
        'drop-in use, concrete evidence (code refs / actual error output), explicit rebuttal-or-',
        'confirmation, and a recommended action.',
      ].join('\n'),
      { label: 'pushback:' + f.id, phase: 'Pushback', schema: PUSHBACK_SCHEMA }
    )
  )
)).filter(Boolean)

return {
  dropin_verdicts: reviews.map((r) => r.dropin_verdict),
  reviewer_assessments: reviews.map((r) => r.overall_assessment),
  raw_finding_count: totalRaw,
  aggregated_findings: findings,
  pushbacks,
}
