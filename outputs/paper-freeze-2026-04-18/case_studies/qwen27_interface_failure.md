# Qwen 27B low-scoring trajectory with action-shell collapse

- Model: `qwen3.5:27b`
- Split: `core`
- Task: `task3_startup_week`
- Seed: `116`
- Score: `0.3685`
- Strict episode: `False`
- Empty fallback episode: `True`
- Response cap hit: `False`
- Hidden shift present: `True`

## Timeline

### Step 0

- Reward: `-0.0682`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 2` with `5` pending effects
- Action summary: `{"chosen_initiatives": ["optimize_infra", "refactor_incident_tooling", "automate_support_triage"], "messaging_action": "cost_comms", "pricing_change_pct": 0.0, "support_policy": "automation_first"}`

### Step 1

- Reward: `0.0763`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 3` with `2` pending effects
- Action summary: `{"chosen_initiatives": ["optimize_infra", "refactor_incident_tooling", "automate_support_triage"], "messaging_action": "cost_comms", "pricing_change_pct": 0.0, "support_policy": "automation_first"}`

### Step 2

- Reward: `0.0807`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 4` with `0` pending effects
- Action summary: `{"chosen_initiatives": ["optimize_infra", "refactor_incident_tooling", "automate_support_triage"], "messaging_action": "cost_comms", "pricing_change_pct": 0.0, "support_policy": "automation_first"}`

### Step 3

- Reward: `-0.0216`
- Strict step: `False`
- Empty fallback: `True`
- Cap hit: `False`
- Observation: `Day 5` with `0` pending effects
- Action summary: `{}`

### Step 4

- Reward: `-0.0100`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 6` with `3` pending effects
- Action summary: `{"chosen_initiatives": ["fix_login_bug", "improve_onboarding"], "messaging_action": "retention_campaign", "pricing_change_pct": 0.0, "support_policy": "balanced_triage"}`

### Step 5

- Reward: `0.0891`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 7` with `3` pending effects
- Action summary: `{"chosen_initiatives": ["launch_admin_analytics"], "messaging_action": "retention_campaign", "pricing_change_pct": 0.0, "support_policy": "balanced_triage"}`

### Step 6

- Reward: `0.1358`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 8` with `5` pending effects
- Action summary: `{"chosen_initiatives": ["ship_usage_pricing"], "messaging_action": "revenue_upsell", "pricing_change_pct": 0.05, "support_policy": "premium_sla"}`

### Step 7

- Reward: `-0.2013`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 9` with `3` pending effects
- Action summary: `{"chosen_initiatives": ["launch_referral_loop"], "messaging_action": "growth_push", "pricing_change_pct": 0.0, "support_policy": "balanced_triage"}`

### Step 8

- Reward: `-0.0501`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 10` with `1` pending effects
- Action summary: `{"chosen_initiatives": ["launch_referral_loop"], "messaging_action": "growth_push", "pricing_change_pct": 0.0, "support_policy": "balanced_triage"}`

### Step 9

- Reward: `-0.0506`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 11` with `0` pending effects
- Action summary: `{"chosen_initiatives": ["launch_referral_loop"], "messaging_action": "growth_push", "pricing_change_pct": 0.0, "support_policy": "balanced_triage"}`
