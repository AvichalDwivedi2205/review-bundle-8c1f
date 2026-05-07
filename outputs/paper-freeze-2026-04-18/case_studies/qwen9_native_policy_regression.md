# Qwen 9B lower-scoring native-policy trajectory without shell collapse

- Model: `qwen3.5:9b`
- Split: `core`
- Task: `task3_startup_week`
- Seed: `105`
- Score: `0.3070`
- Strict episode: `True`
- Empty fallback episode: `False`
- Response cap hit: `False`
- Hidden shift present: `True`

## Timeline

### Step 0

- Reward: `-0.0688`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 2` with `4` pending effects
- Action summary: `{"chosen_initiatives": ["optimize_infra", "fix_login_bug", "automate_support_triage"], "messaging_action": "cost_comms", "pricing_change_pct": 0.0, "support_policy": "automation_first"}`

### Step 1

- Reward: `0.1191`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 3` with `3` pending effects
- Action summary: `{"chosen_initiatives": ["refactor_incident_tooling"], "messaging_action": "retention_campaign", "pricing_change_pct": 0.0, "support_policy": "balanced_triage"}`

### Step 2

- Reward: `-0.0583`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 4` with `3` pending effects
- Action summary: `{"chosen_initiatives": ["launch_admin_analytics"], "messaging_action": "growth_push", "pricing_change_pct": 0.0, "support_policy": "balanced_triage"}`

### Step 3

- Reward: `0.0696`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 5` with `4` pending effects
- Action summary: `{"chosen_initiatives": ["improve_onboarding"], "messaging_action": "retention_campaign", "pricing_change_pct": 0.0, "support_policy": "balanced_triage"}`

### Step 4

- Reward: `0.0238`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 6` with `4` pending effects
- Action summary: `{"chosen_initiatives": ["ship_usage_pricing"], "messaging_action": "growth_push", "pricing_change_pct": 0.0, "support_policy": "balanced_triage"}`

### Step 5

- Reward: `-0.1262`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 7` with `2` pending effects
- Action summary: `{"chosen_initiatives": ["launch_referral_loop"], "messaging_action": "retention_campaign", "pricing_change_pct": 0.0, "support_policy": "balanced_triage"}`

### Step 6

- Reward: `-0.0506`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 8` with `1` pending effects
- Action summary: `{"chosen_initiatives": ["launch_referral_loop"], "messaging_action": "retention_campaign", "pricing_change_pct": 0.0, "support_policy": "balanced_triage"}`

### Step 7

- Reward: `-0.0507`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 9` with `0` pending effects
- Action summary: `{"chosen_initiatives": ["launch_referral_loop"], "messaging_action": "retention_campaign", "pricing_change_pct": 0.0, "support_policy": "balanced_triage"}`

### Step 8

- Reward: `-0.0513`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 10` with `0` pending effects
- Action summary: `{"chosen_initiatives": ["launch_referral_loop"], "messaging_action": "retention_campaign", "pricing_change_pct": 0.0, "support_policy": "balanced_triage"}`

### Step 9

- Reward: `-0.0515`
- Strict step: `True`
- Empty fallback: `False`
- Cap hit: `False`
- Observation: `Day 11` with `0` pending effects
- Action summary: `{"chosen_initiatives": ["launch_referral_loop"], "messaging_action": "retention_campaign", "pricing_change_pct": 0.0, "support_policy": "balanced_triage"}`
