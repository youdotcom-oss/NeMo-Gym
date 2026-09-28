import json


def stats(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
        n = len(rows)
        if n == 0:
            return None
        rewards = [r.get('reward',0) for r in rows]
        tool_calls = [r.get('num_tool_calls',0) for r in rows]
        resets = [r.get('reset_count',0) for r in rows]
        tokens = [r.get('response',{}).get('usage',{}).get('total_tokens',0) for r in rows]
        tokens = [t for t in tokens if t]
        correct_tool_calls = [r.get('num_tool_calls',0) for r in rows if r.get('reward',0)==1.0]
        return { 'n': n, 'n_correct': sum(1 for r in rewards if r == 1.0), 'mean_reward': sum(rewards)/n, 'avg_tool_calls': sum(tool_calls)/n, 'avg_resets': sum(resets)/n, 'avg_tokens': sum(tokens)/len(tokens) if tokens else 0, 'max_steps_correct': max(correct_tool_calls) if correct_tool_calls else None, }


print(
    "You.com baseline with Base model, 65k context, progress off. Use to compare against tavily",
    stats(
        'results/base_model_interleaved_on_65k_reset_300_steps_force_answer_and_considered_answer_test_progress_false_full_run/evaluator_rollouts.jsonl'
    )
)
print(
    "FULL RL + SFT 65k context, progress ON",
    stats(
        'results/sft_rl_model_interleaved_on_65k_reset_300_steps_force_answer_and_considered_answer_test_progress_true_full_run/evaluator_rollouts.jsonl'
    )
)
print(
    "FULL RL + SFT 65k context, progress off, phase temp/top p",
    stats(
        'results/sft_rl_model_interleaved_on_65k_reset_300_steps_force_answer_and_considered_answer_test_progress_false_new_temps_full_run/evaluator_rollouts.jsonl'
    )
)
print(
    "Tavily Default Baseline",
    stats(
        'results/default_settings_tavily_baseline/evaluator_rollouts.jsonl'
    )
)
print(
    "FULL RL + SFT 65k context, progress off (accidentally overrode results, logs might be salvageable)",
    "{'n': 1263, 'n_correct': 574, 'mean_reward': 0.45447347585114806, 'avg_tool_calls': 115.1076801266825, 'avg_resets': 3.8551068883610453, 'avg_tokens': 3859938.2367379256, 'max_steps_correct': 255}"
)
print(
    "FULL RL + SFT 50k context, progress off",
    stats(
        'results/sft_rl_model_interleaved_on_50k_reset_300_steps_force_answer_and_considered_answer_test_progress_false_full_run/evaluator_rollouts.jsonl'
    )
)
print(
    "FULL RL only 50k context, progress off",
    stats(
        "results/rl_model_interleaved_on_50k_reset_300_steps_num_full/evaluator_rollouts.jsonl"
    )
)
print("65k context, progress off", stats(
    f'results/sft_rl_model_interleaved_on_65k_reset_300_steps_force_answer_and_considered_answer_test_progress_false/evaluator_rollouts.jsonl'))
print("50k context, progress off", stats(
    f'results/sft_rl_model_interleaved_on_50k_reset_300_steps_force_answer_and_considered_answer_test_progress_false/evaluator_rollouts.jsonl'))
print("50k context, progress on", stats(
    f'results/sft_rl_model_interleaved_on_50k_reset_300_steps_force_answer_and_considered_answer_test_progress/evaluator_rollouts.jsonl'))
print("65k context, progress on", stats(
    f'results/sft_rl_model_interleaved_on_65k_reset_300_steps_force_answer_and_considered_answer_test_progress_true_v2/evaluator_rollouts.jsonl'))