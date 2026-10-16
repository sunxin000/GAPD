"""
KBQA-R1 custom reward function for verl 0.8.0.

Dispatches reward computation based on data_source:
- MID-based datasets (webqsp, grailqa, graphq): F1 score on entity MIDs
- QA datasets (nq, triviaqa, etc.): Exact match scoring

This file is referenced by custom_reward_function.path in the training script.
The reward_kwargs (structure_format_score, mid_f1_weight, function_reward_weight)
are injected by verl's get_custom_reward_fn wrapper via custom_reward_function.reward_kwargs.
"""

from kbqa_r1.utils.reward_score import mid_reward, qa_em

MID_DATASETS = {'webqsp', 'grailqa', 'graphq'}
QA_DATASETS = {'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'}


def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  structure_format_score=0.0, mid_f1_weight=1.0,
                  function_reward_weight=0.0, **kwargs):
    if data_source in MID_DATASETS:
        result = mid_reward.compute_mid_reward(
            solution_str=solution_str,
            ground_truth=ground_truth,
            structure_format_score=structure_format_score,
        )
        score = result['total'] * mid_f1_weight
        return {
            'score': score,
            'mid_f1': result['mid_f1'],
            'structure_reward': result['structure_reward'],
        }
    elif data_source in QA_DATASETS:
        score = qa_em.compute_score_em(solution_str, ground_truth)
        return score
    else:
        return 0.0
