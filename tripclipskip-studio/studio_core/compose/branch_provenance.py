"""WP-03B1b: conservative A1111 primary-conditioning call identification.

Labels are *construction scope*, never proof of token content, final branch,
pooled origin or consumption by the denoiser. No encoder forward or A1111
imports are performed here. Unknown user/custom call paths remain unknown.
"""
from __future__ import annotations

from dataclasses import dataclass
import sys


SCOPE_EVIDENCE = 'A1111_SETUP_GET_CONDS_FUNCTION_AND_SD_CONDITIONING_FLAGS'
UNVERIFIED_EVIDENCE = 'NO_MATCHED_SCOPED_A1111_GET_CONDS_CALL'


@dataclass(frozen=True)
class ConstructionScope:
    scope_id: str
    branch: str
    pass_name: str
    evidence: str = SCOPE_EVIDENCE
    pass_evidence: str = 'IS_HR_PASS_FLAG_ONLY'


def classify_construction_request(function, required_prompts):
    """Recognize only exact A1111 parser callables + its SdConditioning.

    The caller must additionally verify that we're inside this processing
    object's setup_conds(). This is deliberately not a function-name or
    call-order heuristic. Doesn't read prompt strings, identifiers or weights.
    """
    parser = sys.modules.get('modules.prompt_parser')
    if parser is None:
        return None
    prompt_class = getattr(parser, 'SdConditioning', None)
    if not isinstance(prompt_class, type) or not isinstance(required_prompts, prompt_class):
        return None
    try:
        # A1111 SdConditioning is a list subclass with a plain bool field.
        # Do not invoke properties or arbitrary repr/str on unknown inputs.
        negative = object.__getattribute__(required_prompts, '__dict__').get('is_negative_prompt')
    except (AttributeError, TypeError):
        return None
    if type(negative) is not bool:
        return None
    if function is getattr(parser, 'get_learned_conditioning', None) and negative:
        return 'negative'
    if function is getattr(parser, 'get_multicond_learned_conditioning', None) and not negative:
        return 'positive'
    return None


def classify_construction_pass(processing, args, kwargs):
    """Label base/hires from A1111 cache *slot identity* when available.

    Both upstream A1111 and the provided custom fork can precompute Hires.fix
    conditioning inside base setup_conds() while is_hr_pass is False. The
    first cache slot is cached_hr_c/uc for hires and cached_c/uc for base.
    No prompt strings, cache contents, or computed tensors are inspected.
    Unknown/aliased custom cache slots fail closed to ``unknown``.
    """
    caches = args[3] if len(args) > 3 else kwargs.get('caches')
    if type(caches) in (list, tuple) and caches:
        try:
            attrs = object.__getattribute__(processing, '__dict__')
            first = caches[0]
            if type(attrs) is not dict:
                return 'unknown', 'CACHE_SLOT_UNRECOGNIZED'
            base = any(first is attrs[key] for key in ('cached_c', 'cached_uc')
                       if key in attrs and attrs[key] is not None)
            hires = any(first is attrs[key] for key in ('cached_hr_c', 'cached_hr_uc')
                        if key in attrs and attrs[key] is not None)
            if base and hires:
                return 'unknown', 'CACHE_SLOT_AMBIGUOUS'
            if hires:
                return 'hires', 'A1111_CACHE_SLOT_IDENTITY_HIRES'
            if base:
                return 'base', 'A1111_CACHE_SLOT_IDENTITY_BASE'
        except (AttributeError, KeyError, TypeError):
            pass
        return 'unknown', 'CACHE_SLOT_UNRECOGNIZED'
    try:
        flag = getattr(processing, 'is_hr_pass', None)
    except Exception:
        flag = None
    if flag is True:
        return 'hires', 'IS_HR_PASS_FLAG_ONLY'
    if flag is False:
        return 'base', 'IS_HR_PASS_FLAG_ONLY'
    return 'unknown', 'NO_VERIFIED_PASS_SIGNAL'


def scope_metadata(scope):
    if scope is None:
        return {
            'construction_branch': 'unknown',
            'construction_branch_evidence': UNVERIFIED_EVIDENCE,
            'construction_scope_id': None,
            'construction_pass': 'unknown',
            'construction_pass_evidence': 'NO_MATCHED_SCOPED_A1111_GET_CONDS_CALL',
            'construction_scope_semantics': 'NOT_VERIFIED',
        }
    return {
        'construction_branch': scope.branch,
        'construction_branch_evidence': scope.evidence,
        'construction_scope_id': scope.scope_id,
        'construction_pass': scope.pass_name,
        'construction_pass_evidence': scope.pass_evidence,
        'construction_scope_semantics': 'CALL_CONTEXT_ONLY_NOT_FINAL_CONDITIONING_PROVENANCE',
    }
