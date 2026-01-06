import argparse
import json
import os
from dataclasses import asdict
from typing import List, Optional, Dict, Any, Tuple

from code_extract import (
    extract_functions,
    FunctionRecord,
)
from rfc import extract_srs_from_rfc_text, SRRecord
from keywords import load_protocol_keywords, match_keywords
from llm import (
    require_openai_key,
    load_text,
    openai_json,
    openai_chat_json,
    gather_limited,
    get_async_openai_client,
)
import asyncio


def _default_final_root() -> str:
    """
    Legacy support: allow pointing to the old `RFC/final`-style directory.
    Prefer env `RFC_FINAL_ROOT`; otherwise try a few common repo layouts.
    """
    env = os.environ.get("RFC_FINAL_ROOT", "").strip()
    if env and os.path.exists(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))  # .../code/rfcflow
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    cands = [
        os.path.join(repo_root, "RFC", "final"),
        os.path.join(repo_root, "final"),
    ]
    for cand in cands:
        if os.path.exists(cand):
            return cand
    return ""


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))  # .../code/rfcflow
    return os.path.abspath(os.path.join(here, "..", ".."))


def _default_raw_rfc_dir() -> str:
    """
    Auto-detect where RFC .txt files live.
    Priority:
      1) env RFC_RAW_RFC_DIR
      2) legacy RFC_FINAL_ROOT/raw_rfc
      3) anonrepo layout: data/rfc/rawrfc
    """
    env = os.environ.get("RFC_RAW_RFC_DIR", "").strip()
    if env and os.path.exists(env):
        return env
    final_root = _default_final_root()
    if final_root:
        cand = os.path.join(final_root, "raw_rfc")
        if os.path.exists(cand):
            return cand
    repo_root = _repo_root()
    cand = os.path.join(repo_root, "data", "rfc", "rawrfc")
    return cand if os.path.exists(cand) else ""


def _default_code_root(project: str) -> str:
    """
    Auto-detect source code root for a project.
    Priority:
      1) legacy RFC_FINAL_ROOT layout: data/<project>/source_code
      2) anonrepo layout: data/repos/<project>/(rawdata|rawcode)
    """
    final_root = _default_final_root()
    if final_root:
        project_to_rel = {
            "httpd": os.path.join("data", "httpd", "source_code"),
            "boringssl": os.path.join("data", "boringssl", "source_code"),
            "nginx": os.path.join("data", "nginx", "source_code"),
            "openssl": os.path.join("data", "openssl", "source_code"),
            "frr": os.path.join("data", "FRR", "frr"),
            "bird": os.path.join("data", "BIRD", "bird"),
        }
        rel = project_to_rel.get(project)
        if rel:
            cand = os.path.join(final_root, rel)
            if os.path.exists(cand):
                return cand

    repo_root = _repo_root()
    # anonrepo layout
    if project == "httpd":
        cand = os.path.join(repo_root, "data", "repos", "httpd", "rawcode")
        return cand if os.path.exists(cand) else ""
    cand = os.path.join(repo_root, "data", "repos", project, "rawdata")
    return cand if os.path.exists(cand) else ""


def _default_libclang() -> Optional[str]:
    """
    Best-effort libclang path detection:
    - env `LIBCLANG_PATH`
    - common local path used by original experiments
    """
    env = os.environ.get("LIBCLANG_PATH", "").strip()
    if env and os.path.exists(env):
        return env
    cand = "/data/a/ykw/local/clang/lib/libclang.so"
    if os.path.exists(cand):
        return cand
    return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: str, header: List[str], rows: List[dict]) -> None:
    import csv

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def cmd_extract_sr(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)

    # --- RFC -> SR ---
    if not args.rfc_id:
        raise ValueError("Provide --rfc-id (e.g. 9110/8446).")
    raw_dir = _default_raw_rfc_dir()
    if not raw_dir:
        raise RuntimeError(
            "Cannot auto-detect RFC raw dir. Provide env RFC_RAW_RFC_DIR, "
            "or set RFC_FINAL_ROOT to a legacy layout, "
            "or keep `data/rfc/rawrfc/` in the repo."
        )
    rfc_path = os.path.join(raw_dir, f"rfc{args.rfc_id}.txt")
    if not os.path.exists(rfc_path):
        raise FileNotFoundError(f"RFC file not found: {rfc_path}")

    with open(rfc_path, "r", encoding="utf-8", errors="ignore") as f:
        rfc_text = f.read()

    srs: List[SRRecord] = extract_srs_from_rfc_text(
        rfc_text,
        rfc_id=args.rfc_id,
        max_srs=args.max_srs,
    )

    sr_jsonl = os.path.join(args.out_dir, "sr.jsonl")
    write_jsonl(sr_jsonl, [asdict(x) for x in srs])

    # Also write a light CSV for inspection
    sr_csv = os.path.join(args.out_dir, "sr.csv")
    write_csv(
        sr_csv,
        header=[
            "sr_id",
            "sr",
            "keywords",
            "section_number",
            "section_title",
            "section_level",
            "para_index",
            "sentence_index",
            "current_para",
            "prev_para",
            "next_para",
            "context_before",
            "context_after",
        ],
        rows=[asdict(x) for x in srs],
    )
    print(f"[ok] wrote SR: {sr_jsonl} , {sr_csv} (count={len(srs)})")
    print("[done] extract-sr finished")


def cmd_extract_functions(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)

    if not args.project:
        raise ValueError("Provide --project.")
    code_root = _default_code_root(args.project)
    if not os.path.exists(code_root):
        raise FileNotFoundError(
            f"code_root not found for project={args.project}. "
            f"Expected legacy layout via RFC_FINAL_ROOT or anonrepo layout under data/repos/. "
            f"Resolved code_root={code_root or '<empty>'}"
        )

    # Defaults: prefer clang+deps(full) when available; fallback to regex otherwise.
    max_files = 2000
    max_functions = 5000
    file_patterns = ["*.c", "*.h", "*.cc", "*.cpp", "*.cxx", "*.hpp"]
    include_dirs, extra_args = _auto_project_includes_and_args(args.project, code_root, _repo_root())

    try:
        from code_analysis import extract_functions_with_deps  # lazy import

        libclang = _default_libclang()
        if not libclang:
            raise RuntimeError("libclang not found")
        funcs: List[FunctionRecord] = extract_functions_with_deps(
            code_root=code_root,
            max_files=max_files,
            max_functions=max_functions,
            libclang_path=libclang,
            include_dirs=include_dirs,
            extra_args=extra_args,
            file_patterns=file_patterns,
        )
    except Exception:
        funcs = extract_functions(
            code_root=code_root,
            backend="regex",
            max_files=max_files,
            max_functions=max_functions,
        )
    func_jsonl = os.path.join(args.out_dir, "functions.jsonl")
    write_jsonl(func_jsonl, [asdict(x) for x in funcs])

    func_csv = os.path.join(args.out_dir, "functions.csv")
    write_csv(
        func_csv,
        header=[
            "function_id",
            "function_name",
            "file_path",
            "start_line",
            "end_line",
            "return_type",
            "condition",
        ],
        rows=[asdict(x) for x in funcs],
    )
    print(f"[ok] wrote functions: {func_jsonl} , {func_csv} (count={len(funcs)})")
    print("[done] extract-functions finished")


def _read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _index_by_int_id(rows: List[dict], key: str) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for r in rows:
        try:
            out[int(r.get(key))] = r
        except Exception:
            continue
    return out


def _load_group_info_subcat_to_srs(path: str) -> Dict[str, List[int]]:
    """
    group_info json layout:
      { "A": { "A1": [sr_id...], ... }, "B": { ... } }
    Return:
      { "A1": [..], "B2": [..], ... }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, List[int]] = {}
    if not isinstance(data, dict):
        return out
    for _, sub in data.items():
        if not isinstance(sub, dict):
            continue
        for subcat, ids in sub.items():
            if not isinstance(subcat, str) or not isinstance(ids, list):
                continue
            cleaned: List[int] = []
            for x in ids:
                try:
                    cleaned.append(int(x))
                except Exception:
                    continue
            out[subcat] = cleaned
    return out


def _auto_project_includes_and_args(project: str, code_root: str, repo_root: str) -> Tuple[List[str], List[str]]:
    """
    Best-effort auto include dirs / clang args, aligned with the original `final/src/code_analyzer.py` presets.
    Only returns paths that exist on the current machine.
    """
    incs: List[str] = []
    args: List[str] = []

    def add_inc(p: str) -> None:
        if p and os.path.exists(p) and p not in incs:
            incs.append(p)

    def add_arg(a: str) -> None:
        if a and a not in args:
            args.append(a)

    code_root = os.path.abspath(code_root) if code_root else ""
    repo_root = os.path.abspath(repo_root) if repo_root else ""

    if project == "httpd":
        # `final/src/code_analyzer.py` defaulted to C11 + USE_SSL (plus lots of system includes)
        add_arg("--std=c11")
        add_arg("-DUSE_SSL")

        # repo/internal includes
        add_inc(os.path.join(code_root, "include"))
        add_inc(os.path.join(code_root, "os", "unix"))
        add_inc(os.path.join(code_root, "modules", "http"))
        add_inc("/data/a/ykw/httpd/include")
        # APR headers (environment-specific; only added if present)
        add_inc("/data/a/ykw/build/httpd-2.4.62/srclib/apr/include")
        add_inc("/data/a/ykw/build/httpd-2.4.62/srclib-util/include")

    elif project == "boringssl":
        add_inc(os.path.join(code_root, "include"))
        add_inc(os.path.join(code_root, "include", "openssl"))

    elif project == "nginx":
        add_inc(os.path.join(code_root, "src"))
        add_inc(os.path.join(code_root, "src", "core"))
        add_inc(os.path.join(code_root, "src", "event"))
        add_inc(os.path.join(code_root, "src", "http"))
        add_inc(os.path.join(code_root, "src", "os"))

    elif project == "openssl":
        add_inc(os.path.join(code_root, "include"))

    elif project == "frr":
        add_inc(os.path.join(code_root, "include"))

    elif project == "bird":
        add_inc(code_root)

    return incs, args


def cmd_filter(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    # Minimal UX: always read functions from out_dir
    funcs = _read_jsonl(os.path.join(args.out_dir, "functions.jsonl"))
    # Optional safety valve for smoke tests / cost control (does not change CLI surface):
    # RFCFLOW_MAX_ITEMS=N will cap items processed in this step.
    try:
        max_items_env = int(os.environ.get("RFCFLOW_MAX_ITEMS", "0") or "0")
    except Exception:
        max_items_env = 0
    if max_items_env > 0:
        funcs = funcs[:max_items_env]

    if args.mode == "quick":
        kws = load_protocol_keywords(args.protocol, assets_dir=assets_dir)
        kept = []
        for r in funcs:
            code = r.get("source_code", "")
            matched = match_keywords(code, kws)
            if len(matched) >= 1:
                r["matched_keywords"] = matched[:200]
                kept.append(r)
        out = os.path.join(args.out_dir, "functions_filtered.jsonl")
        write_jsonl(out, kept)
        print(f"[ok] wrote filtered functions: {out} (kept={len(kept)}/{len(funcs)})")
        return

    # openai mode (paper-compatible)
    require_openai_key()

    async def _run() -> None:
        # Match original protocol_filter.py behavior: system prompt + user prompt prefix + code
        sys_path = os.path.join(assets_dir, "prompts", f"{args.protocol}_prompt.txt")
        user_path = os.path.join(assets_dir, "prompts", f"{args.protocol}_user_prompt.txt")
        sys_prompt = load_text(sys_path)
        user_prefix = load_text(user_path)
        client = get_async_openai_client()
        model = "o3-mini"
        workers = 20

        async def _one(r):
            user_prompt = user_prefix + r.get("source_code", "")
            # Some models error if the response hits the output limit; retry once with a higher cap.
            try:
                data = await openai_chat_json(client, model, sys_prompt, user_prompt, max_tokens=800)
            except Exception as e:
                msg = str(e).lower()
                if "output limit was reached" in msg or "max_tokens" in msg and "reached" in msg:
                    data = await openai_chat_json(client, model, sys_prompt, user_prompt, max_tokens=1600)
                else:
                    raise
            # original keys: Relevance/Reason (sometimes as string)
            rel_val = data.get("Relevance", data.get("relevance", True))
            rel = str(rel_val).lower() == "true" if isinstance(rel_val, str) else bool(rel_val)
            r["relevance"] = rel
            r["reason"] = data.get("Reason", data.get("reason", ""))
            return r if rel else None

        coros = [_one(r) for r in funcs]
        res = await gather_limited(coros, limit=workers)
        kept = [x for x in res if x]
        out = os.path.join(args.out_dir, "functions_filtered.jsonl")
        write_jsonl(out, kept)
        print(f"[ok] wrote filtered functions: {out} (kept={len(kept)}/{len(funcs)})")

    asyncio.run(_run())


def cmd_dual_filter(args: argparse.Namespace) -> None:
    """
    dual_filter:
      input: functions (after filter1), i.e. functions_filtered.jsonl
      output: dual_filter.jsonl with:
        - keyword extraction result (protocol-specific)
        - semantic subcategory classification: matched_subcategories (A1/B2/...)
        - matching_sr_ids: union of sr_id lists from group_info for those subcategories
    """
    ensure_dir(args.out_dir)
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    require_openai_key()

    funcs = _read_jsonl(os.path.join(args.out_dir, "functions_filtered.jsonl"))

    if args.protocol == "http":
        kw_path = os.path.join(assets_dir, "prompts", "kw_filter.txt")
        nonkw_path = os.path.join(assets_dir, "prompts", "nonkw_filter.txt")
        group_info_path = os.path.join(assets_dir, "group_info", "http_group_info.json")
    else:
        kw_path = os.path.join(assets_dir, "prompts", "tls_kw.txt")
        nonkw_path = os.path.join(assets_dir, "prompts", "tls_nonkw.txt")
        group_info_path = os.path.join(assets_dir, "group_info", "tls_group_info.json")

    kw_prompt = load_text(kw_path)
    nonkw_prompt = load_text(nonkw_path)
    subcat_to_srids = _load_group_info_subcat_to_srs(group_info_path)

    async def _run() -> None:
        client = get_async_openai_client()
        model = "o3-mini"
        workers = 20

        async def _one(r: dict) -> dict:
            code = r.get("source_code", "")
            user = f"```\n{code}\n```"
            out = dict(r)
            out["dual_filter_protocol"] = args.protocol

            try:
                kw_data = await openai_chat_json(client, model, kw_prompt, user, max_tokens=1000)
            except Exception as e:
                kw_data = {"_error": f"kw_filter_failed: {e}"}

            try:
                nonkw_data = await openai_chat_json(client, model, nonkw_prompt, user, max_tokens=1000)
            except Exception as e:
                nonkw_data = {"matched_subcategories": [], "_error": f"nonkw_filter_failed: {e}"}

            matched = nonkw_data.get("matched_subcategories", [])
            if not isinstance(matched, list):
                matched = []
            matched = [str(x).strip() for x in matched if str(x).strip()]
            # de-dup preserve order
            seen: set = set()
            matched = [x for x in matched if not (x in seen or seen.add(x))]

            sr_ids: List[int] = []
            sr_seen: set = set()
            for subcat in matched:
                for sid in subcat_to_srids.get(subcat, []):
                    if sid in sr_seen:
                        continue
                    sr_seen.add(sid)
                    sr_ids.append(sid)
                    # no cap by default (do not change candidate set)

            out["dual_filter_keywords"] = kw_data
            out["matched_subcategories"] = matched
            out["matching_sr_ids"] = sr_ids
            return out

        res = await gather_limited([_one(r) for r in funcs], limit=workers)
        out_path = os.path.join(args.out_dir, "dual_filter.jsonl")
        write_jsonl(out_path, res)
        print(f"[ok] wrote dual_filter results: {out_path} (count={len(res)})")

    asyncio.run(_run())


def cmd_extract_constraints(args: argparse.Namespace) -> None:
    """
    Extract (conditions/actions) for each SR using an LLM, so final_verify can
    populate {spec_constraints}.
    """
    ensure_dir(args.out_dir)
    require_openai_key()
    srs = _read_jsonl(os.path.join(args.out_dir, "sr.jsonl"))
    # Optional safety valve for smoke tests / cost control (does not change CLI surface):
    # RFCFLOW_MAX_ITEMS=N will cap SR items processed in this step.
    try:
        max_items_env = int(os.environ.get("RFCFLOW_MAX_ITEMS", "0") or "0")
    except Exception:
        max_items_env = 0
    if max_items_env > 0:
        srs = srs[:max_items_env]

    # a light prompt (ported from final/src/constrain_extract.py intent)
    sys_prompt = (
        "You are an expert of protocol. Extract and identify two structured semantic components from the provided "
        "Specification Requirement (SR) and its context: Trigger Condition(s) and Required Action(s). "
        "Trigger conditions may be empty; actions must not be empty. "
        "Return JSON only with keys: conditionlist (array of strings), actionlist (array of strings)."
    )

    async def _run() -> None:
        client = get_async_openai_client()
        model = "o3-mini"
        workers = 20

        async def _one(sr: dict) -> dict:
            sr_id = int(sr.get("sr_id", -1))
            sr_text = str(sr.get("sr", ""))
            before = str(sr.get("context_before", ""))
            after = str(sr.get("context_after", ""))
            context = ""
            if before:
                context += f"Previous paragraph:\n{before}\n\n"
            context += f"SR:\n{sr_text}\n\n"
            if after:
                context += f"Next paragraph:\n{after}\n"
            user = f"Here are the SR and its context:\n{context}"

            data: Dict[str, Any]
            try:
                data = await openai_chat_json(client, model, sys_prompt, user, max_tokens=600)
            except Exception as e:
                data = {"conditionlist": [], "actionlist": [], "_error": str(e)}

            conds = data.get("conditionlist", [])
            acts = data.get("actionlist", [])
            if not isinstance(conds, list):
                conds = []
            if not isinstance(acts, list):
                acts = []
            return {
                "sr_id": sr_id,
                "conditions": [str(x).strip() for x in conds if str(x).strip()],
                "actions": [str(x).strip() for x in acts if str(x).strip()],
            }

        res = await gather_limited([_one(sr) for sr in srs], limit=workers)
        out_path = os.path.join(args.out_dir, "sr_constraints.jsonl")
        write_jsonl(out_path, res)
        print(f"[ok] wrote SR constraints: {out_path} (count={len(res)})")

    asyncio.run(_run())


def _format_spec_constraints(conditions: List[str], actions: List[str]) -> str:
    cond_lines = [f"C{i+1}: {c}" for i, c in enumerate(conditions or [])]
    act_lines = [f"A{i+1}: {a}" for i, a in enumerate(actions or [])]
    return f"Conditions: {', '.join(cond_lines)}\nActions: {', '.join(act_lines)}"


def cmd_pre_verify(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    require_openai_key()

    funcs = _read_jsonl(os.path.join(args.out_dir, "dual_filter.jsonl"))
    srs = _read_jsonl(os.path.join(args.out_dir, "sr.jsonl"))
    sr_by_id = _index_by_int_id(srs, "sr_id")

    prompt_path = os.path.join(assets_dir, "prompts", f"{args.protocol}_pre_verify.txt")
    sys_prompt = load_text(prompt_path)

    # Optional safety valve for smoke tests / cost control (does not change CLI surface):
    # RFCFLOW_MAX_PAIRS=N caps (func,sr) pairs processed in this step.
    try:
        max_pairs_env = int(os.environ.get("RFCFLOW_MAX_PAIRS", "0") or "0")
    except Exception:
        max_pairs_env = 0

    # build (func, sr) pairs
    pairs: List[Tuple[dict, int]] = []
    for fn in funcs:
        fn_sr_ids = fn.get("matching_sr_ids", [])
        if not isinstance(fn_sr_ids, list):
            continue
        for sid in fn_sr_ids:
            try:
                sr_id = int(sid)
            except Exception:
                continue
            if sr_id not in sr_by_id:
                continue
            pairs.append((fn, sr_id))
            if max_pairs_env > 0 and len(pairs) >= max_pairs_env:
                break
        if max_pairs_env > 0 and len(pairs) >= max_pairs_env:
            break

    async def _run() -> None:
        client = get_async_openai_client()
        model = "o3-mini"
        workers = 20
        retries = 2

        # keep retries internal for end users (don't expose noisy CLI flag)
        async def _call_with_retry(system_prompt: str, user_prompt: str, max_tokens: int) -> Dict[str, Any]:
            last_err: Optional[Exception] = None
            for _ in range(max(1, retries + 1)):
                try:
                    return await openai_chat_json(client, args.model, system_prompt, user_prompt, max_tokens=max_tokens)
                except Exception as e:
                    last_err = e
            raise last_err  # type: ignore[misc]

        async def _one(fn: dict, sr_id: int) -> dict:
            sr = sr_by_id[sr_id]
            sr_text = str(sr.get("sr", ""))
            before = str(sr.get("context_before", ""))
            after = str(sr.get("context_after", ""))
            sr_context_parts = []
            if before:
                sr_context_parts.append(before)
            # include SR sentence itself as anchor
            sr_context_parts.append(sr_text)
            if after:
                sr_context_parts.append(after)
            sr_context = "\n\n".join([p for p in sr_context_parts if p])

            code = str(fn.get("source_code", ""))
            user = (
                "Function code:\n```\n"
                + code
                + "\n```\n\n"
                + f"SR: {sr_text}\n\nSR Context: {sr_context}"
            )

            try:
                data = await _call_with_retry(sys_prompt, user, max_tokens=800)
                is_match = data.get("is_match", data.get("match", True))
                is_match = str(is_match).lower() == "true" if isinstance(is_match, str) else bool(is_match)
                expl = data.get("verify_explanaton", data.get("explanation", ""))
                status = "success"
                err = None
            except Exception as e:
                # Conservative: don't drop candidate due to transient errors.
                is_match = True
                expl = ""
                status = "failed"
                err = str(e)

            return {
                "function_id": fn.get("function_id"),
                "function_name": fn.get("function_name", ""),
                "sr_id": sr_id,
                "sr": sr_text,
                "is_match": is_match,
                "verify_explanation": expl,
                "status": status,
                "error": err,
            }

        res = await gather_limited([_one(fn, sid) for fn, sid in pairs], limit=workers)
        out_path = os.path.join(args.out_dir, "pre_verify.jsonl")
        write_jsonl(out_path, res)
        kept = sum(1 for r in res if r.get("is_match") is True)
        print(f"[ok] wrote pre_verify results: {out_path} (pairs={len(res)}, kept={kept})")

    asyncio.run(_run())


def cmd_final_verify(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    require_openai_key()

    pre = _read_jsonl(os.path.join(args.out_dir, "pre_verify.jsonl"))
    # only keep matches
    pre = [r for r in pre if bool(r.get("is_match"))]
    try:
        max_pairs_env = int(os.environ.get("RFCFLOW_MAX_PAIRS", "0") or "0")
    except Exception:
        max_pairs_env = 0
    if max_pairs_env > 0:
        pre = pre[:max_pairs_env]

    funcs = _read_jsonl(os.path.join(args.out_dir, "dual_filter.jsonl"))
    fn_by_id = _index_by_int_id(funcs, "function_id")

    srs = _read_jsonl(os.path.join(args.out_dir, "sr.jsonl"))
    sr_by_id = _index_by_int_id(srs, "sr_id")

    constraints = _read_jsonl(os.path.join(args.out_dir, "sr_constraints.jsonl"))
    constraints_by_id = _index_by_int_id(constraints, "sr_id")

    prompt_path = os.path.join(assets_dir, "prompts", f"{args.protocol}_final_verify.txt")
    verify_prompt_template = load_text(prompt_path)
    sys_prompt = "You are a professional protocol expert and code verification assistant."

    async def _run() -> None:
        client = get_async_openai_client()
        model = "o3-mini"
        workers = 10
        retries = 2

        async def _call_with_retry(system_prompt: str, user_prompt: str, max_tokens: int) -> Dict[str, Any]:
            last_err: Optional[Exception] = None
            for _ in range(max(1, retries + 1)):
                try:
                    return await openai_chat_json(client, model, system_prompt, user_prompt, max_tokens=max_tokens)
                except Exception as e:
                    last_err = e
            raise last_err  # type: ignore[misc]

        async def _one(r: dict) -> dict:
            function_id = int(r.get("function_id"))
            sr_id = int(r.get("sr_id"))
            fn = fn_by_id.get(function_id, {})
            sr = sr_by_id.get(sr_id, {})
            cons = constraints_by_id.get(sr_id, {})

            sr_text = str(sr.get("sr", r.get("sr", "")))
            before = str(sr.get("context_before", ""))
            after = str(sr.get("context_after", ""))
            sr_context = " ".join([x for x in [before, sr_text, after] if x])

            conditions = cons.get("conditions", []) if isinstance(cons, dict) else []
            actions = cons.get("actions", []) if isinstance(cons, dict) else []
            if not isinstance(conditions, list):
                conditions = []
            if not isinstance(actions, list):
                actions = []

            spec_constraints = _format_spec_constraints(
                [str(x) for x in conditions if str(x).strip()],
                [str(x) for x in actions if str(x).strip()],
            )
            function_body = str(fn.get("source_code", ""))
            dependencies_obj = fn.get("dependencies", {})
            if not isinstance(dependencies_obj, dict):
                dependencies_obj = {}
            dependencies = json.dumps(dependencies_obj, ensure_ascii=False)

            prompt = verify_prompt_template
            prompt = prompt.replace("{spec}", sr_text)
            prompt = prompt.replace("{spec_constraints}", spec_constraints)
            prompt = prompt.replace("{sr_context}", sr_context)
            prompt = prompt.replace("{function_body}", function_body)
            prompt = prompt.replace("{dependencies}", dependencies)

            try:
                data = await _call_with_retry(sys_prompt, prompt, max_tokens=1200)
                status = "success"
                err = None
            except Exception as e:
                data = {"outcome": "Error", "met_constrains": [], "OverallExplanation": str(e)}
                status = "failed"
                err = str(e)

            return {
                "function_id": function_id,
                "function_name": fn.get("function_name", r.get("function_name", "")),
                "sr_id": sr_id,
                "sr": sr_text,
                "outcome": data.get("outcome", ""),
                "met_constrains": data.get("met_constrains", []),
                "OverallExplanation": data.get("OverallExplanation", data.get("overall_explanation", "")),
                "status": status,
                "error": err,
            }

        res = await gather_limited([_one(x) for x in pre], limit=workers)
        out_path = os.path.join(args.out_dir, "final_verify.jsonl")
        write_jsonl(out_path, res)
        print(f"[ok] wrote final_verify results: {out_path} (pairs={len(res)})")

    asyncio.run(_run())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="rfcflow: minimal runnable research code for RFC→SR + code parsing + filter1 + dual_filter + verification."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sr = sub.add_parser("extract-sr", help="RFC text -> SR extraction (paper-compatible).")
    p_sr.add_argument("--rfc-id", required=True, help="RFC id (e.g., 9110/8446).")
    p_sr.add_argument("--out-dir", required=True, help="Output directory.")
    p_sr.add_argument("--max-srs", type=int, default=2000, help="Cap SR count for quick runs.")
    p_sr.set_defaults(func=cmd_extract_sr)

    p_code = sub.add_parser("extract-functions", help="code root -> function extraction (regex/clang).")
    p_code.add_argument("--project", choices=["httpd", "boringssl", "nginx", "openssl", "frr", "bird"], required=True)
    p_code.add_argument("--out-dir", required=True, help="Output directory.")
    p_code.set_defaults(func=cmd_extract_functions)

    p_f = sub.add_parser("filter", help="filter1: protocol relevance filter for functions (reads out_dir/functions.jsonl).")
    p_f.add_argument("--protocol", choices=["http", "tls"], required=True)
    p_f.add_argument("--out-dir", required=True)
    p_f.add_argument("--mode", choices=["openai", "quick"], default="openai")
    p_f.set_defaults(func=cmd_filter)

    p_df = sub.add_parser("dual-filter", help="dual_filter (reads out_dir/functions_filtered.jsonl).")
    p_df.add_argument("--protocol", choices=["http", "tls"], required=True)
    p_df.add_argument("--out-dir", required=True)
    p_df.set_defaults(func=cmd_dual_filter)

    p_c = sub.add_parser("extract-constraints", help="SR -> (conditions/actions) (reads out_dir/sr.jsonl).")
    p_c.add_argument("--protocol", choices=["http", "tls"], required=True)
    p_c.add_argument("--out-dir", required=True)
    p_c.set_defaults(func=cmd_extract_constraints)

    p_pv = sub.add_parser("pre-verify", help="pre_verify (reads out_dir/dual_filter.jsonl + out_dir/sr.jsonl).")
    p_pv.add_argument("--protocol", choices=["http", "tls"], required=True)
    p_pv.add_argument("--out-dir", required=True)
    p_pv.set_defaults(func=cmd_pre_verify)

    p_fv = sub.add_parser("final-verify", help="final_verify (reads out_dir/pre_verify.jsonl + out_dir/dual_filter.jsonl + out_dir/sr*.jsonl).")
    p_fv.add_argument("--protocol", choices=["http", "tls"], required=True)
    p_fv.add_argument("--out-dir", required=True)
    p_fv.set_defaults(func=cmd_final_verify)

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()


