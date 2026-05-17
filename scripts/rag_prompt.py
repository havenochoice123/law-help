"""Prompt builders shared by RAG API and offline comparison scripts."""


DIRECT_SYSTEM_PROMPT = (
    "你是中文法律问答助手。请根据案件事实进行法律三段论推理，"
    "不要把案情中的 QQ、日期、金额等事实改写成数学题、号码题或闲聊问题。"
)

RAG_SYSTEM_PROMPT = (
    "你是中文法律问答助手。请结合给定法条片段和案件事实进行法律三段论推理。"
    "只能使用片段中明确出现的罪名、法条号和量刑规则。"
    "如果片段不能支持某个结论，必须写“当前知识库无法确认”。"
    "不要把案情中的 QQ、日期、金额等事实改写成数学题、号码题或闲聊问题。"
    "回答必须使用中文，不要夹杂英文。"
)


def build_direct_prompt(question: str) -> str:
    question = sanitize_case_text(question)
    return (
        f"{DIRECT_SYSTEM_PROMPT}\n\n"
        f"案件与问题：\n{question}\n\n"
        "请按“大前提（法条）-小前提（案件事实）-结论（判决结果）”作答。\n\n"
        "答案：\n"
    )


def build_direct_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": build_direct_user_content(question)},
    ]


def build_direct_user_content(question: str) -> str:
    question = sanitize_case_text(question)
    return (
        f"案件与问题：\n{question}\n\n"
        "请按“大前提（法条）-小前提（案件事实）-结论（判决结果）”作答。"
    )


def build_rag_user_content(question: str, contexts: list[dict]) -> str:
    question = sanitize_case_text(question)
    context_text = "\n\n".join(
        f"[S{index}] {item['text']}" for index, item in enumerate(contexts, start=1)
    )
    return (
        f"法条片段：\n{context_text}\n\n"
        f"案件与问题：\n{question}\n\n"
        "作答要求：请按“大前提（法条）-小前提（案件事实）-结论（判决结果）”作答，"
        "最后列出引用片段编号，例如 [S1]; 未被片段支持的结论不要写。"
    )


def build_rag_prompt(question: str, contexts: list[dict]) -> str:
    return (
        f"{RAG_SYSTEM_PROMPT}\n\n"
        f"{build_rag_user_content(question, contexts)}\n\n"
        "答案：\n"
    )


def build_rag_messages(question: str, contexts: list[dict]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_rag_user_content(question, contexts),
        },
    ]


def sanitize_case_text(text: str) -> str:
    return (
        text.replace("QQ网络工具", "网络通讯工具")
        .replace("QQ", "网络通讯工具")
        .replace("qq", "网络通讯工具")
    )
