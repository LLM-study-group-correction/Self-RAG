from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import re

# === OpenAI SparkMax 配置 ===
api_key = "sk-oTjfLHkHcx4RXpf05a15B919669f4bC0A4Fe9738C6776c53"
api_base = "http://maas-api.cn-huabei-1.xf-yun.com/v1"
client = OpenAI(api_key=api_key, base_url=api_base)

# === 嵌入模型和知识库初始化 ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
loader = TextLoader("law_knowledge.txt", encoding='utf-8')
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(documents)
vector_db = FAISS.from_documents(texts, embeddings)

# === 多轮对话历史缓存（简化为内存） ===
session_memory = {}

# === LLM 调用封装 ===
def call_llm_with_history(messages):
    try:
        response = client.chat.completions.create(
            model="xsparkprox",
            messages=messages,
            stream=False,
            temperature=0.7,
            max_tokens=4096,
            extra_headers={"lora_id": "0"},
            stream_options={"include_usage": True}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM 调用失败: {str(e)}"

# === Self-RAG 主逻辑 ===
def self_rag_with_memory(user_question: str, session_id: str = "test-session"):
    history = session_memory.get(session_id, [])
    docs = vector_db.similarity_search(user_question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    context_prompt = "".join([f"{m['role']}：{m['content']}\n" for m in history[-6:]])

    prompt_1 = f"""以下是你与AI法律顾问的对话历史（供参考）：\n{context_prompt}\n请根据以下资料回答最新问题：\n【资料】：\n{context}\n【问题】：{user_question}"""
    answer = call_llm_with_history([{"role": "user", "content": prompt_1}])

    reflection_prompt = f"""请对下列AI法律回答进行自我审查：是否存在逻辑漏洞、依据不足或引用错误？如有，请给出最多3个关键词用于改进检索。\n【回答】：{answer}"""
    reflection = call_llm_with_history([{"role": "user", "content": reflection_prompt}])

    refined_query = ""
    new_context = ""
    keyword_match = re.findall(r"(?:关键词|推荐词)[:：]?(.*)", reflection)
    if keyword_match:
        refined_query = keyword_match[0].strip().split("，")[0].strip()
        new_docs = vector_db.similarity_search(refined_query, k=3)
        new_context = "\n".join([doc.page_content for doc in new_docs])
        refine_prompt = f"请根据下列新资料修正你的回答：\n【资料】：\n{new_context}\n【问题】：{user_question}"
        answer = call_llm_with_history([{"role": "user", "content": refine_prompt}])

    history.extend([
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": answer}
    ])
    session_memory[session_id] = history[-10:]

    print("\n初次检索内容：\n", context)
    print("\n初次回答：\n", answer)
    print("\n自我反思内容：\n", reflection)
    print("\n提取关键词：", refined_query)
    print("\n再次检索内容：\n", new_context)
    print("\n修正后最终回答：\n", answer)
    return answer

# === 示例运行 ===
# === 示例运行 ===
if __name__ == "__main__":
    print("AIGC 法律顾问系统")
    session_id = "test-session"
    while True:
        question = input("\n请输入你的法律问题（输入 'exit' 退出）：\n> ")
        if question.lower() in ["exit", "quit"]:
            print("已退出对话。")
            break
        answer = self_rag_with_memory(question, session_id=session_id)
