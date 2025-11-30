"""
CSV検索のデバッグスクリプト
ベクトルストアが正しく構築されているか、検索が正しく動作するかを確認
"""
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import csv

# 環境変数の読み込み
load_dotenv()

# 定数
DATA_DIR_PATH = "./data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVER_SEARCH_K = 5

def load_employee_csv():
    """社員名簿CSVを部署ごとにグループ化して読み込む"""
    csv_path = os.path.join(DATA_DIR_PATH, "社員について", "社員名簿.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSVファイルが見つかりません: {csv_path}")
        return []
    
    print(f"✅ CSVファイルを読み込み中: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"✅ {len(rows)}行のデータを読み込みました")
    
    # 部署ごとにグループ化
    dept_groups = {}
    for row in rows:
        dept = row.get('部署', '不明')
        if dept not in dept_groups:
            dept_groups[dept] = []
        dept_groups[dept].append(row)
    
    print(f"✅ {len(dept_groups)}個の部署に分類されました:")
    for dept, employees in dept_groups.items():
        print(f"   - {dept}: {len(employees)}名")
    
    # 各部署ごとに1つのドキュメントを作成
    docs = []
    for dept, employees in dept_groups.items():
        content_lines = [f"【{dept}の従業員一覧】\n"]
        for emp in employees:
            emp_info = (
                f"社員ID: {emp.get('社員ID', '')}, "
                f"氏名: {emp.get('氏名(フルネーム)', '')}, "
                f"性別: {emp.get('性別', '')}, "
                f"年齢: {emp.get('年齢', '')}歳, "
                f"従業員区分: {emp.get('従業員区分', '')}, "
                f"部署: {emp.get('部署', '')}, "
                f"役職: {emp.get('役職', '')}, "
                f"スキルセット: {emp.get('スキルセット', '')}, "
                f"保有資格: {emp.get('保有資格', '')}"
            )
            content_lines.append(emp_info)
        
        content = "\n".join(content_lines)
        doc = Document(page_content=content, metadata={"source": csv_path, "department": dept})
        docs.append(doc)
    
    print(f"✅ {len(docs)}個のドキュメントを作成しました")
    
    # 最初のドキュメントの内容をサンプル表示
    if docs:
        print(f"\n📄 サンプル(最初のドキュメント):")
        print(f"部署: {docs[0].metadata['department']}")
        print(f"内容(最初の500文字):\n{docs[0].page_content[:500]}...\n")
    
    return docs

def create_vector_store(docs):
    """ベクトルストアを作成"""
    print("\n🔧 ベクトルストアを作成中...")
    
    # テキスト分割
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(docs)
    print(f"✅ {len(texts)}個のチャンクに分割しました")
    
    # Embeddingモデルの初期化
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # ベクトルストアの作成
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./.chroma_debug"
    )
    print("✅ ベクトルストアを作成しました")
    
    return vectorstore

def test_search(vectorstore, query):
    """検索テスト"""
    print(f"\n🔍 検索クエリ: '{query}'")
    
    # Retrieverの作成
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_SEARCH_K}
    )
    
    # 検索実行
    results = retriever.invoke(query)
    
    print(f"✅ {len(results)}件の結果が見つかりました:\n")
    
    for i, doc in enumerate(results, 1):
        print(f"--- 結果 {i} ---")
        print(f"部署: {doc.metadata.get('department', 'N/A')}")
        print(f"内容(最初の300文字):\n{doc.page_content[:300]}...\n")
    
    return results

def main():
    print("=" * 60)
    print("CSV検索デバッグスクリプト")
    print("=" * 60)
    
    # 1. CSVデータの読み込み
    docs = load_employee_csv()
    
    if not docs:
        print("❌ ドキュメントが作成できませんでした")
        return
    
    # 2. ベクトルストアの作成
    vectorstore = create_vector_store(docs)
    
    # 3. 検索テスト
    test_queries = [
        "人事部に所属している従業員",
        "人事部",
        "営業部の社員",
        "IT部のスタッフ"
    ]
    
    for query in test_queries:
        test_search(vectorstore, query)
    
    print("\n" + "=" * 60)
    print("デバッグ完了")
    print("=" * 60)

if __name__ == "__main__":
    main()
