import os
from typing import List, Optional
from dotenv import load_dotenv
import typer
from rich.console import Console
from rich.prompt import Prompt
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langgraph.graph import StateGraph, END

# 環境変数の読み込み
load_dotenv()

# Richコンソールの初期化
console = Console()

# アプリケーションの初期化
app = typer.Typer(help="技術記事のレコメンデーションツール")

# 状態の型定義
class ArticleState:
    def __init__(self):
        self.read_articles: List[str] = []
        self.recommendations: List[str] = []

# 記事の読み取り履歴を収集するノード
def collect_history(state: ArticleState) -> ArticleState:
    console.print("[bold blue]読んだ技術記事の情報を入力してください（終了するには 'done' と入力）[/bold blue]")
    while True:
        article = Prompt.ask("記事のタイトルと内容の要約")
        if article.lower() == "done":
            break
        state.read_articles.append(article)
    return state

# 記事の推薦を生成するノード
def generate_recommendations(state: ArticleState) -> ArticleState:
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたは技術記事の推薦エキスパートです。
        ユーザーがこれまで読んだ技術記事の情報を基に、関連性の高い新しい技術記事を推薦してください。
        推薦する際は以下の点を考慮してください：
        1. ユーザーの興味分野との関連性
        2. 最新の技術トレンド
        3. 学習の連続性
        4. 記事の質と信頼性
        
        推薦記事は以下の形式で出力してください：
        - タイトル
        - 推薦理由
        - 想定される学習効果"""),
        ("user", "読んだ記事の履歴：\n{history}\n\nこれらの記事を基に、おすすめの技術記事を3つ提案してください。")
    ])
    
    chain = (
        {"history": lambda x: "\n".join(x.read_articles)}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    recommendations = chain.invoke(state)
    state.recommendations = recommendations.split("\n\n")
    return state

# 結果を表示するノード
def display_results(state: ArticleState) -> ArticleState:
    console.print("\n[bold green]おすすめの技術記事[/bold green]")
    for i, rec in enumerate(state.recommendations, 1):
        console.print(f"\n[bold yellow]推薦 {i}[/bold yellow]")
        console.print(rec)
    return state

# グラフの構築
def build_graph():
    workflow = StateGraph(ArticleState)
    
    # ノードの追加
    workflow.add_node("collect_history", collect_history)
    workflow.add_node("generate_recommendations", generate_recommendations)
    workflow.add_node("display_results", display_results)
    
    # エッジの追加
    workflow.add_edge("collect_history", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "display_results")
    workflow.add_edge("display_results", END)
    
    # エントリーポイントの設定
    workflow.set_entry_point("collect_history")
    
    return workflow.compile()

def main():
    """技術記事の推薦を開始します"""
    try:
        # OpenAI APIキーの確認
        if not os.getenv("OPENAI_API_KEY"):
            console.print("[bold red]エラー: OPENAI_API_KEYが設定されていません。[/bold red]")
            console.print("環境変数ファイル(.env)にOPENAI_API_KEYを設定してください。")
            return
        
        # グラフの実行
        graph = build_graph()
        graph.invoke(ArticleState())
        
    except Exception as e:
        console.print(f"[bold red]エラーが発生しました: {str(e)}[/bold red]")

if __name__ == "__main__":
    app.command()(main)
    app()
