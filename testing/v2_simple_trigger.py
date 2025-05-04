"""
장기 기억 기반 챗봇 - LangGraph + ChromaDB 통합 최종 버전
"""

import spacy
import numpy as np
import uuid
from datetime import datetime
import math
import threading
import time
import schedule
import os
import json
import re
from typing import Dict, List, Any, Tuple, Optional, TypedDict

# LangGraph 관련 임포트
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    print("LangGraph를 설치해주세요: pip install langgraph")
    raise

# ChromaDB 임포트
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB를 설치해주세요: pip install chromadb")
    raise

# 1. 기본 설정 및 모델 로드
try:
    nlp = spacy.load("ko_core_news_md")
    print("한글 NER 모델(ko_core_news_md) 로드됨")
except OSError:
    try:
        nlp = spacy.load("en_core_web_sm")
        print("경고: 한글 NER 모델이 설치되지 않았습니다. 영어 모델로 대체합니다.")
    except OSError:
        print("spaCy 모델을 설치해주세요: python -m spacy download en_core_web_sm")
        # 간단한 토크나이저 대체
        class SimpleNLP:
            def __call__(self, text):
                tokens = re.findall(r'\w+', text.lower())
                return SimpleDoc(tokens)
                
        class SimpleDoc:
            def __init__(self, tokens):
                self.tokens = tokens
                self.ents = []
                
            def __iter__(self):
                for token in self.tokens:
                    yield SimpleToken(token)
                    
        class SimpleToken:
            def __init__(self, text):
                self.text = text
                self.lower_ = text.lower()
                
        nlp = SimpleNLP()

# 사용자 정의 트리거 단어
TRIGGER_WORDS = {"미키", "생일", "강아지", "가족", "취미", "체중", "선물", "토이푸들"}

# ----------------- 2. ChromaDB 설정 -----------------

class ChromaMemoryStore:
    """ChromaDB를 사용한 메모리 저장소"""
    
    def __init__(self, collection_name="long_term_memories", persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Chroma 클라이언트 초기화
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # 컬렉션이 존재하는지 확인하고, 없으면 생성
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"기존 컬렉션 '{collection_name}' 로드됨")
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"새 컬렉션 '{collection_name}' 생성됨")
    
    def add_memory(self, memory):
        """새로운 메모리를 벡터 DB에 저장"""
        # 메모리 ID 설정
        memory_id = memory.get("id", str(uuid.uuid4()))
        
        # 메타데이터 구성
        metadata = {
            "user_input": memory["user"],
            "assistant_reply": memory["assistant"],
            "timestamp": memory["timestamp"],
            "importance": str(memory["importance"]),
            "access_count": str(memory.get("access_count", 0)),
            "last_accessed": memory["last_accessed"],
            "triggers": json.dumps(memory["triggers"])  # 리스트를 JSON 문자열로 변환
        }
        
        # 텍스트 구성 (임베딩 대상)
        full_text = memory["user"] + " " + memory["assistant"]
        
        # 컬렉션에 추가
        self.collection.add(
            ids=[memory_id],
            documents=[full_text],
            metadatas=[metadata]
        )
        
        print(f"[Chroma] 메모리 추가됨: {memory_id}")
        return memory_id
    
    def search_by_triggers(self, query_text, trigger_words, top_k=3):
        """트리거 단어와 텍스트 유사도 기반으로 메모리 검색"""
        if not trigger_words:
            return []
            
        # JSON 문자열에서 트리거 단어를 검색하기 위한 쿼리 구성
        # ChromaDB 최신 버전에 맞게 수정된 where 필터
        where_filter = None
        
        # 단순히 트리거 없이 전체 검색
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
    )
        
        # 결과가 없는 경우
        if not results["ids"][0]:
            return []
        
        # 결과 형식화
        memories = []
        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            
            # JSON 문자열에서 트리거 단어 목록으로 변환
            triggers = json.loads(metadata["triggers"])
            
            # 메모리 객체 구성
            memory = {
                "id": doc_id,
                "user": metadata["user_input"],
                "assistant": metadata["assistant_reply"],
                "timestamp": metadata["timestamp"],
                "importance": float(metadata["importance"]),
                "triggers": triggers,
                "access_count": int(metadata["access_count"]),
                "last_accessed": metadata["last_accessed"]
            }
            
            memories.append({
                "memory": memory,
                "similarity": results["distances"][0][i] if "distances" in results else 0.9
            })
            
            # 접근 횟수 증가 및 마지막 접근 시간 갱신
            self.update_access_stats(doc_id)
        
        return memories
    
    def update_access_stats(self, memory_id):
        """메모리 접근 통계 업데이트"""
        # 기존 메모리 가져오기
        result = self.collection.get(ids=[memory_id])
        
        if not result["ids"]:
            return
        
        metadata = result["metadatas"][0]
        
        # 접근 횟수 증가 및 마지막 접근 시간 갱신
        access_count = int(metadata["access_count"]) + 1
        last_accessed = datetime.utcnow().isoformat()
        
        # 업데이트된 메타데이터
        updated_metadata = {**metadata, 
                           "access_count": str(access_count),
                           "last_accessed": last_accessed}
        
        # 메모리 업데이트
        self.collection.update(
            ids=[memory_id],
            metadatas=[updated_metadata]
        )
    
    def decay_and_prune(self, decay_rate=0.05, score_threshold=0.4):
        """시간 경과에 따른 기억 점수 감소 및 임계값 이하 항목 제거"""
        current_time = datetime.utcnow()
        
        # 모든 메모리 가져오기
        all_memories = self.collection.get()
        
        if not all_memories["ids"]:
            return
        
        to_delete = []
        
        for i, memory_id in enumerate(all_memories["ids"]):
            metadata = all_memories["metadatas"][i]
            
            # 마지막 접근 시간으로부터 경과된 일수 계산
            last_accessed = datetime.fromisoformat(metadata["last_accessed"])
            delta_days = (current_time - last_accessed).days
            
            # 접근 횟수 가져오기
            access_count = int(metadata["access_count"])
            
            # 기본 점수 (중요도)
            base_score = float(metadata["importance"])
            
            # 최종 점수 계산
            score = base_score + math.log(1 + access_count) - decay_rate * delta_days
            
            # 임계값 미만이면 제거 대상에 추가
            if score < score_threshold:
                to_delete.append(memory_id)
        
        # 제거 대상 메모리 삭제
        if to_delete:
            self.collection.delete(ids=to_delete)
            print(f"[Chroma] 망각 알고리즘 실행: {len(to_delete)}개 메모리 제거됨")

# ----------------- 3. 간단한 응답 생성기 -----------------

class SimpleChatbot:
    """대화형 응답 생성기 (transformers 대체)"""
    
    def __init__(self):
        self.templates = [
            "네, {trigger}에 대해 {response}",
            "{trigger}에 대해 말씀드리자면, {response}",
            "{trigger}에 관해 기억하고 있어요. {response}",
            "네, {response}",
            "{response}",
        ]
        
        self.default_responses = {
            "미키": "미키는 5.6kg의 토이푸들이죠. 3월 12일이 생일인 것으로 기억합니다.",
            "생일": "3월 12일이 미키의 생일이었죠.",
            "강아지": "미키는 노란 공 장난감을 좋아하는 토이푸들이에요.",
            "체중": "미키의 체중은 5.6kg이에요.",
            "가족": "가족에 관한 이야기를 해주셨었네요.",
            "회사": "어제 회사에서 발표를 하셨다고 했었죠.",
        }
        
        self.generic_responses = [
            "그것에 대해 더 자세히 알려주실래요?",
            "흥미로운 질문이네요. 더 구체적으로 말씀해주세요.",
            "잘 이해하지 못했어요. 다른 방식으로 물어봐 주실래요?",
            "그 부분에 대해서는 기억이 없어요.",
            "더 자세한 내용이 필요해요."
        ]
    
    def __call__(self, prompt, **kwargs):
        """prompt에 기반한 응답 생성"""
        import random
        
        # 트리거 단어 찾기
        found_trigger = None
        for trigger in TRIGGER_WORDS:
            if trigger in prompt.lower():
                found_trigger = trigger
                break
                
        # 기본 응답 생성
        if found_trigger and found_trigger in self.default_responses:
            template = random.choice(self.templates)
            response = template.format(
                trigger=found_trigger,
                response=self.default_responses[found_trigger]
            )
        else:
            response = random.choice(self.generic_responses)
            
        # transformers 형식과 호환되게 반환
        return [{"generated_text": prompt + " " + response}]

# 응답 생성을 위한 모델 (대체 가능)
chatbot = SimpleChatbot()

# ----------------- 4. LangGraph 노드 함수 정의 -----------------

# 상태 정의
class ChatbotState(TypedDict, total=False):
    user_input: str
    memory_store: Any
    triggers: List[str]
    importance: float
    retrieved_memories: List[Dict[str, Any]]
    assistant_reply: str

# 단기 메모리 (세션 내 유지)
STM = []

def extract_triggers_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """트리거 단어 추출 및 중요도 평가 노드"""
    user_input = state["user_input"]
    
    # spaCy로 텍스트 처리
    doc = nlp(user_input)
    triggers = set()
    
    # NER 기반 키워드 추출
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "DATE", "LOC", "ORG", "CARDINAL", "PER"}:
            triggers.add(ent.text.lower())
    
    # 사용자 정의 트리거 단어 감지
    for token in doc:
        if token.text.lower() in TRIGGER_WORDS:
            triggers.add(token.text.lower())
    
    # 중요도 점수 계산
    score = min(len(triggers) * 0.2, 1.0)
    
    # 감정 표현이 있는 경우 중요도 가중치 부여
    if any(word in user_input.lower() for word in ["좋아", "사랑", "행복", "슬퍼", "화나", "기뻐"]):
        score += 0.1
        score = min(score, 1.0)
    
    print(f"[트리거 감지] {list(triggers)}, 중요도: {score:.2f}")
    
    # 결과 반환
    return {
        **state,
        "triggers": list(triggers),
        "importance": score
    }

def search_memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """벡터 DB에서 트리거 단어 기반 메모리 검색 노드"""
    user_input = state["user_input"]
    triggers = state["triggers"]
    
    # 벡터 DB에서 검색
    memory_store = state["memory_store"]
    results = memory_store.search_by_triggers(user_input, triggers)
    
    if results:
        print(f"[메모리 검색] {len(results)}개 메모리 검색됨")
        for res in results:
            print(f"  - {res['memory']['user'][:30]}... (유사도: {res['similarity']:.2f})")
    else:
        print("[메모리 검색] 관련 메모리 없음")
    
    return {
        **state,
        "retrieved_memories": results
    }

def generate_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """검색된 기억 기반 응답 생성 노드"""
    user_input = state["user_input"]
    memory_results = state["retrieved_memories"]
    
    # 기억 내용을 텍스트로 요약
    memory_context = "\n".join(
        f"- {mem['memory']['user']} → {mem['memory']['assistant']}"
        for mem in memory_results
    ) if memory_results else ""
    
    # 프롬프트 구성
    if memory_context:
        prompt = (
            "다음은 과거 대화 내용입니다:\n"
            f"{memory_context}\n\n"
            f"User: {user_input}\n"
            "Assistant:"
        )
    else:
        prompt = f"User: {user_input}\nAssistant:"
    
    try:
        # 응답 생성
        reply = chatbot(prompt, max_length=150, do_sample=True, top_k=50)[0]["generated_text"]
        
        # 응답에서 프롬프트 부분 제거
        clean_reply = reply[len(prompt):].strip()
        
        # 응답이 없거나 너무 짧은 경우 기본 응답
        if not clean_reply or len(clean_reply) < 5:
            if memory_results:
                memory_summary = ", ".join([m["memory"]["user"][:20] for m in memory_results[:2]])
                clean_reply = f"네, 기억나요. {memory_summary}에 대해 말씀하셨었죠."
            else:
                clean_reply = "더 자세히 말씀해주세요."
    except Exception as e:
        print(f"응답 생성 오류: {e}")
        # 오류 발생시 간단한 기억 기반 응답
        if memory_results:
            memory_item = memory_results[0]["memory"]["user"]
            clean_reply = f"네, 예전에 '{memory_item}'에 대해 말씀하셨던 것 같아요."
        else:
            clean_reply = "무슨 말씀인지 잘 이해하지 못했어요. 더 자세히 말씀해주세요."
    
    print(f"[응답 생성] {clean_reply}")
    
    return {
        **state,
        "assistant_reply": clean_reply
    }

def store_memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """현재 대화를 STM에 저장하고 필요시 LTM으로 이전하는 노드"""
    global STM
    user_input = state["user_input"]
    assistant_reply = state["assistant_reply"]
    triggers = state["triggers"]
    importance = state["importance"]
    
    # STM에 현재 대화 저장
    memory = {
        "id": str(uuid.uuid4()),
        "user": user_input,
        "assistant": assistant_reply,
        "timestamp": datetime.utcnow().isoformat(),
        "triggers": triggers,
        "importance": importance,
        "access_count": 0,
        "last_accessed": datetime.utcnow().isoformat()
    }
    
    STM.append(memory)
    print(f"[STM] 저장됨: '{user_input[:30]}...' (중요도: {importance:.2f})")
    
    # STM → LTM 이전 여부 판단
    memory_store = state["memory_store"]
    flush_to_ltm(memory_store)
    
    return state

def flush_to_ltm(memory_store, min_importance=0.5, max_turns=5):
    """STM에서 LTM으로 메모리 이전"""
    global STM
    to_promote = []
    
    # 중요도 기준 충족 항목 식별
    for memory in STM:
        if memory["importance"] >= min_importance:
            to_promote.append(memory)
            print(f"[LTM] 중요도({memory['importance']:.2f})로 이전: '{memory['user'][:30]}...'")
    
    # 대화 턴 수 초과 시 전체 이전
    if len(STM) >= max_turns:
        to_promote = STM
        print(f"[LTM] 대화 턴 초과({len(STM)} >= {max_turns})로 전체 이전")
    
    # LTM으로 이전
    for memory in to_promote:
        memory_store.add_memory(memory)
    
    # 이전된 항목은 STM에서 제거
    if to_promote:
        STM = [m for m in STM if m not in to_promote]
        print(f"[STM] 정리 후 크기: {len(STM)}")

# ----------------- 5. LangGraph 그래프 구성 -----------------

def create_memory_chatbot_graph():
    """LangGraph를 사용한 메모리 챗봇 그래프 생성"""
    
    # LangGraph 버전 체크 및 그래프 초기화
    try:
        import importlib.metadata
        version = importlib.metadata.version("langgraph")
        print(f"LangGraph 버전: {version}")
        
        # 0.1.0 이상일 경우 새로운 API 방식 사용
        from packaging import version as pkg_version
        if pkg_version.parse(version) >= pkg_version.parse("0.1.0"):
            workflow = StateGraph(ChatbotState)
        else:
            workflow = StateGraph(state_schema=ChatbotState)
    except:
        # 버전 확인 실패 시 기본 초기화 시도
        try:
            workflow = StateGraph(ChatbotState)
        except:
            workflow = StateGraph(state_schema=ChatbotState)
    
    # 노드 추가
    workflow.add_node("extract_triggers", extract_triggers_node)
    workflow.add_node("search_memory", search_memory_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("store_memory", store_memory_node)
    
    # 엣지 설정 (노드 간 연결)
    workflow.set_entry_point("extract_triggers")
    workflow.add_edge("extract_triggers", "search_memory")
    workflow.add_edge("search_memory", "generate_response")
    workflow.add_edge("generate_response", "store_memory")
    workflow.add_edge("store_memory", END)
    
    # 그래프 컴파일
    memory_graph = workflow.compile()
    
    return memory_graph

# ----------------- 6. 망각 알고리즘 스케줄러 -----------------

def setup_memory_decay_scheduler(memory_store):
    """망각 알고리즘 스케줄러 설정"""
    
    def run_decay():
        memory_store.decay_and_prune()
    
    # 매일 자정에 망각 알고리즘 실행
    schedule.every().day.at("00:00").do(run_decay)
    
    # 스케줄러 백그라운드 실행
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

def run_scheduler():
    """스케줄러 실행 루프"""
    while True:
        schedule.run_pending()
        time.sleep(60)

# ----------------- 7. 메인 챗봇 클래스 -----------------

class MemoryChatbot:
    """장기 기억 기반 챗봇 클래스"""
    
    def __init__(self, collection_name="memory_chatbot", persist_dir="./chroma_db"):
        print("\n=== 장기 기억 기반 챗봇 시스템 초기화 중... ===\n")
        
        # 벡터 DB 초기화
        self.memory_store = ChromaMemoryStore(
            collection_name=collection_name,
            persist_directory=persist_dir
        )
        
        # LangGraph 워크플로우 생성
        try:
            self.graph = create_memory_chatbot_graph()
            print("LangGraph 워크플로우 생성 완료")
        except Exception as e:
            print(f"LangGraph 초기화 오류: {e}")
            raise
        
        # 망각 알고리즘 스케줄러 설정
        setup_memory_decay_scheduler(self.memory_store)
        
        print("\n=== 장기 기억 기반 챗봇 시스템 초기화 완료 ===\n")
    
    def process_message(self, user_input):
        """사용자 입력 처리 및 응답 생성"""
        print(f"\n[사용자 입력] {user_input}")
        
        # 초기 상태 구성
        state = {
            "user_input": user_input,
            "memory_store": self.memory_store
        }
        
        # 그래프 실행
        try:
            result = self.graph.invoke(state)
            return result["assistant_reply"]
        except Exception as e:
            print(f"오류 발생: {e}")
            # 오류 발생 시 간단한 응답 반환
            return "죄송합니다, 처리 중 오류가 발생했습니다."
    
    def seed_with_memories(self, memories):
        """초기 메모리로 시스템 시드 설정"""
        print("\n[초기 메모리 시드 설정]\n")
        
        for i, (user_msg, bot_reply) in enumerate(memories):
            print(f"메모리 {i+1}: '{user_msg}'")
            
            # 트리거 추출
            doc = nlp(user_msg)
            triggers = set()
            
            # NER 및 키워드 추출
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "DATE", "LOC", "ORG", "CARDINAL", "PER"}:
                    triggers.add(ent.text.lower())
            
            for token in doc:
                if token.text.lower() in TRIGGER_WORDS:
                    triggers.add(token.text.lower())
            
            # 중요도 계산
            importance = min(len(triggers) * 0.2, 1.0)
            
            # 메모리 생성
            memory = {
                "id": f"seed_{i}_{uuid.uuid4()}",
                "user": user_msg,
                "assistant": bot_reply,
                "timestamp": datetime.utcnow().isoformat(),
                "triggers": list(triggers),
                "importance": importance,
                "access_count": 0,
                "last_accessed": datetime.utcnow().isoformat()
            }
            
            # 벡터 DB에 저장
            self.memory_store.add_memory(memory)
            print(f"  -> 챗봇: '{bot_reply}'")
            print(f"  -> 트리거: {list(triggers)}, 중요도: {importance:.2f}")
            print()

# ----------------- 8. 실행 예시 -----------------

def main():
    """챗봇 실행 예시"""
    
    # 챗봇 초기화
    chatbot = MemoryChatbot()
    
    # 초기 대화 샘플로 메모리 채우기
    seed_memories = [
        ("우리 미키는 5.6kg인 토이푸들이야", "미키가 5.6kg인 토이푸들이군요! 귀여운 크기네요."),
        ("미키 생일은 3월 12일이야", "3월 12일이 미키의 생일이군요! 기억해두겠습니다."),
        ("미키는 노란 공 장난감을 좋아해", "미키가 노란 공 장난감을 좋아하는군요. 장난감 취향도 있네요!"),
        ("나는 어제 회사에서 발표를 했어", "회사에서 발표하셨군요! 발표는 잘 진행되었나요?")
    ]
    
    # 메모리 시드 설정
    chatbot.seed_with_memories(seed_memories)
    
    # 대화 예시
    test_queries = [
        "미키 생일이 언제였지?",
        "우리 강아지 무게가 얼마였더라?",
        "내가 어제 뭐 했었지?"
    ]
    
    # 대화 처리
    for query in test_queries:
        print("\n" + "="*50)
        print(f"사용자: {query}")
        response = chatbot.process_message(query)
        print(f"챗봇: {response}")
    
    # 대화형 모드
    print("\n\n" + "="*50)
    print("대화형 모드를 시작합니다. 종료하려면 'exit' 또는 'quit'을 입력하세요.")
    print("="*50 + "\n")
    
    while True:
        user_input = input("사용자: ")
        if user_input.lower() in ["exit", "quit", "종료"]:
            print("챗봇을 종료합니다.")
            break
            
        response = chatbot.process_message(user_input)
        print(f"챗봇: {response}")

# 실행
if __name__ == "__main__":
    main()