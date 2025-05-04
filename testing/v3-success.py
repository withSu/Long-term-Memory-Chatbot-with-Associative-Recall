"""
장기 기억 기반 챗봇 - 향상된 인간형 연상 기억 및 망각 알고리즘 적용 버전
"""

import spacy
import numpy as np
import uuid
from datetime import datetime, timedelta
import math
import threading
import time
import schedule
import os
import json
import re
from typing import Dict, List, Any, Tuple, Optional, TypedDict, Set
from collections import defaultdict, Counter
import random
from transformers import pipeline
import json
from functools import lru_cache
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

# 사용자 정의 트리거 단어 및 연관 개념 매핑
# 기본 트리거 단어
BASE_TRIGGER_WORDS = {
    "미키", "생일", "강아지", "가족", "취미", "체중", "선물", "토이푸들"
}

# 연관 개념 확장 (관련 단어 그룹화)
RELATED_CONCEPTS = {
    "강아지": {"개", "토이푸들", "댕댕이", "애완견", "반려견", "반려동물", "펫", "멍멍이", "멍멍"},
    "생일": {"탄생일", "생신", "생일선물", "케이크", "파티", "축하", "기념일", "축하연", "생일파티"},
    "체중": {"몸무게", "무게", "kg", "킬로", "살", "다이어트", "체급", "체형", "몸매"},
    "미키": {"미키마우스", "마우스", "캐릭터"},
}

# 감정 단어 및 가중치
EMOTION_WORDS = {
    "좋아": 0.1, "사랑": 0.2, "행복": 0.15, "슬퍼": 0.15, "화나": 0.15, "기뻐": 0.1,
    "즐거워": 0.1, "그리워": 0.15, "놀라": 0.1, "무서워": 0.15, "걱정": 0.1
}

# 확장된 트리거 단어 집합 생성 (메인 + 연관 개념)
TRIGGER_WORDS = set(BASE_TRIGGER_WORDS)
for concept, related in RELATED_CONCEPTS.items():
    TRIGGER_WORDS.update(related)

# 메모리 유형 정의
class MemoryType:
    CORE = "core"         # 핵심 정보 (사용자 ID, 설정 등)
    EPISODIC = "episodic" # 일화적 기억 (특정 사건, 경험)
    SEMANTIC = "semantic" # 의미적 기억 (일반적 사실, 지식)
    EMOTIONAL = "emotional" # 감정적 기억 (강한 감정이 포함된 대화)

# LLMConceptExpander 클래스 추가
class LLMConceptExpander:
    """LLM을 사용하여 트리거 단어의 관련 개념을 자동으로 확장하는 클래스"""
    
    def __init__(self, use_local_model=True, model_name="google/flan-t5-base"):
        """
        Args:
            use_local_model: 로컬 모델 사용 여부 (True면 무료 모델 사용)
            model_name: 사용할 모델명
        """
        self.use_local_model = use_local_model
        
        if use_local_model:
            # 무료 로컬 LLM 사용 (예: Flan-T5)
            try:
                self.llm = pipeline("text2text-generation", model=model_name)
                print(f"로컬 LLM 로드됨: {model_name}")
            except Exception as e:
                print(f"LLM 로드 실패: {e}, 개념 확장 비활성화")
                self.llm = None
    
    @lru_cache(maxsize=128)  # 자주 사용되는 단어의 결과 캐싱
    def expand_concept(self, trigger_word, context=None, max_concepts=10):
        """
        주어진 트리거 단어에 대해 관련 개념들을 확장
        
        Args:
            trigger_word: 확장할 기본 트리거 단어
            context: 문맥 정보 (선택사항)
            max_concepts: 반환할 최대 개념 수
            
        Returns:
            set: 관련 개념들의 집합
        """
        if not self.llm:
            return set()
            
        # 프롬프트 구성
        prompt = self.create_expansion_prompt(trigger_word, context)
        
        # 개념 확장 실행
        result = self.expand_with_local_model(prompt)
        
        # 결과 파싱
        concepts = self.parse_expansion_result(result, max_concepts)
        
        return concepts
    
    def create_expansion_prompt(self, trigger_word, context=None):
        """관련 개념 확장을 위한 프롬프트 생성"""
        
        # 기본 프롬프트
        prompt = f"단어 '{trigger_word}'과 관련된 동의어, 유사어, 관련 개념을 한글로 나열해주세요.\n"
        
        # 문맥이 있으면 추가
        if context:
            prompt += f"문맥: {context}\n"
        
        # 출력 형식 지정
        prompt += "JSON 형식으로 출력해주세요: {\"concepts\": [\"단어1\", \"단어2\", ...]}"
        
        return prompt
    
    def expand_with_local_model(self, prompt):
        """로컬 모델을 사용한 개념 확장"""
        try:
            # 모델 실행
            outputs = self.llm(prompt, max_length=512, num_beams=4, early_stopping=True)
            result = outputs[0]["generated_text"]
            
            return result
        except Exception as e:
            print(f"로컬 모델 실행 오류: {e}")
            return ""
    
    def parse_expansion_result(self, result, max_concepts):
        """LLM 응답 파싱"""
        try:
            # JSON 부분 추출
            if "{" in result and "}" in result:
                start = result.find("{")
                end = result.rfind("}") + 1
                json_str = result[start:end]
                
                # JSON 파싱
                data = json.loads(json_str)
                concepts = data.get("concepts", [])
                
                # 최대 개념 수 제한
                return set(concepts[:max_concepts])
            else:
                print("JSON 형식을 찾을 수 없음. 문자열 파싱 시도...")
                # JSON이 아닌 경우 단순 문자열 파싱
                concepts = result.split(",")
                return set([c.strip() for c in concepts[:max_concepts] if c.strip()])
        except Exception as e:
            print(f"결과 파싱 오류: {e}")
            return set()




# ----------------- 2. 향상된 ChromaDB 메모리 저장소 -----------------

class EnhancedMemoryStore:
    """확장된 ChromaDB 메모리 저장소 - 다양한 메모리 유형 및 망각 알고리즘 지원"""
    
    def __init__(self, collection_name="long_term_memories", persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.concept_expander = LLMConceptExpander(use_local_model=True)
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
        
        # 메모리 연결 그래프 (관련 메모리를 연결하는 그래프 구조)
        self.memory_graph = defaultdict(set)
        
        # 세션 기반 트리거 단어 빈도 추적
        self.session_triggers = Counter()
        
        # 오늘 날짜 기록 (망각 알고리즘용)
        self.today = datetime.utcnow().date()
    
    def add_memory(self, memory, related_memories=None):
        """
        새로운 메모리를 벡터 DB에 저장하고 관련 메모리와 연결
        
        Args:
            memory (dict): 저장할 메모리 객체
            related_memories (list): 관련 메모리 ID 목록
        """
        # 메모리 ID 설정
        memory_id = memory.get("id", str(uuid.uuid4()))
        
        # 메모리 유형 확인 (기본값: episodic)
        memory_type = memory.get("memory_type", MemoryType.EPISODIC)
        
        # 현재 시간 및 망각 관련 필드 추가
        current_time = datetime.utcnow()
        
        # 메타데이터 구성
        metadata = {
            "user_input": memory["user"],
            "assistant_reply": memory["assistant"],
            "timestamp": memory["timestamp"],
            "importance": str(memory["importance"]),
            "access_count": str(memory.get("access_count", 0)),
            "last_accessed": memory["last_accessed"],
            "memory_type": memory_type,
            "triggers": json.dumps(memory["triggers"]),  # 리스트를 JSON 문자열로 변환
            "creation_date": current_time.isoformat(),
            "decay_score": str(memory.get("decay_score", 1.0)),  # 초기 decay_score = 1.0 (최대)
            "emotional_weight": str(memory.get("emotional_weight", 0.0))  # 감정적 가중치
        }
        
        # 텍스트 구성 (임베딩 대상)
        full_text = memory["user"] + " " + memory["assistant"]
        
        # 컬렉션에 추가
        self.collection.add(
            ids=[memory_id],
            documents=[full_text],
            metadatas=[metadata]
        )
        
        # 메모리 그래프에 추가
        if related_memories:
            for related_id in related_memories:
                self.memory_graph[memory_id].add(related_id)
                self.memory_graph[related_id].add(memory_id)
        
        print(f"[Chroma] 메모리 추가됨: {memory_id}")
        
        # 세션 트리거 단어 빈도 업데이트
        for trigger in memory["triggers"]:
            self.session_triggers[trigger] += 1
        
        return memory_id
    
    def associate_memories(self, memory_id1, memory_id2):
        """두 메모리 ID를 연관시킴 (그래프에 연결 추가)"""
        self.memory_graph[memory_id1].add(memory_id2)
        self.memory_graph[memory_id2].add(memory_id1)
    
    # search_by_triggers 메서드 내부 수정:

    def search_by_triggers(self, query_text, trigger_words, top_k=5):
        """
        트리거 단어와 텍스트 유사도 기반으로 메모리 검색
        향상된 검색 로직: 트리거 단어의 연관 개념까지 확장하여 검색
        """
        if not trigger_words:
            return []
        
        # 연관 개념 확장
        expanded_triggers = set(trigger_words)
        
        for trigger in trigger_words:
            # 모든 연관 개념 그룹 검사
            for concept, related_terms in RELATED_CONCEPTS.items():
                # 트리거가 연관 개념 그룹의 단어와 일치하는지 확인
                if trigger in related_terms or trigger == concept:
                    # 동일 그룹의 다른 단어들 추가
                    expanded_triggers.update(related_terms)
                    expanded_triggers.add(concept)
        
        print(f"[트리거 확장] {trigger_words} → {expanded_triggers}")
        
        # 검색 실행 - where 필터 대신 전체 검색 후 파이썬에서 필터링
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(20, top_k * 3)  # 더 많은 결과를 가져와서 필터링
        )
        
        # 결과가 없는 경우
        if not results["ids"][0]:
            return []
        
        # 트리거 단어 기반 필터링 및 결과 형식화
        filtered_memories = []  # 빈 리스트로 초기화
        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            
            # JSON 문자열에서 트리거 단어 목록으로 변환
            mem_triggers = json.loads(metadata["triggers"])
            
            # 메모리의 트리거 단어와 확장된 트리거 단어 간 교집합 확인
            if any(trigger in expanded_triggers for trigger in mem_triggers):
                # 메모리 객체 구성
                memory = {
                    "id": doc_id,
                    "user": metadata["user_input"],
                    "assistant": metadata["assistant_reply"],
                    "timestamp": metadata["timestamp"],
                    "importance": float(metadata["importance"]),
                    "triggers": mem_triggers,
                    "access_count": int(metadata["access_count"]),
                    "last_accessed": metadata["last_accessed"],
                    "memory_type": metadata.get("memory_type", MemoryType.EPISODIC),
                    "decay_score": float(metadata.get("decay_score", "1.0")),
                    "emotional_weight": float(metadata.get("emotional_weight", "0.0"))
                }
                
                # 유사도 점수 계산
                similarity = results["distances"][0][i] if "distances" in results else 0.9
                
                # 가중치 계산
                weighted_score = (
                    0.7 * similarity +
                    0.2 * float(metadata.get("decay_score", "1.0")) +
                    0.1 * float(metadata.get("emotional_weight", "0.0"))
                )
                
                filtered_memories.append({
                    "memory": memory,
                    "similarity": similarity,
                    "weighted_score": weighted_score
                })
        
        # 가중치 점수로 정렬
        filtered_memories.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        # top_k개만 선택
        top_memories = filtered_memories[:top_k]
        
        # 접근 통계 업데이트
        for result in top_memories:
            self.update_access_stats(result["memory"]["id"])
            
            # 메모리 그래프에서 연관 메모리 찾기
            if result["memory"]["id"] in self.memory_graph:
                related_ids = self.memory_graph[result["memory"]["id"]]
                # 연관 메모리의 접근 통계도 업데이트 (약한 강화)
                for related_id in related_ids:
                    self.update_access_stats(related_id, reinforcement=0.5)
        
        # 결과 로깅
        for i, res in enumerate(top_memories):
            print(f"  - {i+1}. '{res['memory']['user'][:30]}...' (점수: {res['weighted_score']:.2f})")
        
        return top_memories


    def update_access_stats(self, memory_id, reinforcement=1.0):
        """
        메모리 접근 통계 업데이트 - 접근 횟수, 마지막 접근 시간, 강화 학습
        
        Args:
            memory_id (str): 메모리 ID
            reinforcement (float): 강화 계수 (1.0=완전 강화, <1.0=약한 강화)
        """
        # 기존 메모리 가져오기
        result = self.collection.get(ids=[memory_id])
        
        if not result["ids"]:
            return
        
        metadata = result["metadatas"][0]
        
        # 접근 횟수 증가 및 마지막 접근 시간 갱신
        access_count = int(metadata["access_count"]) + reinforcement
        last_accessed = datetime.utcnow().isoformat()
        
        # decay_score 강화 (접근에 따른 기억 강화)
        current_decay = float(metadata.get("decay_score", "1.0"))
        # 강화는 decay_score를 높이되 최대 1.0 유지
        new_decay = min(1.0, current_decay + 0.1 * reinforcement)
        
        # 업데이트된 메타데이터
        updated_metadata = {
            **metadata, 
            "access_count": str(access_count),
            "last_accessed": last_accessed,
            "decay_score": str(new_decay)
        }
        
        # 메모리 업데이트
        self.collection.update(
            ids=[memory_id],
            metadatas=[updated_metadata]
        )
    
    def apply_decay(self, decay_rate=0.05):
        """
        시간 경과에 따른 기억 점수 감소 적용 (모든 메모리 대상)
        
        Args:
            decay_rate (float): 일별 감소율
        """
        # 오늘 날짜 확인
        today = datetime.utcnow().date()
        
        # 이전 실행 후 날짜가 변경된 경우에만 실행
        if today == self.today:
            print(f"[Decay] 오늘({today})은 이미 decay가 적용되었습니다.")
            return
        
        # 날짜 업데이트
        days_elapsed = (today - self.today).days
        self.today = today
        
        print(f"[Decay] {days_elapsed}일 경과, decay 적용 중...")
        
        # 모든 메모리 가져오기
        all_memories = self.collection.get()
        
        if not all_memories["ids"]:
            return
        
        update_ids = []
        update_metadatas = []
        
        for i, memory_id in enumerate(all_memories["ids"]):
            metadata = all_memories["metadatas"][i]
            
            # 현재 decay_score 가져오기
            current_decay = float(metadata.get("decay_score", "1.0"))
            
            # 메모리 유형에 따른 감소율 조정
            adjusted_decay_rate = decay_rate
            memory_type = metadata.get("memory_type", MemoryType.EPISODIC)
            
            # 메모리 유형별 감소율 차등 적용
            if memory_type == MemoryType.CORE:
                adjusted_decay_rate = 0.01  # 핵심 기억은 매우 느리게 감소
            elif memory_type == MemoryType.SEMANTIC:
                adjusted_decay_rate = 0.03  # 의미적 기억은 중간 속도로 감소
            elif memory_type == MemoryType.EMOTIONAL:
                adjusted_decay_rate = 0.02  # 감정적 기억은 느리게 감소
            
            # 접근 횟수와 중요도에 따른 decay 감소 보정
            access_count = int(metadata.get("access_count", "0"))
            importance = float(metadata.get("importance", "0.5"))
            
            # decay_score 계산: 현재값 - (감소율 * 경과일 * (1 - 중요도/3 - log(1+접근횟수)/5))
            # 중요도가 높거나 자주 접근한 메모리는 decay가 느리게 진행됨
            protection_factor = (1 - importance/3 - math.log(1 + access_count)/5)
            protection_factor = max(0.1, protection_factor)  # 최소 0.1로 제한
            
            new_decay = current_decay - (adjusted_decay_rate * days_elapsed * protection_factor)
            new_decay = max(0.0, new_decay)  # 음수 방지
            
            # 메타데이터 업데이트 준비
            updated_metadata = {**metadata, "decay_score": str(new_decay)}
            
            # 업데이트 큐에 추가
            update_ids.append(memory_id)
            update_metadatas.append(updated_metadata)
        
        # 모든 업데이트 한 번에 실행 (성능 최적화)
        if update_ids:
            self.collection.update(
                ids=update_ids,
                metadatas=update_metadatas
            )
            
            print(f"[Decay] {len(update_ids)}개 메모리에 decay 적용 완료")
    
    def prune_memories(self, score_threshold=0.2):
        """
        decay_score가 임계값 이하인 메모리 비활성화
        완전히 삭제하지 않고 검색 결과에서 제외되도록 처리
        
        Args:
            score_threshold (float): 유지할 최소 decay_score
        """
        # 모든 메모리 가져오기
        all_memories = self.collection.get()
        
        if not all_memories["ids"]:
            return
        
        inactive_ids = []
        inactive_metadatas = []
        
        for i, memory_id in enumerate(all_memories["ids"]):
            metadata = all_memories["metadatas"][i]
            
            # decay_score 확인
            decay_score = float(metadata.get("decay_score", "1.0"))
            
            # 임계값 미만이면 비활성화 표시
            if decay_score < score_threshold:
                # 해당 메모리가 이미 비활성 상태인지 확인
                is_inactive = metadata.get("inactive", "false").lower() == "true"
                
                if not is_inactive:
                    # 비활성화 상태로 업데이트
                    updated_metadata = {**metadata, "inactive": "true"}
                    inactive_ids.append(memory_id)
                    inactive_metadatas.append(updated_metadata)
        
        # 비활성화 처리
        if inactive_ids:
            self.collection.update(
                ids=inactive_ids,
                metadatas=inactive_metadatas
            )
            
            print(f"[Prune] {len(inactive_ids)}개 메모리 비활성화 처리됨")
    
    def consolidate_memories(self):
        """
        메모리 통합 (유사한 메모리 압축)
        동일 주제의 여러 메모리를 하나의 요약 메모리로 통합
        """
        # 이 기능은 실제 구현에서 LLM을 사용하여 유사 메모리를 요약해야 함
        # 여기서는 간소화를 위해 주기적 정리 로직만 포함
        print("[Consolidate] 메모리 통합 처리 (실제 구현에서는 LLM 필요)")

# ----------------- 3. 향상된 트리거 단어 추출 -----------------

def extract_triggers_enhanced(text):
    """
    향상된 트리거 단어 추출 및 중요도 평가
    - NER, 사용자 정의 단어, 연관 개념 확장, 감정 인식 포함
    
    Args:
        text (str): 사용자 입력 텍스트
        
    Returns:
        tuple: (트리거 단어 리스트, 중요도 점수, 감정 가중치)
    """
    # 텍스트 전처리
    text_lower = text.lower()
    
    # 트리거 집합 초기화
    triggers = set()
    
    # 1. 기본 NER 기반 추출
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "DATE", "LOC", "ORG", "CARDINAL", "PER"}:
            triggers.add(ent.text.lower())
    
    # 2. 사용자 정의 트리거 단어 감지 (정규식 방식으로 개선)
    for trigger in TRIGGER_WORDS:
        # 단어 경계를 고려한 패턴 매칭
        pattern = r'\b' + re.escape(trigger) + r'\b'
        if re.search(pattern, text_lower):
            triggers.add(trigger)
    
    # 3. 감정 표현 감지 및 감정 가중치 계산
    emotion_weight = 0.0
    for emotion, weight in EMOTION_WORDS.items():
        if emotion in text_lower:
            emotion_weight += weight
    
    # 감정 가중치 상한 설정
    emotion_weight = min(emotion_weight, 0.5)
    
    # 4. 중요도 점수 계산
    # 기본 점수: 트리거 단어 수 기반 (최대 5개까지 고려)
    base_score = min(len(triggers) * 0.2, 1.0)
    
    # 조정된 중요도: 기본 점수 + 감정 가중치 (최대 1.0)
    importance = min(base_score + emotion_weight, 1.0)
    
    return list(triggers), importance, emotion_weight

# ----------------- 4. LangGraph 노드 함수 정의 -----------------

class ChatbotState(TypedDict, total=False):
    user_input: str
    memory_store: Any
    triggers: List[str]
    importance: float
    emotional_weight: float
    retrieved_memories: List[Dict[str, Any]]
    assistant_reply: str

# 단기 메모리 (세션 내 유지)
STM = []

def extract_triggers_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """향상된 트리거 단어 추출 및 중요도 평가 노드"""
    user_input = state["user_input"]
    
    # 향상된 트리거 추출 함수 사용
    triggers, importance, emotional_weight = extract_triggers_enhanced(user_input)
    
    print(f"[트리거 감지] {triggers}, 중요도: {importance:.2f}, 감정: {emotional_weight:.2f}")
    
    # 결과 반환
    return {
        **state,
        "triggers": triggers,
        "importance": importance,
        "emotional_weight": emotional_weight
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
    
    try:
        # 기억 내용을 텍스트로 요약
        if memory_results:
            # 검색된 메모리로부터 응답 생성
            response = generate_memory_based_response(user_input, memory_results)
        else:
            # 메모리가 없을 때 기본 응답
            response = generate_default_response(user_input)
    except Exception as e:
        print(f"응답 생성 오류: {e}")
        response = "죄송합니다, 처리 중 오류가 발생했습니다."
    
    print(f"[응답 생성] {response}")
    
    return {
        **state,
        "assistant_reply": response
    }

def generate_memory_based_response(user_input, memory_results):
    """메모리 기반 응답 생성"""
    # 가장 관련성 높은 메모리 추출
    top_memory = memory_results[0]["memory"]
    
    # 메모리 유형에 따른 응답 템플릿 선택
    memory_type = top_memory.get("memory_type", MemoryType.EPISODIC)
    
    # 응답 템플릿 - 메모리 유형별
    templates = {
        MemoryType.EPISODIC: [
            "네, 기억나요. {user_input}에 대해 말씀하셨죠.",
            "전에 {user_input}에 대해 이야기했었네요.",
            "{user_input}에 대해 기억하고 있어요.",
        ],
        MemoryType.SEMANTIC: [
            "{user_input}은(는) {fact}인 것으로 알고 있어요.",
            "{fact}라고 알고 있어요.",
            "{user_input}에 대해 {fact}라는 정보가 있네요.",
        ],
        MemoryType.EMOTIONAL: [
            "{user_input}에 대한 이야기를 들었을 때 {emotion} 느낌이었어요.",
            "{user_input}... 그 이야기가 {emotion} 기억으로 남아있네요.",
            "{emotion}했던 그 {user_input}에 대해 말씀하시는 거죠?",
        ],
        MemoryType.CORE: [
            "{fact}",
            "제가 알기로는 {fact}입니다.",
            "기록에 따르면 {fact}입니다.",
        ]
    }
    
    # 템플릿 선택
    template_list = templates.get(memory_type, templates[MemoryType.EPISODIC])
    template = random.choice(template_list)
    
    # 변수 준비
    context = {
        "user_input": top_memory["user"],
        "fact": f"{top_memory['user']} → {top_memory['assistant']}",
        "emotion": "인상적인"  # 실제로는 감정 분석 필요
    }
    
    # 템플릿 적용
    base_response = template.format(**context)
    
    # 사용자의 질문에 맞게 조정된 응답 생성
    if "생일" in user_input.lower() and any("생일" in m["memory"]["user"].lower() for m in memory_results):
        for mem in memory_results:
            if "생일" in mem["memory"]["user"].lower():
                # 생일 관련 메모리에서 날짜 정보 추출
                date_match = re.search(r'\d+월\s*\d+일', mem["memory"]["user"])
                if date_match:
                    return f"미키의 생일은 {date_match.group(0)}이에요. 기억하고 있어요."
    
    elif "무게" in user_input.lower() or "체중" in user_input.lower():
        for mem in memory_results:
            # 무게 관련 정보 추출
            weight_match = re.search(r'(\d+\.?\d*)kg', mem["memory"]["user"])
            if weight_match:
                return f"미키의 체중은 {weight_match.group(0)}이에요. 기억하고 있어요."
    
    # 그 외의 경우 기본 응답 사용
    return base_response

def generate_default_response(user_input):
    """관련 메모리가 없을 때의 기본 응답 생성"""
    default_responses = [
        "그것에 대해서는 기억이 없어요. 더 자세히 말씀해주실래요?",
        "흥미로운 질문이네요. 처음 듣는 내용인 것 같아요.",
        "잘 이해하지 못했어요. 다른 방식으로 물어봐 주실래요?",
        "그 부분에 대해서는 기억이 없어요. 좀 더 알려주시겠어요?",
        "더 많은 내용을 알려주시면 도움이 될 것 같아요."
    ]
    
    return random.choice(default_responses)

def store_memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """현재 대화를 STM에 저장하고 필요시 LTM으로 이전하는 노드"""
    global STM
    user_input = state["user_input"]
    assistant_reply = state["assistant_reply"]
    triggers = state["triggers"]
    importance = state["importance"]
    emotional_weight = state.get("emotional_weight", 0.0)
    
    # 메모리 유형 결정
    memory_type = MemoryType.EPISODIC  # 기본값
    
    # 중요도가 높거나 감정 가중치가 높으면 감정적 기억으로 분류
    if emotional_weight > 0.2:
        memory_type = MemoryType.EMOTIONAL
    
    # 사실 정보가 포함된 경우 (예: 이름, 날짜, 숫자 등) 의미적 기억으로 분류
    if any(pattern in user_input.lower() for pattern in ["이름", "생일", "나이", "사는 곳", "직업"]):
        memory_type = MemoryType.SEMANTIC
    
    # 직접적인 명령이나 중요 설정인 경우 핵심 기억으로 분류
    if "기억해" in user_input.lower() or importance > 0.8:
        memory_type = MemoryType.CORE
    
    # STM에 현재 대화 저장
    memory = {
        "id": str(uuid.uuid4()),
        "user": user_input,
        "assistant": assistant_reply,
        "timestamp": datetime.utcnow().isoformat(),
        "triggers": triggers,
        "importance": importance,
        "emotional_weight": emotional_weight,
        "memory_type": memory_type,
        "access_count": 0,
        "last_accessed": datetime.utcnow().isoformat(),
        "decay_score": 1.0  # 초기 decay_score = 1.0 (최대)
    }
    
    STM.append(memory)
    print(f"[STM] 저장됨: '{user_input[:30]}...' (중요도: {importance:.2f}, 유형: {memory_type})")
    
    # 관련 메모리 ID 추출 (연관기억 그래프 구축용)
    related_memory_ids = []
    if "retrieved_memories" in state and state["retrieved_memories"]:
        related_memory_ids = [mem["memory"]["id"] for mem in state["retrieved_memories"][:2]]
    
    # STM → LTM 이전 여부 판단
    memory_store = state["memory_store"]
    flush_to_ltm(memory_store, related_memory_ids)
    
    return state

def flush_to_ltm(memory_store, related_memory_ids=None, min_importance=0.5, max_turns=5):
    """STM에서 LTM으로 메모리 이전 (향상된 버전)"""
    global STM
    to_promote = []
    
    # 중요도 기준 충족 항목 식별
    for memory in STM:
        # 중요도 점수가 임계값 이상이거나 핵심/의미적/감정적 메모리인 경우
        if (memory["importance"] >= min_importance or 
            memory["memory_type"] in [MemoryType.CORE, MemoryType.SEMANTIC, MemoryType.EMOTIONAL]):
            to_promote.append(memory)
            print(f"[LTM] 중요도({memory['importance']:.2f}) 또는 유형({memory['memory_type']})으로 이전: '{memory['user'][:30]}...'")
    
    # 대화 턴 수 초과 시 전체 이전
    if len(STM) >= max_turns:
        to_promote = STM
        print(f"[LTM] 대화 턴 초과({len(STM)} >= {max_turns})로 전체 이전")
    
    # LTM으로 이전
    for memory in to_promote:
        memory_store.add_memory(memory, related_memory_ids)
    
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

def setup_memory_maintenance(memory_store):
    """
    망각 알고리즘 및 메모리 유지보수 스케줄러 설정
    - 매일 자정: 망각(decay) 적용
    - 매주 일요일: 메모리 정리(prune)
    - 매월 1일: 메모리 통합(consolidate)
    """
    # 매일 자정에 망각 알고리즘 실행
    schedule.every().day.at("00:00").do(lambda: memory_store.apply_decay())
    
    # 매주 일요일에 메모리 정리
    schedule.every().sunday.at("01:00").do(lambda: memory_store.prune_memories())
    
    # 매월 1일에 메모리 통합 - day.at 사용하고 월초 확인 로직 추가
    def monthly_task():
        # 현재 날짜가 1일인 경우에만 실행
        if datetime.utcnow().day == 1:
            memory_store.consolidate_memories()
    
    schedule.every().day.at("02:00").do(monthly_task)
    
    # 스케줄러 백그라운드 실행
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    print("[스케줄러] 메모리 유지보수 스케줄러 설정 완료")

def run_scheduler():
    """스케줄러 실행 루프"""
    while True:
        schedule.run_pending()
        time.sleep(60)

# ----------------- 7. 메인 챗봇 클래스 -----------------

class EnhancedMemoryChatbot:
    """향상된 장기 기억 기반 챗봇 클래스"""
    
    def __init__(self, collection_name="memory_chatbot", persist_dir="./chroma_db"):
        print("\n=== 향상된 장기 기억 기반 챗봇 시스템 초기화 중... ===\n")
        
        # 벡터 DB 초기화
        self.memory_store = EnhancedMemoryStore(
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
        setup_memory_maintenance(self.memory_store)
        
        # 정기적인 decay 적용
        self.memory_store.apply_decay()
        
        print("\n=== 향상된 장기 기억 기반 챗봇 시스템 초기화 완료 ===\n")
    
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
        """초기 메모리로 시스템 시드 설정 (향상된 버전)"""
        print("\n[초기 메모리 시드 설정]\n")
        
        for i, (user_msg, bot_reply, memory_type) in enumerate(memories):
            print(f"메모리 {i+1}: '{user_msg}'")
            
            # 트리거 추출
            triggers, importance, emotional_weight = extract_triggers_enhanced(user_msg)
            
            # 메모리 생성
            memory = {
                "id": f"seed_{i}_{uuid.uuid4()}",
                "user": user_msg,
                "assistant": bot_reply,
                "timestamp": datetime.utcnow().isoformat(),
                "triggers": triggers,
                "importance": importance,
                "emotional_weight": emotional_weight,
                "memory_type": memory_type,
                "access_count": 0,
                "last_accessed": datetime.utcnow().isoformat(),
                "decay_score": 1.0
            }
            
            # 벡터 DB에 저장
            self.memory_store.add_memory(memory)
            print(f"  -> 챗봇: '{bot_reply}'")
            print(f"  -> 트리거: {triggers}, 중요도: {importance:.2f}, 유형: {memory_type}")
            print()

# ----------------- 8. 실행 예시 -----------------

def main():
    """챗봇 실행 예시"""
    
    # 챗봇 초기화
    chatbot = EnhancedMemoryChatbot()
    
    # 초기 대화 샘플로 메모리 채우기
    seed_memories = [
        # (사용자 입력, 챗봇 응답, 메모리 유형)
        ("우리 미키는 5.6kg인 토이푸들이야", "미키가 5.6kg인 토이푸들이군요! 귀여운 크기네요.", MemoryType.SEMANTIC),
        ("미키 생일은 3월 12일이야", "3월 12일이 미키의 생일이군요! 기억해두겠습니다.", MemoryType.SEMANTIC),
        ("미키는 노란 공 장난감을 좋아해", "미키가 노란 공 장난감을 좋아하는군요. 장난감 취향도 있네요!", MemoryType.EPISODIC),
        ("나는 어제 회사에서 발표를 했어", "회사에서 발표하셨군요! 발표는 잘 진행되었나요?", MemoryType.EPISODIC),
        ("발표할 때 너무 긴장돼서 식은땀을 흘렸어", "발표 때 긴장하셨군요. 많이 불안하셨겠어요.", MemoryType.EMOTIONAL)
    ]
    
    # 메모리 시드 설정
    chatbot.seed_with_memories(seed_memories)
    
    # 대화 예시
    test_queries = [
        "미키 생일이 언제였지?",
        "우리 강아지 무게가 얼마였더라?",
        "내가 어제 뭐 했었지?",
        "발표할 때 기분이 어땠어?"
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