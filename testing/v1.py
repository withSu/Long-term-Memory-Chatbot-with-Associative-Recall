import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import uuid
from datetime import datetime
import math
import threading
import time
import schedule

# 1. 한글 지원을 위한 모델 설정
# 참고: 한글 NER을 위해서는 ko_core_news_md 또는 ko_core_news_lg 모델이 필요합니다
# pip install https://github.com/explosion/spacy-models/releases/download/ko_core_news_md-3.6.0/ko_core_news_md-3.6.0-py3-none-any.whl

try:
    nlp = spacy.load("ko_core_news_md")
except OSError:
    # 한글 모델이 없는 경우 영어 모델로 대체 (실제 사용시 한글 모델 설치 필요)
    nlp = spacy.load("en_core_web_sm")
    print("경고: 한글 NER 모델(ko_core_news_md)이 설치되지 않았습니다. 영어 모델로 대체합니다.")
    print("한글 모델 설치: pip install https://github.com/explosion/spacy-models/releases/download/ko_core_news_md-3.6.0/ko_core_news_md-3.6.0-py3-none-any.whl")

# 사용자 정의 트리거 단어 (한글 지원)
TRIGGER_WORDS = {"미키", "생일", "강아지", "가족", "취미", "체중", "선물", "토이푸들"}

# 무료 임베딩 모델 로드 - 다국어 모델 중 하나를 선택
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 메모리 저장소 초기화
STM = []  # 단기 메모리
LTM = []  # 장기 메모리

# ----------------- 1단계: 트리거 단어 감지 및 중요도 평가 -----------------

def extract_triggers_and_score(text):
    """
    사용자 입력에서 트리거 단어를 감지하고 중요도 점수를 계산
    
    Args:
        text (str): 사용자 입력 텍스트
        
    Returns:
        tuple: (트리거 단어 리스트, 중요도 점수)
    """
    doc = nlp(text)
    triggers = set()
    
    # NER 기반 키워드 추출 (한글의 경우 PER, LOC, DAT 등의 태그 사용)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "DATE", "LOC", "ORG", "CARDINAL", "PER"}:
            triggers.add(ent.text.lower())
    
    # 사용자 정의 트리거 단어 감지
    for token in doc:
        if token.text.lower() in TRIGGER_WORDS:
            triggers.add(token.text.lower())
    
    # 중요도 점수 계산: 트리거 수에 비례
    # 기본 점수 0.2, 최대 5개 트리거로 1.0에 도달
    score = min(len(triggers) * 0.2, 1.0)
    
    # 추가: 감정 표현이 있는 경우 중요도 가중치 부여
    # 한글에서는 감정 분석 로직 추가 필요 (현재는 간단한 규칙 기반)
    if any(word in text.lower() for word in ["좋아", "사랑", "행복", "슬퍼", "화나", "기뻐"]):
        score += 0.1
        score = min(score, 1.0)  # 최대 1.0으로 제한
    
    return list(triggers), score

# ----------------- 2단계: STM -> LTM 이전 및 임베딩 벡터 저장 -----------------

def store_to_stm(user_input, assistant_reply, triggers, importance_score):
    """
    단기 메모리(STM)에 대화 저장
    
    Args:
        user_input (str): 사용자 입력
        assistant_reply (str): 챗봇 응답
        triggers (list): 감지된 트리거 단어 목록
        importance_score (float): 중요도 점수
    """
    STM.append({
        "id": str(uuid.uuid4()),
        "user": user_input,
        "assistant": assistant_reply,
        "timestamp": datetime.utcnow().isoformat(),
        "triggers": triggers,
        "importance": importance_score,
        "access_count": 0,
        "last_accessed": datetime.utcnow().isoformat()
    })
    
    print(f"[DEBUG] STM에 저장됨: '{user_input[:30]}...' (중요도: {importance_score:.2f})")

def flush_stm_to_ltm(min_importance=0.5, max_turns=5):
    """
    단기 메모리(STM)에서 장기 메모리(LTM)로 이전
    - 중요도가 높거나 대화 턴이 일정 수를 초과하면 이전
    
    Args:
        min_importance (float): 이전을 위한 최소 중요도 점수
        max_turns (int): 이전을 위한 최대 대화 턴 수
    """
    global STM, LTM
    to_promote = []
    
    # 중요도 기준 충족 항목 식별
    for memory in STM:
        if memory["importance"] >= min_importance:
            to_promote.append(memory)
            print(f"[DEBUG] 중요도({memory['importance']:.2f})로 LTM 이전 대상: '{memory['user'][:30]}...'")
    
    # 대화 턴 수 초과 시 전체 이전
    if len(STM) >= max_turns:
        to_promote = STM
        print(f"[DEBUG] 대화 턴 초과({len(STM)} >= {max_turns})로 전체 STM을 LTM으로 이전")
    
    # LTM으로 이전 및 임베딩 생성
    for memory in to_promote:
        # 사용자 입력과 챗봇 응답을 결합하여 컨텍스트 형성
        full_text = memory["user"] + " " + memory["assistant"]
        # 임베딩 생성 (다국어 모델 사용)
        embedding = model.encode(full_text)
        memory["embedding"] = embedding.tolist()  # numpy 배열을 리스트로 변환하여 저장
        LTM.append(memory)
        print(f"[DEBUG] LTM에 저장 완료: '{memory['user'][:30]}...'")
    
    # 이전된 항목은 STM에서 제거
    if to_promote:
        STM = [m for m in STM if m not in to_promote]
        print(f"[DEBUG] LTM으로 이전 후 STM 크기: {len(STM)}")

# ----------------- 3단계: 트리거 단어 기반 검색 -----------------

def search_ltm_by_trigger(query_text, trigger_words, top_k=3):
    """
    장기 메모리에서 트리거 단어 기반으로 관련 메모리 검색
    
    Args:
        query_text (str): 현재 사용자 질의
        trigger_words (list): 감지된 트리거 단어 목록
        top_k (int): 반환할 최대 결과 수
        
    Returns:
        list: 유사도 점수를 포함한 관련 메모리 목록
    """
    if not LTM or not trigger_words:
        return []
    
    # 쿼리 임베딩 생성
    query_embedding = model.encode(query_text).reshape(1, -1)
    
    # 1차 필터: 트리거 단어가 포함된 메모리만 선택
    filtered_memories = [
        mem for mem in LTM if any(trigger in mem["triggers"] for trigger in trigger_words)
    ]
    
    if not filtered_memories:
        print(f"[DEBUG] 트리거 단어({trigger_words})와 일치하는 메모리 없음")
        return []
    
    print(f"[DEBUG] 트리거 단어({trigger_words})로 {len(filtered_memories)}개 메모리 필터링됨")
    
    # 2차: 임베딩 유사도 기반 정렬
    memory_embeddings = np.array([mem["embedding"] for mem in filtered_memories])
    similarities = cosine_similarity(query_embedding, memory_embeddings)[0]
    
    # 유사도 기준 상위 k개 인덱스 선택
    top_indices = similarities.argsort()[::-1][:top_k]
    
    # 선택된 메모리 접근 카운트 증가 및 마지막 접근 시간 갱신
    results = []
    for i in top_indices:
        filtered_memories[i]["access_count"] += 1
        filtered_memories[i]["last_accessed"] = datetime.utcnow().isoformat()
        
        results.append({
            "memory": filtered_memories[i],
            "similarity": similarities[i]
        })
        
        print(f"[DEBUG] 유사도 {similarities[i]:.4f}로 '{filtered_memories[i]['user'][:30]}...' 검색됨")
    
    return results

# ----------------- 4단계: 검색된 기억을 활용한 챗봇 응답 생성 -----------------

# 참고: 실제 서비스라면 OpenAI API나 다른 LLM API를 사용하는 것이 좋음
# 예제에서는 단순 Hugging Face 모델로 대체

# transformers 모델 로드 (실제 서비스시 더 강력한 모델로 대체 필요)
try:
    chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")
except Exception as e:
    print(f"모델 로드 실패: {e}")
    # 모델 로드 실패시 간단한 응답 함수로 대체
    def simple_response(text):
        if "미키" in text:
            return "네, 미키에 대해 기억하고 있어요!"
        return "더 자세히 말씀해주세요."
    
    # 간단한 응답 generator 래퍼
    class DummyChatbot:
        def __call__(self, prompt, **kwargs):
            return [{"generated_text": prompt + " " + simple_response(prompt)}]
    
    chatbot = DummyChatbot()

def generate_reply_with_memory(user_input, memory_results):
    """
    검색된 기억을 활용하여 챗봇 응답 생성
    
    Args:
        user_input (str): 사용자 입력
        memory_results (list): 검색된 메모리 결과
        
    Returns:
        str: 생성된 챗봇 응답
    """
    # 관련 기억이 없는 경우
    if not memory_results:
        prompt = f"User: {user_input}\nAssistant:"
        try:
            reply = chatbot(prompt, max_length=100, do_sample=True, top_k=50)[0]["generated_text"]
            return reply[len(prompt):].strip()
        except Exception as e:
            print(f"응답 생성 오류: {e}")
            return "무슨 말씀인지 잘 이해하지 못했어요. 더 자세히 말씀해주세요."
    
    # 기억 내용을 텍스트로 요약
    memory_context = "\n".join(
        f"- {mem['memory']['user']} → {mem['memory']['assistant']}"
        for mem in memory_results
    )
    
    # 프롬프트 구성 (한글 지원)
    prompt = (
        "다음은 과거 대화 내용입니다:\n"
        f"{memory_context}\n\n"
        f"User: {user_input}\n"
        "Assistant:"
    )
    
    try:
        # 응답 생성
        reply = chatbot(prompt, max_length=150, do_sample=True, top_k=50)[0]["generated_text"]
        
        # 응답에서 프롬프트 부분 제거
        clean_reply = reply[len(prompt):].strip()
        
        # 응답이 없거나 너무 짧은 경우 기본 응답
        if not clean_reply or len(clean_reply) < 5:
            memory_summary = ", ".join([m["memory"]["user"] for m in memory_results[:2]])
            return f"네, 기억나요. {memory_summary}에 대해 말씀하셨었죠."
            
        return clean_reply
    except Exception as e:
        print(f"응답 생성 오류: {e}")
        # 오류 발생시 간단한 기억 기반 응답
        memory_item = memory_results[0]["memory"]["user"]
        return f"네, 예전에 '{memory_item}'에 대해 말씀하셨던 것 같아요."

# ----------------- 5단계: 기억 점수 기반 망각 알고리즘 -----------------

def decay_and_prune_ltm(current_time=None, decay_rate=0.05, score_threshold=0.4):
    """
    시간 경과에 따른 기억 점수 감소 및 임계값 이하 항목 제거
    
    Args:
        current_time (datetime): 현재 시간 (None이면 현재 시간 사용)
        decay_rate (float): 일별 감소율
        score_threshold (float): 유지할 최소 점수
    """
    global LTM
    if not current_time:
        current_time = datetime.utcnow()
    
    to_keep = []
    removed = 0
    
    for mem in LTM:
        # 마지막 접근 시간으로부터 경과된 일수 계산
        last_accessed = datetime.fromisoformat(mem["last_accessed"])
        delta_days = (current_time - last_accessed).days
        
        # 접근 횟수 가져오기 (없으면 0)
        access_count = mem.get("access_count", 0)
        
        # 기본 점수 (중요도)
        base_score = mem["importance"]
        
        # 최종 점수 계산: 기본 점수 + 로그(접근 횟수 + 1) - (감소율 * 경과 일수)
        # 접근 횟수가 많을수록 점수 증가, 시간이 지날수록 점수 감소
        score = base_score + math.log(1 + access_count) - decay_rate * delta_days
        mem["score"] = score  # 계산된 점수 저장
        
        # 임계값 이상이면 유지
        if score >= score_threshold:
            to_keep.append(mem)
        else:
            removed += 1
    
    # 유지할 메모리만 남기기
    LTM = to_keep
    
    print(f"[DEBUG] 망각 알고리즘 실행: {removed}개 메모리 제거됨, {len(LTM)}개 유지됨")

# ----------------- 6단계: LangGraph 통합 -----------------

# LangGraph가 설치되어 있다면 여기에 워크플로우 정의 (선택 사항)
# 아래는 LangGraph가 없을 때 대신 사용할 수 있는 간단한 상태 관리

class SimpleChatWorkflow:
    """
    LangGraph 없이 간단한 워크플로우 관리를 위한 클래스
    """
    def __init__(self):
        self.setup_scheduler()
    
    def setup_scheduler(self):
        """망각 알고리즘 스케줄러 설정"""
        # 매일 자정에 망각 알고리즘 실행
        schedule.every().day.at("00:00").do(
            lambda: decay_and_prune_ltm(datetime.utcnow())
        )
        
        # 스케줄러 백그라운드 실행
        self.scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
        self.scheduler_thread.start()
    
    def run_scheduler(self):
        """스케줄러 실행 루프"""
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def process_message(self, user_input):
        """
        사용자 메시지 처리 파이프라인
        
        Args:
            user_input (str): 사용자 입력
            
        Returns:
            dict: 처리 결과 (응답 및 상태 정보 포함)
        """
        # 상태 초기화
        state = {"user_input": user_input}
        
        # 1. 트리거 추출 및 중요도 평가
        triggers, importance = extract_triggers_and_score(user_input)
        state["triggers"] = triggers
        state["importance"] = importance
        
        # 2. LTM에서 관련 기억 검색
        memory_results = search_ltm_by_trigger(user_input, triggers)
        state["retrieved_memories"] = memory_results
        
        # 3. 검색된 기억 기반 응답 생성
        response = generate_reply_with_memory(user_input, memory_results)
        state["llm_response"] = response
        
        # 4. STM에 현재 대화 저장
        store_to_stm(user_input, response, triggers, importance)
        
        # 5. 필요시 STM → LTM 이전
        flush_stm_to_ltm()
        
        return state

# ----------------- 예시 실행 코드 -----------------

def main():
    """
    메인 실행 함수
    """
    print("=== 장기 기억 기반 챗봇 시스템 시작 ===")
    
    # 워크플로우 초기화
    workflow = SimpleChatWorkflow()
    
    # 초기 대화 샘플로 메모리 채우기
    samples = [
        "우리 미키는 5.6kg인 토이푸들이야",
        "미키 생일은 3월 12일이야",
        "미키는 노란 공 장난감을 좋아해",
        "나는 어제 회사에서 발표를 했어"
    ]
    
    # 샘플 대화로 메모리 초기화
    for sample in samples:
        print(f"\n사용자: {sample}")
        state = workflow.process_message(sample)
        print(f"챗봇: {state['llm_response']}")
    
    # 사용자 입력 처리 (예시)
    test_queries = [
        "미키 생일이 언제였지?",
        "우리 강아지 무게가 얼마였더라?",
        "내가 어제 뭐 했었지?"
    ]
    
    for query in test_queries:
        print(f"\n사용자: {query}")
        state = workflow.process_message(query)
        print(f"챗봇: {state['llm_response']}")
        print(f"활성화된 기억: {[m['memory']['user'][:20] + '...' for m in state['retrieved_memories']]}")

if __name__ == "__main__":
    main()