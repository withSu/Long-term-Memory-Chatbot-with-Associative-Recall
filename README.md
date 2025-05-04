# 실시간 연관 기억 챗봇 시스템

<img src="/images/image.png" alt="연관 기억 네트워크 시각화 예시" />

## 목차
1. [개요](#개요)
2. [핵심 기능](#핵심-기능)
3. [전체 시스템 아키텍처](#전체-시스템-아키텍처)
4. [데이터 흐름 및 처리 과정](#데이터-흐름-및-처리-과정)
5. [기억 및 회상 메커니즘](#기억-및-회상-메커니즘)
6. [핵심 클래스 및 책임](#핵심-클래스-및-책임)
7. [가중치 시스템 상세](#가중치-시스템-상세)
8. [대화 예시와 메모리 구축 과정](#대화-예시와-메모리-구축-과정)
9. [시각화 기능](#시각화-기능)
10. [상세 사용 가이드](#상세-사용-가이드)
11. [성능 최적화 및 확장성](#성능-최적화-및-확장성)
12. [문제 해결](#문제-해결)
13. [개발 가이드](#개발-가이드)
14. [시스템 한계 및 향후 개선 방향](#시스템-한계-및-향후-개선-방향)

## 개요

실시간 연관 기억 챗봇 시스템은 사람의 인지 구조에서 영감을 받은 고급 대화형 AI 시스템입니다. 일반적인 챗봇과 달리, 이 시스템은 대화 내용을 단순히 저장하는 것을 넘어 개념 간의 연관 관계를 구축하고 '연결망' 형태로 기억을 관리합니다. 이는 인간의 연상 메모리와 유사한 방식으로, 사용자와의 대화가 쌓일수록 더 풍부한 맥락적 이해와 회상 능력을 갖추게 됩니다.

## 핵심 기능

- **계층적 메모리 아키텍처**: 워킹 메모리 → 단기 메모리 → 장기 메모리 구조로 효율적 기억 관리
- **연관 네트워크 구축**: 대화 내용에서 추출한 개념(키워드) 간의 관계를 자동으로 연결하고 강화
- **개념 기반 검색**: 단순 키워드 검색이 아닌 연관 관계를 따라 의미적으로 연결된 기억 검색
- **망각 및 강화 메커니즘**: 중요한 기억은 강화되고, 덜 중요한 기억은 시간이 지남에 따라 약화
- **실시간 시각화**: 형성된 연관 네트워크를 실시간으로 시각화하여 기억 구조 확인 가능

## 전체 시스템 아키텍처

### 아키텍처 다이어그램

```
┌───────────────────────────────────────────────────────────────────────┐
│                              사용자 인터페이스                            │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│                      RealtimeAssociativeChatbot                       │
│                                                                       │
│  ┌─────────────────┐  ┌───────────────────┐  ┌────────────────────┐  │
│  │  ConversationSes│  │  회상 및 응답 생성   │  │  시각화 엔진         │  │
│  │  sion 관리      │  │                   │  │                    │  │
│  └─────────────────┘  └───────────────────┘  └────────────────────┘  │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              메모리 관리 계층                              │
│                                                                          │
│ ┌────────────────────┐ ┌───────────────────┐ ┌─────────────────────────┐ │
│ │   MemoryManager    │ │ AssociationNetwork │ │    LifecycleManager     │ │
│ │ • 메모리 계층 통합   │ │ • 개념 간 연결 관리  │ │ • 승격/강등 관리        │ │
│ │ • 검색 및 저장 조정  │ │ • 확산 활성화       │ │ • 망각 프로세스          │ │
│ └────────────────────┘ └───────────────────┘ └─────────────────────────┘ │
└───────────┬────────────────────┬────────────────────────┬────────────────┘
            │                    │                        │
            ▼                    ▼                        ▼
┌─────────────────┐   ┌──────────────────┐   ┌───────────────────────────┐
│  워킹 메모리     │   │    단기 메모리     │   │       장기 메모리          │
│   (Redis)      │──▶│    (SQLite)      │──▶│       (ChromaDB)          │
│ • TTL: 30분    │   │ • 보존: ~30일     │   │ • 영구 저장                │
│ • 즉시 접근 필요 │   │ • 주기적 접근 필요  │   │ • 의미 기반 벡터 검색      │
└─────────────────┘   └──────────────────┘   └───────────────────────────┘
```

### 모듈 구조

```
project/
├── main.py                   # 애플리케이션 진입점
├── config/                   # 설정 관리
│   ├── __init__.py
│   └── settings.py          # 시스템 설정 클래스
├── models/                   # 데이터 모델 정의
│   ├── __init__.py
│   ├── enums.py             # 열거형 정의 (메모리 계층, 연결 유형 등)
│   └── memory_entry.py      # 메모리 엔트리 데이터 클래스
├── storage/                  # 저장소 인터페이스
│   ├── __init__.py
│   ├── redis_storage.py     # Redis 워킹 메모리 인터페이스
│   ├── sqlite_storage.py    # SQLite 단기 메모리 인터페이스
│   └── vector_storage.py    # ChromaDB 장기 메모리 인터페이스
├── core/                     # 핵심 로직
│   ├── __init__.py
│   ├── memory_manager.py    # 메모리 통합 관리
│   ├── association_network.py # 연관 네트워크 관리
│   └── lifecycle_manager.py # 메모리 생명주기 관리
├── utils/                    # 유틸리티
│   ├── __init__.py
│   ├── cache.py             # LRU 캐시 구현
│   ├── bloom_filter.py      # 블룸 필터 래퍼
│   └── visualization.py     # 네트워크 시각화
└── chatbot/                  # 챗봇 인터페이스
    ├── __init__.py
    ├── chatbot.py           # 메인 챗봇 클래스
    └── session.py           # 대화 세션 관리
```

## 데이터 흐름 및 처리 과정

### 전체 데이터 흐름도

```
┌───────────────┐    ┌──────────────┐    ┌────────────────┐    ┌────────────────┐
│  사용자 입력   │───▶│  개념 추출    │───▶│  병렬 메모리 검색  │───▶│  네트워크 탐색   │
└───────────────┘    └──────────────┘    └────────────────┘    └────────────────┘
        │                                                              │
        │                                                              ▼
┌───────────────┐    ┌──────────────┐    ┌────────────────┐    ┌────────────────┐
│  메모리 저장   │◀───│  연관 업데이트  │◀───│   응답 생성     │◀───│  결과 병합/순위  │
└───────────────┘    └──────────────┘    └────────────────┘    └────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                                  백그라운드 프로세스                             │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐  ┌──────────────┐│
│  │  승격 프로세스   │  │  망각 프로세스   │  │  세션 정리 프로세스 │  │  모니터링     ││
│  └────────────────┘  └────────────────┘  └─────────────────┘  └──────────────┘│
└───────────────────────────────────────────────────────────────────────────────┘
```

### 1. 사용자 입력 처리

사용자가 메시지를 입력하면, 다음과 같은 과정이 진행됩니다:

**실행 경로**: `chatbot.py` → `chat()` 메서드

```python
async def chat(self, user_id: str, message: str) -> str:
    # 1. 세션 관리
    session = self._get_or_create_session(user_id)
    
    # 2. 개념 추출
    concepts = self._extract_concepts(message)
    
    # 3. 연관 검색
    search_results = await self.memory_manager.search_memories(concepts, message)
    
    # 4. 네트워크 연관 분석
    network_associations = self.association_network.find_associations(concepts)
    
    # 5. 응답 생성
    response = await self._generate_response(
        user_input=message,
        search_results=search_results,
        network_associations=network_associations,
        session_context=session.get_context()
    )
    
    # 6-7. 메모리 저장 및 연관 업데이트
    memory_id = await self._store_memory(user_id, message, response, concepts)
    await self._update_associations(concepts, memory_id)
    
    # 8. 세션 업데이트
    session.update(message, response, concepts)
    
    return response
```

### 2. 연관 기억 검색

추출된 개념을 기반으로 여러 메모리 계층에서 관련 기억을 병렬로 검색합니다:

**실행 경로**: `memory_manager.py` → `search_memories()` 메서드

```python
async def search_memories(self, concepts: List[str], query_text: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
    # 캐시 확인
    cache_key = f"search:{':'.join(concepts)}"
    cached_result = self.search_cache.get(cache_key)
    if cached_result:
        return cached_result
    
    # 병렬 검색 실행
    search_tasks = [
        self._search_working_memory(concepts),
        self._search_short_term(concepts),
        self._search_long_term(concepts, query_text)
    ]
    
    results = await asyncio.gather(*search_tasks)
    
    # 연관 네트워크 결과 포함
    network_results = {}
    if hasattr(self, 'network') and self.network:
        network_results = self.network.find_associations(concepts, depth=2)
    
    # 결과 병합 및 재순위
    merged_results = self._merge_search_results(results, network_results, concepts)
    
    # 캐시에 저장
    self.search_cache.put(cache_key, merged_results)
    
    return merged_results
```

## 기억 및 회상 메커니즘

### 기억 저장 흐름도

```
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────────┐
│  대화 내용 분석  │────▶│  중요도/감정 계산   │────▶│  메모리 엔트리 생성    │
└─────────────────┘     └───────────────────┘     └─────────────────────┘
                                                            │
                                                            ▼
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────────┐
│  개념 간 연결 생성 │◀────│   개념 활성화      │◀────│   워킹 메모리 저장    │
└─────────────────┘     └───────────────────┘     └─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             주기적 생명주기 관리                           │
│  ┌────────────────┐    ┌─────────────────┐    ┌────────────────────┐    │
│  │중요 메모리 승격   │───▶│ 덜 중요한 메모리 강등│───▶│  연관 강도 감소     │    │
│  └────────────────┘    └─────────────────┘    └────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 회상 과정 흐름도

```
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────────┐
│  개념 추출       │────▶│  워킹 메모리 검색    │────▶│  단기 메모리 검색     │
└─────────────────┘     └───────────────────┘     └─────────────────────┘
                                                            │
                                                            ▼
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────────┐
│  결과 종합/순위   │◀────│  연관 경로 탐색     │◀────│  장기 메모리 검색     │
└─────────────────┘     └───────────────────┘     └─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             후처리 및 최적화                              │
│  ┌────────────────┐    ┌─────────────────┐    ┌────────────────────┐    │
│  │검색된 메모리 강화  │───▶│  캐시 업데이트     │───▶│ 연관 구조 업데이트   │    │
│  └────────────────┘    └─────────────────┘    └────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 핵심 클래스 및 책임

### RealtimeAssociativeChatbot
- **파일**: `chatbot/chatbot.py`
- **역할**: 전체 시스템 통합 및 대화 인터페이스 제공
- **주요 메서드**:
  - `chat()`: 대화 처리 메인 파이프라인
  - `_extract_concepts()`: 텍스트에서 개념 추출
  - `_generate_response()`: 검색 결과 기반 응답 생성
  - `_store_memory()`: 새 메모리 저장 처리
  - `visualize_associations()`: 연관 네트워크 시각화

### MemoryManager
- **파일**: `core/memory_manager.py`
- **역할**: 다양한 메모리 계층 통합 관리
- **주요 메서드**:
  - `search_memories()`: 모든 메모리 계층에서 통합 검색
  - `save_memory()`: 새 메모리 저장
  - `_merge_search_results()`: 여러 계층의 검색 결과 병합
  - `_search_working_memory()`: Redis 워킹 메모리 검색
  - `_search_short_term()`: SQLite 단기 메모리 검색 
  - `_search_long_term()`: ChromaDB 장기 메모리 검색

### AssociationNetwork
- **파일**: `core/association_network.py`
- **역할**: 개념 간 연관 관계 관리
- **주요 메서드**:
  - `activate_concept()`: 개념 활성화
  - `connect_concepts()`: 개념 간 연결 생성
  - `find_associations()`: 개념 관련 연관 검색
  - `_analyze_co_activation()`: 동시 활성화 패턴 분석
  - `apply_decay()`: 연결 강도 감소 적용

### LifecycleManager
- **파일**: `core/lifecycle_manager.py`
- **역할**: 메모리 생명주기 및 자원 관리
- **주요 메서드**:
  - `run_lifecycle_cycle()`: 모든 생명주기 처리 실행
  - `_check_promotions()`: 메모리 승격 조건 확인 및 처리
  - `_check_demotions()`: 메모리 강등 조건 확인 및 처리
  - `_apply_decay()`: 연결 강도 감소 적용

### StorageInterface 구현체
- **파일**: `storage/redis_storage.py`, `storage/sqlite_storage.py`, `storage/vector_storage.py`
- **역할**: 각 메모리 계층의 저장소 인터페이스 제공
- **주요 메서드**:
  - `save()`: 메모리 저장
  - `find_by_concepts()`: 개념으로 메모리 검색
  - `get()`: ID로 메모리 조회
  - `delete()`: 메모리 삭제

## 가중치 시스템 상세

시스템은 여러 유형의 가중치를 사용하여 기억과 연관 관계를 관리합니다. 각 가중치의 상세 내용과 계산 방식은 다음과 같습니다:

### 1. 메모리 중요도 (Importance)

메모리의 중요성을 나타내는 0.0~1.0 사이의 값으로, 메모리의 장기 보존 여부와 검색 우선순위에 영향을 줍니다.

**계산 방식**: `_calculate_importance()` 메서드
```python
def _calculate_importance(self, user_input: str, assistant_reply: str) -> float:
    # 길이 기반 점수 (긴 대화일수록 중요)
    length_score = (len(user_input) + len(assistant_reply)) / 200  # 최대 1.0
    
    # 키워드 기반 점수
    important_keywords = ['중요', '약속', '기억', '생일', '전화번호', '주소']
    keyword_score = sum(1 for k in important_keywords if k in user_input) / len(important_keywords)
    
    # 질문 기반 점수
    question_score = 0.2 if '?' in user_input else 0.0
    
    # 가중 평균
    total_score = (length_score * 0.4 + keyword_score * 0.4 + question_score * 0.2)
    return min(total_score, 1.0)
```

**영향**: 
- 중요도 ≥ 0.7: 단기 메모리로 승격 대상
- 중요도 ≥ 0.8: 장기 메모리로 승격 대상
- 중요도 < 0.3: 빠른 망각 대상

### 2. 감정 가중치 (Emotional Weight)

메모리의 감정적 중요성을 나타내는 0.0~1.0 사이의 값으로, 개념 간 연결 형성 및 강도에 영향을 줍니다.

**계산 방식**: `_calculate_emotional_weight()` 메서드
```python
def _calculate_emotional_weight(self, text: str) -> float:
    positive_keywords = ['좋아', '행복', '사랑', '기쁘', '즐겁']
    negative_keywords = ['싫어', '슬프', '화나', '짜증', '우울']
    
    positive_count = sum(1 for k in positive_keywords if k in text)
    negative_count = sum(1 for k in negative_keywords if k in text)
    
    if positive_count > 0:
        # 긍정적 감정: 0.5~1.0
        return 0.5 + min(positive_count * 0.1, 0.5)
    elif negative_count > 0:
        # 부정적 감정: 0.0~0.5
        return max(0.5 - negative_count * 0.1, 0.0)
    else:
        # 중립: 0.5
        return 0.5
```

**영향**:
- 감정 가중치 > 0.7: '감정적' 유형의 강한 연결 생성
- 감정 가중치 < 0.3: 약한 연결 생성, 빠른 망각 가능성

### 3. 연결 강도 (Connection Strength)

개념 간 연관 관계의 강도를 나타내는 0.0~1.0 사이의 값으로, 검색 과정에서 연관 경로 탐색에 영향을 줍니다.

**기본값**: 연결 유형별로 사전 정의
```python
self.connection_strengths = {
    ConnectionType.SEMANTIC: 0.8,    # 의미적 유사성
    ConnectionType.TEMPORAL: 0.6,    # 시간적 공동 발생
    ConnectionType.CAUSAL: 0.9,      # 인과 관계 
    ConnectionType.EMOTIONAL: 0.7,   # 감정적 연관
    ConnectionType.SPATIAL: 0.5,     # 공간적 관계
    ConnectionType.PROCEDURAL: 0.4   # 절차적 관계
}
```

**강화 메커니즘**: 동일 문맥에서 반복 등장 시 강화
```python
def _strengthen_connection(self, from_concept: str, to_concept: str, delta: float):
    """연결 강도 증가"""
    if self.graph.has_edge(from_concept, to_concept):
        edge_data = self.graph[from_concept][to_concept]
        edge_data['weight'] = min(1.0, edge_data['weight'] + delta)
        edge_data['strengthening_count'] += 1
    else:
        # 새 연결 생성
        self.connect_concepts(from_concept, to_concept, custom_strength=delta)
```

**영향**:
- 강도 < 0.1: 연결 제거 (망각)
- 강도 ≥ 0.7: 검색 시 높은 우선순위 부여
- 강도에 따른 시각화에서 엣지 두께 결정

### 4. 검색 점수 (Search Score)

검색 결과의 관련성을 나타내는 0.0~1.0 사이의 값으로, 검색 결과 순위에 영향을 줍니다.

**계산 방식**: 메모리 계층과 관련성에 따라 다양하게 계산
```python
def _merge_search_results(self, search_results, network_results, query_concepts):
    # 워킹 메모리: 최고 우선순위
    for memory in search_results[0]:
        all_results.append({
            'memory': memory,
            'score': 1.0,  # 워킹 메모리는 최고 점수
            'source': 'working'
        })
    
    # 단기 메모리: 중요도와 개념 매칭 점수 조합
    for memory in search_results[1]:
        concept_match_score = sum(1 for c in memory.concepts 
                               if c in query_concepts) / len(query_concepts)
        score = 0.8 * memory.importance + 0.2 * concept_match_score
        all_results.append({
            'memory': memory,
            'score': score,
            'source': 'short_term'
        })
    
    # 장기 메모리: 벡터 유사도 기반
    for result in search_results[2]:
        score = 0.6 * result['similarity']
        all_results.append({
            'memory': result,
            'score': score,
            'source': 'long_term'
        })
```

**영향**:
- 점수가 높을수록 상위에 랭크되어 응답 생성에 더 많은 영향
- 동일 메모리가 여러 경로로 검색될 경우 가장 높은 점수만 반영

### 5. 생명주기 제어 매개변수

메모리의 승격, 강등 및 망각을 제어하는 설정값들입니다.

**승격 규칙**:
```python
self.promotion_rules = {
    MemoryTier.WORKING: {
        'importance_threshold': 0.7,    # 최소 중요도
        'access_threshold': 3,          # 최소 접근 횟수 
        'age_threshold': timedelta(hours=6)  # 최소 경과 시간
    },
    MemoryTier.SHORT_TERM: {
        'importance_threshold': 0.8,
        'access_threshold': 5,
        'age_threshold': timedelta(days=7)
    }
}
```

**강등 규칙**:
```python
self.demotion_rules = {
    MemoryTier.WORKING: {
        'max_age': timedelta(hours=24),           # 최대 보존 시간
        'inactivity_threshold': timedelta(hours=12)  # 비활성 시간
    },
    MemoryTier.SHORT_TERM: {
        'max_age': timedelta(days=30),
        'inactivity_threshold': timedelta(days=14)
    }
}
```
**망각 메커니즘**:
```python
def apply_decay(self, time_passed=None):
    """시간 경과에 따른 연결 강도 감소"""
    if time_passed is None:
        time_passed = timedelta(days=1)
    
    # 경과 일수에 따른 감소율 계산
    decay_amount = 1 - (self.decay_factor ** time_passed.days)
    
    # 모든 연결에 감소 적용
    edges_to_remove = []
    for u, v, data in self.graph.edges(data=True):
        new_weight = data['weight'] * (1 - decay_amount)
        if new_weight < self.min_connection_strength:
            edges_to_remove.append((u, v))  # 약한 연결 제거 대상
        else:
            data['weight'] = new_weight     # 강도 감소
    
    # 약한 연결 제거 (망각)
    for u, v in edges_to_remove:
        self.graph.remove_edge(u, v)
```

## 대화 예시와 메모리 구축 과정

### 1. 첫 번째 대화

```
사용자: 우리집 강아지 이름은 미키야
챗봇: 그 부분에 대해 더 알려주시면 좋겠어요.
```

**내부 처리**:
1. **개념 추출**: `["우리집", "강아지", "이름", "미키"]`
2. **메모리 검색**: 과거 기억 없음 → 기본 응답 생성
3. **메모리 저장**: 
   ```python
   memory = MemoryEntry(
       content={"user": "우리집 강아지 이름은 미키야", 
                "assistant": "그 부분에 대해 더 알려주시면 좋겠어요."},
       concepts=["우리집", "강아지", "이름", "미키"],
       importance=0.62,  # 계산된 중요도
       emotional_weight=0.5  # 중립 감정
   )
   ```
4. **연관 네트워크 업데이트**:
   - "미키" ⟷ "강아지" 연결 생성 (강도: 0.8)
   - "강아지" ⟷ "우리집" 연결 생성 (강도: 0.5)
   - "미키" ⟷ "이름" 연결 생성 (강도: 0.6)

### 2. 두 번째 대화

```
사용자: 미키는 7살이야
챗봇: 그 부분에 대해 더 알려주시면 좋겠어요.
```

**내부 처리**:
1. **개념 추출**: `["미키", "7살"]`
2. **메모리 검색**: 
   - 워킹 메모리: 첫 번째 대화 메모리 발견
   - 검색 점수: 1.0 (워킹 메모리이므로 최고 점수)
3. **메모리 저장**:
   ```python
   memory = MemoryEntry(
       content={"user": "미키는 7살이야", 
                "assistant": "그 부분에 대해 더 알려주시면 좋겠어요."},
       concepts=["미키", "7살"],
       importance=0.58,
       emotional_weight=0.5
   )
   ```
4. **연관 네트워크 업데이트**:
   - "미키" 개념 재활성화 → 활성화 카운트 증가
   - "미키" ⟷ "7살" 연결 생성 (강도: 0.7)
   - "미키" ⟷ "강아지" 연결 강화 (강도: 0.85)

### 3. 세 번째 대화

```
사용자: 미키는 케이크를 좋아해
챗봇: 그 부분에 대해 더 알려주시면 좋겠어요. 최근 강아지에 대해 자주 이야기하시네요.
```

**내부 처리**:
1. **개념 추출**: `["미키", "케이크", "좋아해"]`
2. **메모리 검색**:
   - 워킹 메모리: 이전 두 대화 메모리 발견
   - 검색 결과: `[{memory: 첫번째_대화, score: 1.0}, {memory: 두번째_대화, score: 1.0}]`
3. **연관 네트워크 탐색**:
   - "미키" → "강아지" → "우리집" 경로 발견
   - 컨텍스트 인식: "강아지" 개념이 자주 등장함을 감지
4. **메모리 저장**:
   ```python
   memory = MemoryEntry(
       content={"user": "미키는 케이크를 좋아해", 
                "assistant": "그 부분에 대해 더 알려주시면 좋겠어요. 최근 강아지에 대해 자주 이야기하시네요."},
       concepts=["미키", "케이크", "좋아해"],
       importance=0.65,
       emotional_weight=0.67  # "좋아해"로 인한 긍정적 감정
   )
   ```
5. **연관 네트워크 업데이트**:
   - "미키" 개념 재활성화 → 활성화 카운트 증가
   - "미키" ⟷ "케이크" 연결 생성 (강도: 0.7)
   - "케이크" ⟷ "좋아해" 연결 생성 (강도: 0.75, 감정적 연결)

### 4. 회상 질문

```
사용자: 미키는 몇 살이야?
챗봇: 미키는 7살이라고 말씀하셨어요.
```

**내부 처리**:
1. **개념 추출**: `["미키", "몇", "살"]`
2. **메모리 검색**:
   - 워킹 메모리: 이전 대화들 발견
   - 질문 인식: "몇 살" 패턴 감지, 연령 정보 추출
3. **연관 네트워크 탐색**:
   - "미키" → "7살" 직접 연결 발견 (강도: 0.7)
4. **응답 생성**: 
   - 질문 유형 분석: 정보 요청 (연령)
   - 해당 정보 포함된 메모리 선택
   - 정보 추출 후 자연어 응답 생성
5. **메모리 강화**:
   - "미키" ⟷ "7살" 연결 강화 (0.7 → 0.77)
   - 해당 메모리의 접근 카운트 증가

## 시각화 기능

시스템은 현재 형성된 연관 네트워크를 시각화하는 기능을 제공합니다:

```
사용자: 시각화 미키
```

이 명령은 "미키" 개념을 중심으로 한 연관 네트워크를 생성하여 PNG 이미지로 저장합니다. 

### 시각화 프로세스

```python
async def visualize_associations(self, concept: str, save_path: str = None) -> Optional[str]:
    """연관 네트워크 시각화"""
    return visualize_association_network(
        graph=self.association_network.graph,
        center_concept=concept,
        save_path=save_path
    )
```

### 시각화 매개변수

- **노드(개념)**:
  - 크기: 활성화 빈도에 비례 (`activation_count`)
  - 색상: 최근 활성화된 개념일수록 진한 색상 (`last_activated`)
  - 라벨: 개념 텍스트

- **엣지(연결)**:
  - 두께: 연결 강도에 비례 (`weight`)
  - 방향: 연관 방향 표시
  - 라벨: 연결 강도 숫자

- **레이아웃**: 
  - 중심 개념을 가운데 배치
  - 연결된 개념들을 주변에 배치 (스프링 레이아웃)
  - 연결 강도에 따라 거리 조정

## 상세 사용 가이드

### 설치 및 기본 설정

1. **사전 요구사항 설치**
   ```bash
   # 필수 패키지 설치
   pip install redis chromadb sentence-transformers networkx matplotlib \
               psutil aiohttp sqlite3 pybloom-live langdetect
   
   # Redis 서버 설치 (Ubuntu 예시)
   sudo apt-get install redis-server
   sudo systemctl enable redis-server.service
   ```

2. **프로젝트 클론 및 설정**
   ```bash
   git clone https://github.com/username/realtime-associative-chatbot.git
   cd realtime-associative-chatbot
   
   # 설정 파일 생성 (필요시)
   cp config/settings.example.py config/settings.py
   ```

3. **Redis 서버 실행**
   ```bash
   # 서비스로 실행
   sudo systemctl start redis-server
   
   # 또는 직접 실행
   redis-server
   ```

4. **시스템 실행**
   ```bash
   python main.py
   ```

### 기본 명령어

- **일반 대화**: 자연어로 자유롭게 대화
- **시각화 요청**: `시각화 <개념>` 형식으로 입력
- **시스템 종료**: `종료` 입력

### 고급 설정 옵션

`config/settings.py` 파일에서 다양한 설정을 조정할 수 있습니다:

```python
@dataclass
class SystemConfig:
    # Redis 설정
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_ttl: int = 1800  # 워킹 메모리 유지 시간 (초)
    
    # 저장소 설정
    db_path: str = './memory.db'  # SQLite 경로
    chroma_path: str = './chroma_db'  # ChromaDB 경로
    collection_name: str = 'long_term_memories'
    
    # 메모리 관리 설정
    max_active_memory: int = 1000  # 최대 활성 메모리 수
    max_session_duration: int = 3600  # 최대 세션 유지 시간 (초)
    
    # 연관 네트워크 설정
    min_connection_strength: float = 0.1  # 최소 연결 강도
    decay_factor: float = 0.95  # 망각 감소율 (높을수록 느리게 망각)
    
    # 임베딩 모델 설정
    embedding_model: str = 'paraphrase-multilingual-MiniLM-L12-v2'
```

### 로그 모니터링

시스템 로그를 활용하여 내부 작동을 모니터링할 수 있습니다:

```bash
# 로그 실시간 확인
tail -f chatbot.log

# 특정 로그 수준만 확인
grep "INFO" chatbot.log | tail -n 50
```

### 시스템 성능 튜닝

성능 향상을 위한 설정 조정:

1. **메모리 사용량 최적화**
   ```python
   # 워킹 메모리 크기 제한
   max_active_memory: int = 500  # 적은 리소스 환경에서 줄이기
   
   # 캐시 크기 제한
   search_cache = LRUCache(capacity=500)  # 메모리 제한 환경에서 줄이기
   ```

2. **검색 성능 최적화**
   ```python
   # ChromaDB 설정 최적화
   chroma_client = chromadb.PersistentClient(
       path=config['chroma_path'],
       settings=chromadb.Settings(
           chroma_api_impl="rest",
           chroma_server_host="localhost",
           chroma_server_http_port=8000
       )
   )
   ```

3. **Redis 성능 최적화**
   ```bash
   # redis.conf 편집
   maxmemory 100mb
   maxmemory-policy allkeys-lru
   ```

## 성능 최적화 및 확장성

### 성능 최적화

시스템은 다양한 최적화 기법을 통해 실시간 응답성을 유지합니다:

1. **캐싱 전략**
   - **LRU 캐시**: 자주 검색되는 결과 메모리 캐싱
   - **Bloom 필터**: 빠른 멤버십 테스트로 불필요한 검색 방지
   - **임베딩 캐싱**: 반복 쿼리의 임베딩 재사용

   ```python
   # 캐싱 예시
   cache_key = f"search:{':'.join(concepts)}"
   cached_result = self.search_cache.get(cache_key)
   if cached_result:
       return cached_result
   
   # 새 결과 캐싱
   self.search_cache.put(cache_key, result)
   ```

2. **병렬 처리**
   - **비동기 I/O**: 모든 데이터베이스 작업 비동기 처리
   - **병렬 검색**: 여러 저장소를 동시에 검색

   ```python
   # 병렬 검색 예시
   search_tasks = [
       self._search_working_memory(concepts),
       self._search_short_term(concepts),
       self._search_long_term(concepts)
   ]
   results = await asyncio.gather(*search_tasks)
   ```

3. **비동기 저장**
   - 응답 생성 후 백그라운드에서 메모리 저장
   - 사용자 대기 시간 최소화

   ```python
   # 비동기 저장 예시
   asyncio.create_task(self._store_memory(user_id, message, response, concepts))
   ```

4. **선택적 계산**
   - 필요한 경우에만 무거운 연산 수행
   - 간단한 쿼리에는 경량 프로세스 적용

### 확장 가능성

시스템은 추가 확장을 위한 다양한 옵션을 제공합니다:

1. **분산 메모리**
   - Redis Cluster를 활용한 분산 워킹 메모리
   - 수평 확장 가능한 아키텍처

   ```python
   # Redis Cluster 통합 예시
   from rediscluster import RedisCluster
   
   startup_nodes = [
       {"host": "127.0.0.1", "port": "7000"},
       {"host": "127.0.0.1", "port": "7001"}
   ]
   redis_client = RedisCluster(startup_nodes=startup_nodes, decode_responses=True)
   ```

2. **고급 NLP 통합**
   - 외부 NLP 서비스를 활용한 개념 추출 개선
   - 감정 분석 강화

   ```python
   # 외부 NLP 통합 예시
   async def _extract_concepts_with_nlp(self, text: str) -> List[str]:
       async with aiohttp.ClientSession() as session:
           async with session.post('http://nlp-service/extract', json={'text': text}) as response:
               result = await response.json()
               return result['concepts']
   ```

3. **멀티모달 지원**
   - 이미지, 음성 등 다양한 입력 처리 
   - 모달리티 간 연관 관계 구축

4. **다중 사용자 확장**
   - 사용자별 연관 네트워크 관리
   - 공유 지식과 개인 지식 분리

## 문제 해결

### 일반적인 문제

1. **Redis 연결 오류**
   - 오류: `Error 111 connecting to localhost:6379. Connection refused.`
   - 해결: Redis 서버가 실행 중인지 확인 (`redis-server` 실행)
   - 진단: `redis-cli ping`으로 연결 테스트

2. **SQLite 테이블 오류**
   - 오류: `no such table: concepts`
   - 해결: 데이터베이스 파일 삭제 후 재시작 (`rm memory.db`)
   - 진단: `sqlite3 memory.db .tables`로 테이블 구조 확인

3. **ChromaDB 오류**
   - 오류: `Failed to load chromadb indices`
   - 해결: ChromaDB 디렉토리 재생성 (`rm -rf ./chroma_db`)
   - 진단: ChromaDB 버전 확인 (`pip show chromadb`)

4. **메모리 사용량 과다**
   - 증상: 시스템 메모리 사용률 증가, 응답 지연
   - 해결: `max_active_memory` 설정 감소, 정기적인 `gc.collect()` 호출
   - 진단: `print(psutil.Process().memory_info().rss / 1024 / 1024)`로 메모리 사용량 확인

### 상세 문제 해결 가이드

#### 데이터베이스 일관성 검사

워킹 메모리와 단기 메모리, 장기 메모리 간의 일관성이 손상된 경우:

```python
# memory_manager.py에 디버깅 메서드 추가
async def verify_data_consistency(self):
    """저장소 간 데이터 일관성 검증"""
    # 워킹 메모리 (Redis) 검사
    redis_keys = self.redis_client.keys("working:*")
    print(f"Working memory: {len(redis_keys)} keys")
    
    # 단기 메모리 (SQLite) 검사
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM memories")
    sqlite_count = cursor.fetchone()[0]
    print(f"Short-term memory: {sqlite_count} entries")
    
    # 장기 메모리 (ChromaDB) 검사
    long_term_count = self.chroma_collection.count()
    print(f"Long-term memory: {long_term_count} entries")
    
    # 정합성 검사
    cursor.execute("SELECT id FROM memories WHERE tier = 'long'")
    long_term_ids_in_sqlite = [row[0] for row in cursor.fetchall()]
    
    chroma_ids = self.chroma_collection.get()["ids"]
    
    in_sqlite_not_chroma = set(long_term_ids_in_sqlite) - set(chroma_ids)
    in_chroma_not_sqlite = set(chroma_ids) - set(long_term_ids_in_sqlite)
    
    print(f"Inconsistencies: {len(in_sqlite_not_chroma)} in SQLite but not in ChromaDB")
    print(f"Inconsistencies: {len(in_chroma_not_sqlite)} in ChromaDB but not in SQLite")
    
    conn.close()
```

#### 연관 네트워크 디버깅

연관 네트워크가 제대로 형성되지 않는 경우:

```python
# association_network.py에 디버깅 메서드 추가
def debug_network(self):
    """네트워크 상태 디버깅"""
    print(f"Total concepts: {self.graph.number_of_nodes()}")
    print(f"Total connections: {self.graph.number_of_edges()}")
    
    if self.graph.number_of_nodes() > 0:
        # 가장 많이 활성화된 개념 출력
        activation_counts = []
        for node, data in self.graph.nodes(data=True):
            activation_counts.append((node, data.get('activation_count', 0)))
        
        top_concepts = sorted(activation_counts, key=lambda x: x[1], reverse=True)[:10]
        print("Top activated concepts:")
        for concept, count in top_concepts:
            print(f"  - {concept}: {count} activations")
        
        # 가장 강한 연결 출력
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append((u, v, data.get('weight', 0)))
        
        strongest_edges = sorted(edges, key=lambda x: x[2], reverse=True)[:10]
        print("Strongest connections:")
        for u, v, weight in strongest_edges:
            print(f"  - {u} → {v}: {weight:.2f}")
    
    # 고립된 개념 검사
    isolated = list(nx.isolates(self.graph))
    print(f"Isolated concepts: {len(isolated)}")
    if isolated:
        print(f"Examples: {isolated[:5]}")
```

#### 성능 병목 진단

시스템 응답이 느린 경우 병목 지점 파악:

```python
# 메서드 실행 시간 측정 데코레이터
def timing_decorator(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# 주요 메서드에 적용 (예시)
@timing_decorator
async def search_memories(self, concepts: List[str], query_text: Optional[str] = None, top_k: int = 10):
    # 기존 메서드 내용...
```

## 개발 가이드

### 컴포넌트 확장 방법

#### 1. 새로운 메모리 계층 추가

새로운 저장소를 시스템에 통합하려면 다음과 같은 인터페이스를 구현해야 합니다:

```python
# storage/custom_storage.py
class CustomStorage:
    """새로운 메모리 계층 저장소"""
    
    def __init__(self, config: Dict[str, Any]):
        # 저장소 초기화
        pass
    
    async def save(self, memory: MemoryEntry) -> None:
        """메모리 저장"""
        # 구현...
        pass
    
    async def find_by_concepts(self, concepts: List[str]) -> List[MemoryEntry]:
        """개념으로 메모리 검색"""
        # 구현...
        pass
    
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """ID로 메모리 조회"""
        # 구현...
        pass
    
    async def delete(self, memory_id: str) -> None:
        """메모리 삭제"""
        # 구현...
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """저장소 통계"""
        # 구현...
        pass
```

그런 다음 `MemoryManager` 클래스에 새 저장소 통합:

```python
# core/memory_manager.py
def __init__(self, config: Dict[str, Any]):
    # 기존 저장소 초기화...
    
    # 새 저장소 추가
    self.custom_storage = CustomStorage(config['custom_storage'])
    
    # 새 검색 메서드 추가
    async def _search_custom_storage(self, concepts: List[str]) -> List[Any]:
        try:
            return await self.custom_storage.find_by_concepts(concepts)
        except Exception as e:
            print(f"Custom storage search error: {e}")
            return []
    
    # search_memories 메서드 수정
    search_tasks = [
        self._search_working_memory(concepts),
        self._search_short_term(concepts),
        self._search_long_term(concepts),
        self._search_custom_storage(concepts)  # 새 저장소 검색 추가
    ]
```

#### 2. 맞춤형 연관 유형 정의

특정 도메인에 맞는 새로운 연관 유형을 추가할 수 있습니다:

```python
# models/enums.py의 ConnectionType 열거형에 추가
class ConnectionType(Enum):
    # 기존 유형들...
    CUSTOM = "custom"      # 새 연관 유형
    HIERARCHICAL = "hier"  # 계층 관계 유형
    CONTRAST = "contrast"  # 대조 관계 유형
```

기본 연결 강도 설정:

```python
# core/association_network.py에 추가
self.connection_strengths = {
    # 기존 강도 설정...
    ConnectionType.CUSTOM: 0.6,       # 새 유형 기본 강도
    ConnectionType.HIERARCHICAL: 0.85, # 계층 관계는 강한 연결
    ConnectionType.CONTRAST: 0.4       # 대조 관계는 약한 연결
}
```

#### 3. 새로운 개념 추출 알고리즘 통합

더 정교한 개념 추출 방법을 적용할 수 있습니다:

```python
# 외부 NLP 라이브러리 활용 예시
from transformers import Pipeline, pipeline

class EnhancedConceptExtractor:
    """고급 개념 추출기"""
    
    def __init__(self):
        # 토큰 분류기 로드
        self.nlp = pipeline("token-classification", 
                           model="distilbert-base-cased", 
                           aggregation_strategy="simple")
    
    def extract_concepts(self, text: str) -> List[str]:
        """텍스트에서 핵심 개념 추출"""
        # 개체명 인식
        entities = self.nlp(text)
        
        # 명사구 추출
        noun_phrases = []
        
        # 결과 통합
        concepts = []
        for entity in entities:
            if entity['score'] > 0.7:  # 확률 기반 필터링
                concepts.append(entity['word'])
        
        concepts.extend(noun_phrases)
        return list(set(concepts))  # 중복 제거
```

이후 `chatbot.py`에 통합:

```python
# chatbot.py 수정
from utils.concept_extractor import EnhancedConceptExtractor

def __init__(self, config=None):
    # 기존 초기화...
    self.concept_extractor = EnhancedConceptExtractor()

def _extract_concepts(self, text: str) -> List[str]:
    """개선된 개념 추출기 사용"""
    return self.concept_extractor.extract_concepts(text)
```


## 시스템 한계 및 향후 개선 방향

### 현재 한계

1. **단순 키워드 기반 개념 추출**
   - 문맥 이해 없는 단어 수준 추출
   - 동음이의어/이음동의어 구분 미흡
   - 구문/문장 수준 개념 부재

2. **제한된 시간 인식**
   - 시간적 관계 파악 능력 기본적 수준
   - 시간 표현 정규화 부족
   - 시간에 따른 개념 변화 추적 기능 미흡

3. **대화 깊이 제한**
   - 복잡한 추론 능력 부족
   - 다단계 회상 메커니즘 미흡
   - 장기 컨텍스트 이해 제한적

4. **고정된 가중치 시스템**
   - 중요도 및 관계 강도 계산이 사전 정의된 규칙에 의존
   - 사용자 및 대화별 적응형 가중치 부재
   - 학습을 통한 가중치 최적화 불가

### 향후 개선 방향

1. **문맥적 임베딩**
   - 단어 수준이 아닌 문맥 수준의 임베딩으로 의미적 연결 개선
   - 트랜스포머 모델 기반 문맥 이해 적용
   - 개념 간 의미적 유사성 계산 개선

   ```python
   # 개선된 문맥 임베딩 예시
   from sentence_transformers import SentenceTransformer

   class ContextualEmbedding:
       def __init__(self):
           self.model = SentenceTransformer('all-mpnet-base-v2')
       
       def encode_context(self, text: str) -> np.ndarray:
           # 문장 수준의 임베딩
           return self.model.encode(text)
       
       def compute_similarity(self, text1: str, text2: str) -> float:
           # 코사인 유사도 계산
           emb1 = self.encode_context(text1)
           emb2 = self.encode_context(text2)
           return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
   ```

2. **계층적 개념 구조**
   - 상위-하위 개념 관계를 인식하는 구조화된 지식 표현
   - 온톨로지 통합으로 개념 간 관계 풍부화
   - 메타 개념 관리로 추상화 수준 조정

   ```python
   # 계층적 개념 구조 예시
   class HierarchicalNetwork(AssociationNetwork):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.parent_relations = {}  # 상위 개념 관계
           self.child_relations = {}   # 하위 개념 관계
       
       def add_hierarchical_relation(self, parent: str, child: str):
           """계층적 관계 추가"""
           # 상위-하위 관계 등록
           if parent not in self.parent_relations:
               self.parent_relations[parent] = set()
           self.parent_relations[parent].add(child)
           
           if child not in self.child_relations:
               self.child_relations[child] = set()
           self.child_relations[child].add(parent)
           
           # 강한 계층적 연결 생성
           self.connect_concepts(parent, child, ConnectionType.HIERARCHICAL)
   ```

3. **학습 기반 가중치 최적화**
   - 사용자 피드백을 기반으로 가중치 자동 조정
   - 회상 성공률에 따른 강화학습 적용
   - 사용자별 맞춤형 가중치 프로필

   ```python
   # 강화학습 기반 가중치 조정 예시
   class AdaptiveWeightManager:
       def __init__(self, learning_rate: float = 0.01):
           self.learning_rate = learning_rate
           self.weight_history = {}
       
       def update_weights(self, weights: Dict[str, float], feedback: float) -> Dict[str, float]:
           """피드백에 따른 가중치 업데이트"""
           # feedback: -1.0 ~ 1.0 범위의 피드백 점수
           updated = {}
           for key, value in weights.items():
               # 가중치 조정 (피드백 방향으로)
               delta = self.learning_rate * feedback
               updated[key] = max(0.0, min(1.0, value + delta))
               
               # 이력 기록
               if key not in self.weight_history:
                   self.weight_history[key] = []
               self.weight_history[key].append(updated[key])
           
           return updated
   ```

4. **멀티모달 통합**
   - 텍스트 외에도 이미지, 음성 등 다양한 입력 처리
   - 모달리티 간 연관 관계 구축
   - 멀티모달 컨텍스트 이해

   ```python
   # 멀티모달 통합 예시
   class MultimodalMemory(MemoryEntry):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.modality_data = {}  # 모달리티별 데이터
       
       def add_modality(self, modality_type: str, data: Any):
           """모달리티 데이터 추가"""
           self.modality_data[modality_type] = data
       
       def get_modality(self, modality_type: str) -> Optional[Any]:
           """모달리티 데이터 조회"""
           return self.modality_data.get(modality_type)
   ```

5. **시간적 인식 개선**
   - 시간 표현 정규화 및 추론
   - 시간에 따른 개념 진화 추적
   - 시간적 연관성 강화

   ```python
   # 시간 인식 개선 예시
   from dateparser import parse
   
   class TemporalContext:
       def __init__(self):
           self.time_references = {}  # 개념별 시간 참조
       
       def extract_time(self, text: str) -> Optional[datetime]:
           """텍스트에서 시간 표현 추출"""
           return parse(text, settings={'PREFER_DATES_FROM': 'future'})
       
       def associate_concept_with_time(self, concept: str, time_ref: datetime):
           """개념과 시간 연관 지정"""
           self.time_references[concept] = time_ref
       
       def get_concepts_in_timeframe(self, start: datetime, end: datetime) -> List[str]:
           """특정 시간대의 개념 조회"""
           return [c for c, t in self.time_references.items() 
                  if start <= t <= end]
   ```

6. **추론 엔진 통합**
   - 연관 네트워크 기반 추론 능력 강화
   - 인과 관계 및 논리적 추론 지원
   - 다단계 참조 따라가기 개선

   ```python
   # 추론 엔진 예시
   class ReasoningEngine:
       def __init__(self, association_network):
           self.network = association_network
           self.inference_rules = []
       
       def add_rule(self, premise: List[str], conclusion: str, confidence: float):
           """추론 규칙 추가"""
           self.inference_rules.append({
               'premise': premise,
               'conclusion': conclusion,
               'confidence': confidence
           })
       
       def infer(self, concepts: List[str]) -> List[Dict]:
           """개념 집합으로부터 추론"""
           inferences = []
           
           # 직접 연결된 개념 확인
           related = self.network.find_associations(concepts)
           
           # 규칙 기반 추론
           for rule in self.inference_rules:
               # 전제 조건 충족 여부 확인
               if all(p in concepts for p in rule['premise']):
                   inferences.append({
                       'conclusion': rule['conclusion'],
                       'confidence': rule['confidence'],
                       'source': 'rule'
                   })
           
           return inferences
   ```

## 참고 자료

이 프로젝트는 다음 연구와 기술에 기초합니다:

- 연관 네트워크 알고리즘: "Spreading Activation Networks in Memory" (Collins & Loftus, 1975)
- 메모리 계층 구조: "Multi-Store Model of Memory" (Atkinson & Shiffrin, 1968)
- 벡터 임베딩: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (Reimers & Gurevych, 2019)
- 감정 가중치: "Emotional Tagging of Memories: The Effects of Physiological Arousal on Memory" (Cahill & McGaugh, 1998)


## 기여 방법

이슈와 PR은 언제나 환영합니다. 대규모 변경 사항은 먼저 이슈를 통해 논의해주세요.

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add some amazing feature'`)
4. 브랜치 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 제출