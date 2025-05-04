"""
인간의 연관 기억을 모방한 키워드 연결 그래프 시스템
"""

import json
import networkx as nx
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Any, Set, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
import math

class AssociativeMemoryNetwork:
    """연관 기억 네트워크 - 키워드 간 자동 연결 및 강도 관리"""
    
    def __init__(self, decay_factor=0.95, min_connection_strength=0.1):
        """
        초기화
        
        Args:
            decay_factor: 시간에 따른 연결 강도 감소율
            min_connection_strength: 최소 연결 강도 (이하로는 연결 제거)
        """
        # NetworkX 그래프 사용 (방향성 있는 연관 그래프)
        self.graph = nx.DiGraph()
        
        # 연결 강화/약화 정책
        self.decay_factor = decay_factor
        self.min_connection_strength = min_connection_strength
        
        # 연결 경로 캐시 (성능 최적화)
        self._path_cache = {}
        self._cache_timeout = timedelta(hours=1)
        self._last_cache_update = datetime.now()
        
        # 최근 활성화 기록 (동시 활성화 추적)
        self.recent_activations = deque(maxlen=10)
        self.activation_window = timedelta(seconds=30)
        
        # 연결 타입 정의
        self.connection_types = {
            'semantic': 0.8,    # 의미적 유사성 (강한 연결)
            'procedural': 0.6,  # 절차적 관계 (순서)
            'temporal': 0.4,    # 시간적 공동 발생
            'emotional': 0.7,   # 감정적 연관
            'spatial': 0.5,     # 공간적 관계
            'causal': 0.9       # 인과 관계
        }
    
    def add_concept(self, concept: str, metadata: Dict[str, Any] = None) -> None:
        """개념(키워드)을 그래프에 추가"""
        if not self.graph.has_node(concept):
            self.graph.add_node(concept, 
                              activation_count=0,
                              last_activated=None,
                              metadata=metadata or {},
                              creation_time=datetime.now())
            print(f"[연관망] 새 개념 추가: '{concept}'")
    
    def activate_concept(self, concept: str, related_concepts: List[str] = None) -> None:
        """
        개념 활성화 및 연관 관계 학습
        
        Args:
            concept: 활성화할 개념
            related_concepts: 함께 활성화된 개념들 (문맥)
        """
        current_time = datetime.now()
        
        # 개념 활성화
        if not self.graph.has_node(concept):
            self.add_concept(concept)
        
        # 활성화 통계 업데이트
        self.graph.nodes[concept]['activation_count'] += 1
        self.graph.nodes[concept]['last_activated'] = current_time
        
        # 최근 활성화 기록에 추가
        self.recent_activations.append({
            'concept': concept,
            'time': current_time,
            'context': related_concepts or []
        })
        
        # 동시 활성화 분석 및 연결 강화
        self._analyze_co_activation(concept)
        
        # 순차 활성화 분석 (절차적 기억)
        self._analyze_sequential_activation(concept)
        
        # 확산 활성화 시뮬레이션
        related = self.get_related_concepts(concept, depth=1)
        for rel_concept, strength in related.items():
            # 간접 활성화로 연결 약간 강화
            self._strengthen_connection(concept, rel_concept, 0.1)
    
    def add_connection(self, from_concept: str, to_concept: str, 
                      connection_type: str = 'semantic', 
                      custom_strength: float = None) -> None:
        """두 개념 간 직접 연결 추가"""
        # 개념이 없으면 추가
        self.add_concept(from_concept)
        self.add_concept(to_concept)
        
        # 연결 강도 결정
        base_strength = self.connection_types.get(connection_type, 0.5)
        strength = custom_strength if custom_strength is not None else base_strength
        
        # 연결 추가 또는 강화
        if self.graph.has_edge(from_concept, to_concept):
            # 기존 연결 강화
            current_strength = self.graph[from_concept][to_concept]['weight']
            new_strength = min(1.0, current_strength + strength * 0.2)
            self.graph[from_concept][to_concept]['weight'] = new_strength
        else:
            # 새 연결 생성
            self.graph.add_edge(from_concept, to_concept, 
                              weight=strength,
                              type=connection_type,
                              creation_time=datetime.now(),
                              strengthening_count=0)
        
        # 양방향 관계 자동 설정 (semantic 연결의 경우)
        if connection_type == 'semantic' and not self.graph.has_edge(to_concept, from_concept):
            self.graph.add_edge(to_concept, from_concept,
                              weight=strength * 0.8,  # 약간 약한 역방향 연결
                              type=connection_type,
                              creation_time=datetime.now(),
                              strengthening_count=0)
    
    def _analyze_co_activation(self, concept: str) -> None:
        """동시 활성화 패턴 분석 및 연결 강화"""
        current_time = datetime.now()
        
        # 최근 활성화된 개념들 중 시간 창 내의 개념들 찾기
        co_activated = []
        for activation in reversed(self.recent_activations):
            if current_time - activation['time'] <= self.activation_window:
                if activation['concept'] != concept:
                    co_activated.append(activation['concept'])
                    
                    # 문맥 기반 연결도 추가
                    if activation['context']:
                        for ctx_concept in activation['context']:
                            if ctx_concept != concept:
                                co_activated.append(ctx_concept)  # 이 부분이 빠졌었어!
            else:
                break
    
        # 동시 활성화된 개념들과의 연결 강화
        for co_concept in co_activated:
            self._strengthen_connection(concept, co_concept, 0.3)
            # 인접 개념과의 통계 업데이트
            self._update_co_occurrence_stats(concept, co_concept)
        
    
    
    def _analyze_sequential_activation(self, concept: str) -> None:
        """순차적 활성화 패턴 분석 (절차적 기억)"""
        if len(self.recent_activations) < 2:
            return
        
        current_idx = len(self.recent_activations) - 1
        previous_activation = self.recent_activations[current_idx - 1]
        
        # 순차적 관계 확인
        time_diff = (self.recent_activations[current_idx]['time'] - 
                    previous_activation['time'])
        
        # 짧은 시간 내 순차 활성화면 절차적 연결 강화
        if time_diff <= timedelta(seconds=5):
            prev_concept = previous_activation['concept']
            if prev_concept != concept:
                self._strengthen_connection(prev_concept, concept, 0.4, 'procedural')
    
    def _strengthen_connection(self, from_concept: str, to_concept: str, 
                             delta: float, connection_type: str = 'semantic') -> None:
        """연결 강도 증가"""
        if self.graph.has_edge(from_concept, to_concept):
            edge_data = self.graph[from_concept][to_concept]
            edge_data['weight'] = min(1.0, edge_data['weight'] + delta)
            edge_data['strengthening_count'] += 1
            edge_data['last_strengthened'] = datetime.now()
        else:
            # 연결이 없으면 새로 생성
            self.add_connection(from_concept, to_concept, connection_type, delta)
    
    def _update_co_occurrence_stats(self, concept1: str, concept2: str) -> None:
        """동시 발생 통계 업데이트"""
        # 각 노드의 동시 발생 통계 업데이트
        for node in [concept1, concept2]:
            if 'co_occurrence' not in self.graph.nodes[node]:
                self.graph.nodes[node]['co_occurrence'] = defaultdict(int)
            
            other = concept2 if node == concept1 else concept1
            self.graph.nodes[node]['co_occurrence'][other] += 1
    
    def get_related_concepts(self, concept: str, depth: int = 2, 
                           min_strength: float = None) -> Dict[str, float]:
        """
        관련 개념 검색 (확산 활성화 알고리즘)
        
        Args:
            concept: 시작 개념
            depth: 탐색 깊이
            min_strength: 최소 연결 강도
            
        Returns:
            Dict[concept, strength]: 관련 개념과 그 연결 강도
        """
        if not self.graph.has_node(concept):
            return {}
        
        min_strength = min_strength or self.min_connection_strength
        related = {}
        
        # BFS로 관련 개념 탐색
        queue = deque([(concept, 0, 1.0)])
        visited = {concept}
        
        while queue:
            current, current_depth, current_strength = queue.popleft()
            
            if current_depth < depth:
                # 이웃 노드 탐색
                for neighbor in self.graph.neighbors(current):
                    edge_strength = self.graph[current][neighbor]['weight']
                    accumulated_strength = current_strength * edge_strength
                    
                    if accumulated_strength >= min_strength:
                        # 더 강한 경로가 있으면 업데이트
                        if neighbor not in related or related[neighbor] < accumulated_strength:
                            related[neighbor] = accumulated_strength
                            
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append((neighbor, current_depth + 1, accumulated_strength))
        
        # 시작 개념 제외
        related.pop(concept, None)
        
        # 강도 순으로 정렬
        return dict(sorted(related.items(), key=lambda x: x[1], reverse=True))
    
    def find_association_path(self, from_concept: str, to_concept: str, 
                             max_depth: int = 4) -> List[Tuple[str, float]]:
        """
        두 개념 간 연결 경로 찾기 (연관 사슬)
        
        Returns:
            List[(concept, strength)]: 개념과 다음 단계로의 연결 강도
        """
        if not (self.graph.has_node(from_concept) and self.graph.has_node(to_concept)):
            return []
        
        try:
            # 캐시 확인
            cache_key = f"{from_concept}->{to_concept}"
            if cache_key in self._path_cache:
                cached_path, cache_time = self._path_cache[cache_key]
                if datetime.now() - cache_time < self._cache_timeout:
                    return cached_path
            
            # A* 알고리즘으로 최적 경로 찾기
            path = nx.shortest_path(self.graph, from_concept, to_concept, 
                                  weight=lambda u, v, d: 1 - d['weight'])
            
            # 경로 상세 정보 구성
            detailed_path = []
            for i in range(len(path) - 1):
                current = path[i]
                next_node = path[i + 1]
                edge_weight = self.graph[current][next_node]['weight']
                detailed_path.append((current, edge_weight))
            
            # 마지막 노드 추가
            detailed_path.append((path[-1], 1.0))
            
            # 캐시에 저장
            self._path_cache[cache_key] = (detailed_path, datetime.now())
            
            return detailed_path
            
        except nx.NetworkXNoPath:
            return []
    
    def apply_decay(self, time_passed: timedelta = None) -> None:
        """시간 경과에 따른 연결 강도 감소"""
        if time_passed is None:
            time_passed = timedelta(days=1)
        
        decay_amount = 1 - (self.decay_factor ** (time_passed.days))
        
        edges_to_remove = []
        
        for u, v, data in self.graph.edges(data=True):
            # 연결 강도 감소
            new_weight = data['weight'] * (1 - decay_amount)
            
            if new_weight < self.min_connection_strength:
                edges_to_remove.append((u, v))
            else:
                data['weight'] = new_weight
        
        # 약한 연결 제거
        for u, v in edges_to_remove:
            self.graph.remove_edge(u, v)
            print(f"[망각] 약한 연결 제거: {u} -> {v}")
    
    def visualize_network(self, central_concept: str = None, depth: int = 2, 
                         save_path: str = None, show: bool = True) -> None:
        """네트워크 시각화"""
        if central_concept:
            # 특정 개념 중심으로 서브그래프 생성
            related = self.get_related_concepts(central_concept, depth)
            nodes = {central_concept} | set(related.keys())
            subgraph = self.graph.subgraph(nodes)
        else:
            subgraph = self.graph
        
        plt.figure(figsize=(15, 10))
        
        # 노드 크기: 활성화 횟수 기반
        node_sizes = []
        for node in subgraph.nodes():
            activation_count = self.graph.nodes[node].get('activation_count', 0)
            node_sizes.append(300 + activation_count * 50)
        
        # 노드 색상: 마지막 활성화 시간 기반
        node_colors = []
        current_time = datetime.now()
        for node in subgraph.nodes():
            last_activated = self.graph.nodes[node].get('last_activated')
            if last_activated:
                time_diff = (current_time - last_activated).total_seconds() / 3600  # hours
                # 최근일수록 짙은 색
                color_value = max(0, 1 - min(time_diff / 24, 1))
                node_colors.append(color_value)
            else:
                node_colors.append(0)
        
        # 엣지 두께: 연결 강도 기반
        edge_widths = []
        for u, v in subgraph.edges():
            weight = self.graph[u][v]['weight']
            edge_widths.append(weight * 5)
        
        # 레이아웃 설정
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # 그래프 그리기
        nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                              node_color=node_colors, cmap=plt.cm.Blues)
        nx.draw_networkx_edges(subgraph, pos, width=edge_widths, 
                              alpha=0.6, edge_color='gray')
        nx.draw_networkx_labels(subgraph, pos, font_size=10, font_weight='bold')
        
        # 엣지 레이블 (연결 강도)
        edge_labels = {}
        for u, v in subgraph.edges():
            weight = self.graph[u][v]['weight']
            edge_labels[(u, v)] = f"{weight:.2f}"
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels, font_size=8)
        
        plt.title(f"연관 기억 네트워크{f' - {central_concept} 중심' if central_concept else ''}")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_network_stats(self) -> Dict[str, Any]:
        """네트워크 통계 정보"""
        return {
            'total_concepts': self.graph.number_of_nodes(),
            'total_connections': self.graph.number_of_edges(),
            'average_connections': self.graph.number_of_edges() / max(1, self.graph.number_of_nodes()),
            'connected_components': nx.number_weakly_connected_components(self.graph),
            'most_activated': self._get_most_activated_concepts(5),
            'strongest_connections': self._get_strongest_connections(5)
        }
    
    def _get_most_activated_concepts(self, top_k: int) -> List[Tuple[str, int]]:
        """가장 많이 활성화된 개념들"""
        concepts = []
        for node, data in self.graph.nodes(data=True):
            activation_count = data.get('activation_count', 0)
            concepts.append((node, activation_count))
        
        return sorted(concepts, key=lambda x: x[1], reverse=True)[:top_k]
    
    def _get_strongest_connections(self, top_k: int) -> List[Tuple[str, str, float]]:
        """가장 강한 연결들"""
        connections = []
        for u, v, data in self.graph.edges(data=True):
            connections.append((u, v, data['weight']))
        
        return sorted(connections, key=lambda x: x[2], reverse=True)[:top_k]

# ----------------- 통합된 향상된 메모리 시스템 -----------------

class EnhancedAssociativeMemoryStore:
    """연관 기억을 포함한 향상된 메모리 저장소"""
    
    def __init__(self, collection_name="associative_memory", persist_directory="./chroma_db"):
        """초기화"""
        # 기존 ChromaDB 메모리 저장소
        self.memory_store = EnhancedMemoryStore(collection_name, persist_directory)
        
        # 연관 기억 네트워크 추가
        self.association_network = AssociativeMemoryNetwork()
        
        # 개념 클러스터링을 위한 임시 저장소
        self.concept_clusters = defaultdict(set)
        
        # 시간대별 개념 활성화 히스토리
        self.temporal_activation = defaultdict(list)
    
    def add_memory_with_associations(self, memory_data: Dict[str, Any], 
                                   related_concepts: List[str] = None) -> str:
        """메모리 추가와 동시에 연관 관계 구축"""
        # 기본 메모리 저장
        memory_id = self.memory_store.add_memory(memory_data)
        
        # 트리거 개념들로부터 연관 관계 학습
        triggers = memory_data.get('triggers', [])
        
        # 메모리 텍스트에서 추가 개념 추출
        extracted_concepts = self._extract_additional_concepts(memory_data)
        all_concepts = list(set(triggers + extracted_concepts + (related_concepts or [])))
        
        # 개념 간 연관 관계 구축
        self._build_associations(all_concepts, memory_data)
        
        # 시간대별 활성화 기록
        timestamp = datetime.now()
        for concept in all_concepts:
            self.temporal_activation[timestamp].append(concept)
        
        return memory_id
    
    def _extract_additional_concepts(self, memory_data: Dict[str, Any]) -> List[str]:
        """메모리 데이터에서 추가 개념 추출"""
        text = memory_data.get('user', '') + ' ' + memory_data.get('assistant', '')
        
        # 간단한 개념 추출 (실제로는 더 정교한 NLP 처리 필요)
        concepts = []
        
        # 명사 추출 (임시 구현)
        words = text.split()
        for word in words:
            if len(word) > 2 and word.isalpha():
                concepts.append(word)
        
        return concepts
    
    def _build_associations(self, concepts: List[str], memory_data: Dict[str, Any]) -> None:
        """개념 간 연관 관계 구축"""
        # 감정적 연결
        emotional_weight = memory_data.get('emotional_weight', 0.0)
        if emotional_weight > 0.2:
            connection_strength = min(emotional_weight * 1.5, 1.0)
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    self.association_network.add_connection(
                        concepts[i], concepts[j], 
                        'emotional', connection_strength
                    )
        
        # 의미적 연결 (카테고리 기반)
        semantic_groups = self._group_by_semantics(concepts)
        for group in semantic_groups:
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        self.association_network.add_connection(
                            group[i], group[j], 
                            'semantic', 0.8
                        )
        
        # 시간적 연결 (공동 발생)
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                self.association_network.add_connection(
                    concepts[i], concepts[j], 
                    'temporal', 0.4
                )
    
    def _group_by_semantics(self, concepts: List[str]) -> List[List[str]]:
        """개념들을 의미별로 그룹화"""
        # 실제로는 워드 임베딩이나 WordNet 등을 사용해야 함
        # 여기서는 간단한 규칙 기반 분류
        groups = defaultdict(list)
        
        for concept in concepts:
            # 간단한 카테고리 분류
            if any(word in concept for word in ['강아지', '개', '고양이', '동물']):
                groups['animal'].append(concept)
            elif any(word in concept for word in ['생일', '축하', '파티']):
                groups['celebration'].append(concept)
            elif any(word in concept for word in ['음식', '먹', '요리']):
                groups['food'].append(concept)
            else:
                groups['general'].append(concept)
        
        return list(groups.values())
    
    def recall_associations(self, query_concept: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        개념과 관련된 기억 연쇄 회상
        
        Returns:
            Dict containing:
            - direct_memories: 직접 관련된 메모리들
            - associated_concepts: 연관된 개념들
            - association_paths: 연관 경로들
            - concept_clusters: 관련 개념 클러스터
        """
        # 직접 관련된 메모리 검색
        direct_memories = self.memory_store.search_by_triggers(query_concept, [query_concept])
        
        # 연관 개념 확장
        related_concepts = self.association_network.get_related_concepts(query_concept, depth=2)
        
        # 연관 개념을 통한 간접 메모리 검색
        indirect_memories = []
        for concept, strength in related_concepts.items():
            if strength > 0.3:  # 연결 강도 임계값
                memories = self.memory_store.search_by_triggers(concept, [concept], top_k=2)
                for mem in memories:
                    mem['association_strength'] = strength
                    mem['associated_concept'] = concept
                    indirect_memories.append(mem)
        
        # 연관 경로 찾기
        association_paths = []
        for mem in direct_memories[:3]:  # 상위 3개 메모리에 대해
            if mem['memory']['triggers']:
                for trigger in mem['memory']['triggers'][:2]:
                    path = self.association_network.find_association_path(query_concept, trigger)
                    if path:
                        association_paths.append((trigger, path))
        
        # 개념 클러스터 형성
        all_concepts = [query_concept] + list(related_concepts.keys())
        concept_clusters = self._create_concept_clusters(all_concepts)
        
        return {
            'direct_memories': direct_memories,
            'indirect_memories': indirect_memories,
            'related_concepts': related_concepts,
            'association_paths': association_paths,
            'concept_clusters': concept_clusters
        }
    
    def _create_concept_clusters(self, concepts: List[str]) -> Dict[str, List[str]]:
        """연관된 개념들을 클러스터로 그룹화"""
        clusters = defaultdict(list)
        
        # 커뮤니티 감지 알고리즘으로 클러스터 생성
        subgraph = self.association_network.graph.subgraph(concepts)
        if subgraph.number_of_nodes() > 0:
            communities = list(nx.community.greedy_modularity_communities(subgraph.to_undirected()))
            
            for i, community in enumerate(communities):
                cluster_name = f"cluster_{i}"
                clusters[cluster_name] = list(community)
        
        return dict(clusters)
    
    def strengthen_association_by_recall(self, concept: str, recalled_memory: Dict[str, Any]) -> None:
        """회상을 통한 연관 강화"""
        # 회상된 메모리의 트리거들과의 연관 강화
        if 'triggers' in recalled_memory:
            for trigger in recalled_memory['triggers']:
                if trigger != concept:
                    self.association_network._strengthen_connection(concept, trigger, 0.2)
    
    def visualize_association_web(self, central_concept: str, save_path: str = None) -> None:
        """연관 웹 시각화"""
        self.association_network.visualize_network(central_concept, depth=2, 
                                                  save_path=save_path, show=True)
    
    def get_association_report(self) -> Dict[str, Any]:
        """연관 기억 시스템 리포트"""
        network_stats = self.association_network.get_network_stats()
        
        # 시간대별 활성화 분석
        activation_patterns = self._analyze_temporal_activation()
        
        return {
            'network_statistics': network_stats,
            'activation_patterns': activation_patterns,
            'cluster_analysis': self._analyze_concept_clusters()
        }
    
    def _analyze_temporal_activation(self) -> Dict[str, Any]:
        """시간대별 개념 활성화 패턴 분석"""
        patterns = defaultdict(int)
        
        # 시간대별 활성화 빈도 분석
        for timestamp, concepts in self.temporal_activation.items():
            hour = timestamp.hour
            patterns[f"hour_{hour}"] += len(concepts)
        
        return dict(patterns)
    
    def _analyze_concept_clusters(self) -> Dict[str, Any]:
        """개념 클러스터 분석"""
        all_nodes = list(self.association_network.graph.nodes())
        if not all_nodes:
            return {}
        
        clusters = self._create_concept_clusters(all_nodes)
        
        # 클러스터별 통계
        cluster_analysis = {}
        for cluster_name, members in clusters.items():
            if members:
                cluster_analysis[cluster_name] = {
                    'size': len(members),
                    'average_activation': sum(
                        self.association_network.graph.nodes[m].get('activation_count', 0) 
                        for m in members) / len(members),
                    'members': members
                }
        
        return cluster_analysis

# ----------------- 사용 예시 -----------------

"""
연관 기억 시스템 데모 및 테스트
"""

# ----------------- 사용 예시 -----------------

def demonstrate_associative_memory():
    """연관 기억 시스템 데모"""
    
    # 향상된 연관 메모리 시스템 초기화
    memory_system = EnhancedAssociativeMemoryStore()
    
    print("=== 연관 기억 시스템 데모 시작 ===\n")
    
    # 1. 초기 메모리 입력 및 연관 관계 구축
    print("[1단계] 초기 메모리 및 연관 관계 구축")
    
    memories = [
        {
            'user': '우리 강아지 미키는 토이푸들이야',
            'assistant': '토이푸들 미키를 기억할게요.',
            'triggers': ['미키', '강아지', '토이푸들'],
            'importance': 0.8,
            'emotional_weight': 0.5,
            'memory_type': 'semantic'
        },
        {
            'user': '미키 생일은 3월 12일이야',
            'assistant': '미키의 생일은 3월 12일이네요.',
            'triggers': ['미키', '생일', '3월 12일'],
            'importance': 0.9,
            'emotional_weight': 0.6,
            'memory_type': 'semantic'
        },
        {
            'user': '미키는 노란 공을 좋아해',
            'assistant': '미키가 노란 공을 좋아하는구나.',
            'triggers': ['미키', '노란 공', '좋아해'],
            'importance': 0.7,
            'emotional_weight': 0.3,
            'memory_type': 'episodic'
        },
        {
            'user': '미키가 저번에 생일파티에서 케이크를 먹고 배탈났었어',
            'assistant': '안타깝네요. 미키가 케이크를 먹고 배탈났었군요.',
            'triggers': ['미키', '생일파티', '케이크', '배탈'],
            'importance': 0.85,
            'memory_type': 'semantic'
        },
        {
            'user': '미키 생일은 3월 12일이야',
            'assistant': '미키의 생일은 3월 12일이네요.',
            'triggers': ['미키', '생일', '3월 12일'],
            'importance': 0.9,
            'emotional_weight': 0.6,
            'memory_type': 'semantic'
        },
        {
            'user': '미키는 노란 공을 좋아해',
            'assistant': '미키가 노란 공을 좋아하는구나.',
            'triggers': ['미키', '노란 공', '좋아해'],
            'importance': 0.7,
            'emotional_weight': 0.3,
            'memory_type': 'episodic'
        },
        {
            'user': '미키가 저번에 생일파티에서 케이크를 먹고 배탈났었어',
            'assistant': '안타깝네요. 미키가 케이크를 먹고 배탈났었군요.',
            'triggers': ['미키', '생일파티', '케이크', '배탈'],
            'importance': 0.85,
            'emotional_weight': 0.7,
            'memory_type': 'emotional'
        }
    ]
    
    # 메모리 추가 및 연관 관계 구축
    for mem in memories:
        memory_id = memory_system.add_memory_with_associations(mem)
        print(f"메모리 추가: {mem['user']}")
        
        # 개념 활성화
        for trigger in mem['triggers']:
            memory_system.association_network.activate_concept(trigger, mem['triggers'])
    
    print("\n[2단계] 연관 네트워크 상태 확인")
    
    # 네트워크 통계 출력
    stats = memory_system.association_network.get_network_stats()
    print(f"개념 수: {stats['total_concepts']}")
    print(f"연결 수: {stats['total_connections']}")
    print(f"평균 연결 수: {stats['average_connections']:.2f}")
    
    # 가장 활성화된 개념들
    print("\n가장 활성화된 개념들:")
    for concept, count in stats['most_activated']:
        print(f"  - {concept}: {count}회")
    
    # 가장 강한 연결들
    print("\n가장 강한 연결들:")
    for from_c, to_c, strength in stats['strongest_connections']:
        print(f"  - {from_c} → {to_c}: {strength:.3f}")
    
    print("\n[3단계] 연관 기반 기억 회상 테스트")
    
    # 다양한 쿼리로 연관 회상 테스트
    queries = [
        "미키",
        "생일",
        "케이크",
        "노란 공"
    ]
    
    for query in queries:
        print(f"\n쿼리: '{query}'")
        print("-" * 50)
        
        # 연관 회상 실행
        recall_result = memory_system.recall_associations(query)
        
        # 직접 관련된 메모리들
        print(f"직접 관련 메모리: {len(recall_result['direct_memories'])}개")
        if recall_result['direct_memories']:
            for i, mem in enumerate(recall_result['direct_memories'][:2]):
                print(f"  {i+1}. {mem['memory']['user']}")
        
        # 간접 관련 메모리들
        print(f"간접 관련 메모리: {len(recall_result['indirect_memories'])}개")
        if recall_result['indirect_memories']:
            for i, mem in enumerate(recall_result['indirect_memories'][:2]):
                concept = mem['associated_concept']
                strength = mem['association_strength']
                print(f"  {i+1}. '{mem['memory']['user']}' (via {concept}, strength: {strength:.3f})")
        
        # 관련 개념들
        print("관련 개념들:")
        for concept, strength in list(recall_result['related_concepts'].items())[:5]:
            print(f"  - {concept}: {strength:.3f}")
        
        # 연관 경로
        if recall_result['association_paths']:
            print("연관 경로:")
            for target, path in recall_result['association_paths'][:2]:
                path_str = " → ".join([f"{step[0]}({step[1]:.2f})" for step in path])
                print(f"  {target}: {path_str}")
    
    print("\n[4단계] 시간에 따른 망각 시뮬레이션")
    
    # 망각 전 연결 상태
    print("망각 전:")
    weak_connections = []
    for u, v, data in memory_system.association_network.graph.edges(data=True):
        if data['weight'] < 0.5:
            weak_connections.append((u, v, data['weight']))
    
    if weak_connections:
        print(f"약한 연결 {len(weak_connections)}개 존재")
        for u, v, weight in weak_connections[:3]:
            print(f"  - {u} → {v}: {weight:.3f}")
    
    # 망각 적용
    memory_system.association_network.apply_decay(timedelta(days=30))
    
    # 망각 후 연결 상태
    print("\n망각 후:")
    remaining_weak = []
    for u, v, data in memory_system.association_network.graph.edges(data=True):
        if data['weight'] < 0.5:
            remaining_weak.append((u, v, data['weight']))
    
    print(f"남은 약한 연결: {len(remaining_weak)}개")
    
    print("\n[5단계] 연관 네트워크 시각화")
    
    # 시각화 저장 경로 설정
    import os
    import time
    timestamp = int(time.time())
    viz_dir = "association_vizualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # 주요 개념들에 대한 연관 네트워크 시각화
    key_concepts = ["미키", "생일", "케이크"]
    
    for concept in key_concepts:
        save_path = os.path.join(viz_dir, f"associations_{concept}_{timestamp}.png")
        print(f"'{concept}' 중심 연관 네트워크 시각화: {save_path}")
        memory_system.visualize_association_web(concept, save_path)
    
    print("\n[6단계] 연관 기억 시스템 리포트")
    
    report = memory_system.get_association_report()
    
    print("\n네트워크 통계 요약:")
    for key, value in report['network_statistics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        elif isinstance(value, list):
            print(f"  {key}: {len(value)}개")
        else:
            print(f"  {key}: {value}")
    
    print("\n개념 클러스터 분석:")
    for cluster_name, analysis in report['cluster_analysis'].items():
        print(f"  {cluster_name}:")
        print(f"    크기: {analysis['size']}")
        print(f"    평균 활성화: {analysis['average_activation']:.2f}")
        print(f"    멤버: {', '.join(analysis['members'][:5])}")
    
    print("\n=== 연관 기억 시스템 데모 완료 ===")

# ----------------- 통합된 챗봇 시스템 -----------------

class AssociativeMemoryChatbot:
    """연관 기억 기능이 통합된 챗봇"""
    
    def __init__(self, collection_name="associative_chatbot", persist_dir="./chroma_db"):
        """초기화"""
        print("=== 연관 기억 챗봇 시스템 초기화 ===")
        
        # 향상된 연관 메모리 시스템 사용
        self.memory_system = EnhancedAssociativeMemoryStore(collection_name, persist_dir)
        
        # 기본 LangGraph 워크플로우는 유지하되, 연관 기억 기능 추가
        self.graph = create_memory_chatbot_graph_with_associations()
        
        print("연관 기억 챗봇 시스템 초기화 완료")
    
    def process_message(self, user_input: str) -> str:
        """사용자 입력 처리 및 응답 생성"""
        print(f"\n[사용자] {user_input}")
        
        # 초기 상태 구성
        state = {
            "user_input": user_input,
            "memory_store": self.memory_system,
            "association_network": self.memory_system.association_network
        }
        
        # 그래프 실행
        try:
            result = self.graph.invoke(state)
            return result["assistant_reply"]
        except Exception as e:
            print(f"오류 발생: {e}")
            return "죄송합니다, 처리 중 오류가 발생했습니다."
    
    def get_association_visualization(self, concept: str, save_path: str = None) -> None:
        """특정 개념의 연관 네트워크 시각화"""
        self.memory_system.visualize_association_web(concept, save_path)

# ----------------- LangGraph 노드 함수 개선 -----------------

def search_associative_memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """연관 기억 기반 메모리 검색 노드"""
    user_input = state["user_input"]
    triggers = state["triggers"]
    
    # 연관 메모리 시스템에서 검색
    memory_store = state["memory_store"]
    
    # 각 트리거에 대해 연관 회상 실행
    all_recall_results = []
    
    for trigger in triggers:
        recall_result = memory_store.recall_associations(trigger)
        all_recall_results.append(recall_result)
        
        # 개념 강화
        if recall_result['direct_memories']:
            memory_store.strengthen_association_by_recall(trigger, 
                                                        recall_result['direct_memories'][0]['memory'])
    
    # 검색 결과 통합
    direct_memories = []
    indirect_memories = []
    
    for result in all_recall_results:
        direct_memories.extend(result['direct_memories'])
        indirect_memories.extend(result['indirect_memories'])
    
    # 중복 제거 및 정렬
    direct_memories = list({m['memory']['id']: m for m in direct_memories}.values())
    indirect_memories = list({m['memory']['id']: m for m in indirect_memories}.values())
    
    # 점수 기반 정렬
    direct_memories.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)
    indirect_memories.sort(key=lambda x: x.get('association_strength', 0), reverse=True)
    
    return {
        **state,
        "retrieved_memories": direct_memories,
        "associated_memories": indirect_memories,
        "association_context": {
            'related_concepts': {t: r['related_concepts'] for t, r in zip(triggers, all_recall_results)},
            'association_paths': [r['association_paths'] for r in all_recall_results]
        }
    }

def generate_associative_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """연관 기억을 활용한 응답 생성 노드"""
    user_input = state["user_input"]
    memory_results = state["retrieved_memories"]
    associated_memories = state.get("associated_memories", [])
    association_context = state.get("association_context", {})
    
    try:
        # 기억 내용을 텍스트로 요약
        if memory_results or associated_memories:
            # 연관 기억을 기반으로 응답 생성
            response = generate_associative_response(user_input, memory_results, 
                                                    associated_memories, association_context)
        else:
            # 연관 기억이 없을 때 기본 응답
            response = generate_default_response(user_input)
    except Exception as e:
        print(f"응답 생성 오류: {e}")
        response = "죄송합니다, 처리 중 오류가 발생했습니다."
    
    print(f"[응답] {response}")
    
    return {
        **state,
        "assistant_reply": response
    }

def generate_associative_response(user_input: str, direct_memories: List[Dict], 
                                indirect_memories: List[Dict], association_context: Dict) -> str:
    """연관 기억을 활용한 응답 생성"""
    
    # 질문 유형 분석
    if "누구" in user_input or "무엇" in user_input:
        # 정보 요청 응답
        if direct_memories:
            top_memory = direct_memories[0]["memory"]
            return f"기억하고 있어요! {top_memory['assistant']}"
        elif indirect_memories:
            indirect_mem = indirect_memories[0]
            related_concept = indirect_mem['associated_concept']
            return f"{related_concept}와(과) 관련해서 기억나는데, {indirect_mem['memory']['assistant']}"
    
    elif "언제" in user_input:
        # 시간 관련 응답
        for mem in direct_memories:
            if any(keyword in mem['memory']['user'] for keyword in ['월', '일', '년']):
                return f"날짜를 기억하고 있어요. {mem['memory']['assistant']}"
    
    elif "왜" in user_input:
        # 이유나 원인 응답 (연관 경로 활용)
        if association_context.get('association_paths'):
            for paths in association_context['association_paths']:
                if paths:
                    path_concept = paths[0][0]
                    return f"{path_concept}와(과) 관련이 있는 것 같아요."
    
    # 기본 응답 생성
    if direct_memories:
        top_memory = direct_memories[0]["memory"]
        return f"그것에 대해 알고 있어요. {top_memory['assistant']}"
    elif indirect_memories:
        return f"비슷한 내용으로 {indirect_memories[0]['memory']['user']}가 생각나네요."
    else:
        return "관련된 기억이 없네요. 더 자세히 말씀해주실래요?"

def create_memory_chatbot_graph_with_associations():
    """연관 기억이 포함된 LangGraph 워크플로우 생성"""
    
    # LangGraph 버전 체크 및 그래프 초기화
    try:
        import importlib.metadata
        version = importlib.metadata.version("langgraph")
        from packaging import version as pkg_version
        if pkg_version.parse(version) >= pkg_version.parse("0.1.0"):
            workflow = StateGraph(ChatbotState)
        else:
            workflow = StateGraph(state_schema=ChatbotState)
    except:
        try:
            workflow = StateGraph(ChatbotState)
        except:
            workflow = StateGraph(state_schema=ChatbotState)
    
    # 노드 추가
    workflow.add_node("extract_triggers", extract_triggers_node)
    workflow.add_node("search_associative_memory", search_associative_memory_node)  # 수정됨
    workflow.add_node("generate_associative_response", generate_associative_response_node)  # 수정됨
    workflow.add_node("store_memory", store_memory_node)
    
    # 엣지 설정
    workflow.set_entry_point("extract_triggers")
    workflow.add_edge("extract_triggers", "search_associative_memory")
    workflow.add_edge("search_associative_memory", "generate_associative_response")
    workflow.add_edge("generate_associative_response", "store_memory")
    workflow.add_edge("store_memory", END)
    
    # 그래프 컴파일
    memory_graph = workflow.compile()
    
    return memory_graph

# ----------------- 실행 예시 -----------------

def main():
    """연관 기억 챗봇 실행 예시"""
    
    # 데모 실행
    print("=== 연관 기억 시스템 데모 ===")
    demonstrate_associative_memory()
    
    print("\n\n=== 연관 기억 챗봇 대화 테스트 ===")
    
    # 챗봇 초기화
    chatbot = AssociativeMemoryChatbot()
    
    # 대화 테스트
    test_queries = [
        "강아지에 대해 뭘 알고 있어?",
        "생일이 언제야?",
        "케이크 먹고 어떻게 됐어?",
        "미키가 좋아하는 것은?",
        "생일에 뭐하면 좋을까?"
    ]
    
    for query in test_queries:
        print("\n" + "="*50)
        print(f"사용자: {query}")
        response = chatbot.process_message(query)
        print(f"챗봇: {response}")
    
    # 대화 종료 후 연관 네트워크 시각화
    print("\n연관 네트워크 시각화 생성...")
    viz_path = f"final_association_network.png"
    chatbot.get_association_visualization("미키", viz_path)
    print(f"연관 네트워크 저장: {viz_path}")

if __name__ == "__main__":
    main()          'emotional_weight': 0.7,
            'memory_type': 'emotional'
        }
    ]
    
    # 메모리 추가 및 연관 관계 구축
    for mem in memories:
        memory_id = memory_system.add_memory_with_associations(mem)
        print(f"메모리 추가: {mem['user']}")
        
        # 개념 활성화
        for trigger in mem['triggers']:
            memory_system.association_network.activate_concept(trigger, mem['triggers'])
    
    print("\n[2단계] 연관 네트워크 상태 확인")
    
    # 네트워크 통계 출력
    stats = memory_system.association_network.get_network_stats()
    print(f"개념 수: {stats['total_concepts']}")
    print(f"연결 수: {stats['total_connections']}")
    print(f"평균 연결 수: {stats['average_connections']:.2f}")
    
    # 가장 활성화된 개념들
    print("\n가장 활성화된 개념들:")
    for concept, count in stats['most_activated']:
        print(f"  - {concept}: {count}회")
    
    # 가장 강한 연결들
    print("\n가장 강한 연결들:")
    for from_c, to_c, strength in stats['strongest_connections']:
        print(f"  - {from_c} → {to_c}: {strength:.3f}")
    
    print("\n[3단계] 연관 기반 기억 회상 테스트")
    
    # 다양한 쿼리로 연관 회상 테스트
    queries = [
        "미키",
        "생일",
        "케이크",
        "노란 공"
    ]
    
    for query in queries:
        print(f"\n쿼리: '{query}'")
        print("-" * 50)
        
        # 연관 회상 실행
        recall_result = memory_system.recall_associations(query)
        
        # 직접 관련된 메모리들
        print(f"직접 관련 메모리: {len(recall_result['direct_memories'])}개")
        if recall_result['direct_memories']:
            for i, mem in enumerate(recall_result['direct_memories'][:2]):
                print(f"  {i+1}. {mem['memory']['user']}")
        
        # 간접 관련 메모리들
        print(f"간접 관련 메모리: {len(recall_result['indirect_memories'])}개")
        if recall_result['indirect_memories']:
            for i, mem in enumerate(recall_result['indirect_memories'][:2]):
                concept = mem['associated_concept']
                strength = mem['association_strength']
                print(f"  {i+1}. '{mem['memory']['user']}' (via {concept}, strength: {strength:.3f})")
        
        # 관련 개념들
        print("관련 개념들:")
        for concept, strength in list(recall_result['related_concepts'].items())[:5]:
            print(f"  - {concept}: {strength:.3f}")
        
        # 연관 경로
        if recall_result['association_paths']:
            print("연관 경로:")
            for target, path in recall_result['association_paths'][:2]:
                path_str = " → ".join([f"{step[0]}({step[1]:.2f})" for step in path])
                print(f"  {target}: {path_str}")
    
    print("\n[4단계] 시간에 따른 망각 시뮬레이션")
    
    # 망각 전 연결 상태
    print("망각 전:")
    weak_connections = []
    for u, v, data in memory_system.association_network.graph.edges(data=True):
        if data['weight'] < 0.5:
            weak_connections.append((u, v, data['weight']))
    
    if weak_connections:
        print(f"약한 연결 {len(weak_connections)}개 존재")
        for u, v, weight in weak_connections[:3]:
            print(f"  - {u} → {v}: {weight:.3f}")
    
    # 망각 적용
    memory_system.association_network.apply_decay(timedelta(days=30))
    
    # 망각 후 연결 상태
    print("\n망각 후:")
    remaining_weak = []
    for u, v, data in memory_system.association_network.graph.edges(data=True):
        if data['weight'] < 0.5:
            remaining_weak.append((u, v, data['weight']))
    
    print(f"남은 약한 연결: {len(remaining_weak)}개")
    
    print("\n[5단계] 연관 네트워크 시각화")
    
    # 시각화 저장 경로 설정
    import os
    import time
    timestamp = int(time.time())
    viz_dir = "association_vizualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # 주요 개념들에 대한 연관 네트워크 시각화
    key_concepts = ["미키", "생일", "케이크"]
    
    for concept in key_concepts:
        save_path = os.path.join(viz_dir, f"associations_{concept}_{timestamp}.png")
        print(f"'{concept}' 중심 연관 네트워크 시각화: {save_path}")
        memory_system.visualize_association_web(concept, save_path)
    
    print("\n[6단계] 연관 기억 시스템 리포트")
    
    report = memory_system.get_association_report()
    
    print("\n네트워크 통계 요약:")
    for key, value in report['network_statistics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        elif isinstance(value, list):
            print(f"  {key}: {len(value)}개")
        else:
            print(f"  {key}: {value}")
    
    print("\n개념 클러스터 분석:")
    for cluster_name, analysis in report['cluster_analysis'].items():
        print(f"  {cluster_name}:")
        print(f"    크기: {analysis['size']}")
        print(f"    평균 활성화: {analysis['average_activation']:.2f}")
        print(f"    멤버: {', '.join(analysis['members'][:5])}")
    
    print("\n=== 연관 기억 시스템 데모 완료 ===")

# ----------------- 통합된 챗봇 시스템 -----------------

class AssociativeMemoryChatbot:
    """연관 기억 기능이 통합된 챗봇"""
    
    def __init__(self, collection_name="associative_chatbot", persist_dir="./chroma_db"):
        """초기화"""
        print("=== 연관 기억 챗봇 시스템 초기화 ===")
        
        # 향상된 연관 메모리 시스템 사용
        self.memory_system = EnhancedAssociativeMemoryStore(collection_name, persist_dir)
        
        # 기본 LangGraph 워크플로우는 유지하되, 연관 기억 기능 추가
        self.graph = create_memory_chatbot_graph_with_associations()
        
        print("연관 기억 챗봇 시스템 초기화 완료")
    
    def process_message(self, user_input: str) -> str:
        """사용자 입력 처리 및 응답 생성"""
        print(f"\n[사용자] {user_input}")
        
        # 초기 상태 구성
        state = {
            "user_input": user_input,
            "memory_store": self.memory_system,
            "association_network": self.memory_system.association_network
        }
        
        # 그래프 실행
        try:
            result = self.graph.invoke(state)
            return result["assistant_reply"]
        except Exception as e:
            print(f"오류 발생: {e}")
            return "죄송합니다, 처리 중 오류가 발생했습니다."
    
    def get_association_visualization(self, concept: str, save_path: str = None) -> None:
        """특정 개념의 연관 네트워크 시각화"""
        self.memory_system.visualize_association_web(concept, save_path)

# ----------------- LangGraph 노드 함수 개선 -----------------

def search_associative_memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """연관 기억 기반 메모리 검색 노드"""
    user_input = state["user_input"]
    triggers = state["triggers"]
    
    # 연관 메모리 시스템에서 검색
    memory_store = state["memory_store"]
    
    # 각 트리거에 대해 연관 회상 실행
    all_recall_results = []
    
    for trigger in triggers:
        recall_result = memory_store.recall_associations(trigger)
        all_recall_results.append(recall_result)
        
        # 개념 강화
        if recall_result['direct_memories']:
            memory_store.strengthen_association_by_recall(trigger, 
                                                        recall_result['direct_memories'][0]['memory'])
    
    # 검색 결과 통합
    direct_memories = []
    indirect_memories = []
    
    for result in all_recall_results:
        direct_memories.extend(result['direct_memories'])
        indirect_memories.extend(result['indirect_memories'])
    
    # 중복 제거 및 정렬
    direct_memories = list({m['memory']['id']: m for m in direct_memories}.values())
    indirect_memories = list({m['memory']['id']: m for m in indirect_memories}.values())
    
    # 점수 기반 정렬
    direct_memories.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)
    indirect_memories.sort(key=lambda x: x.get('association_strength', 0), reverse=True)
    
    return {
        **state,
        "retrieved_memories": direct_memories,
        "associated_memories": indirect_memories,
        "association_context": {
            'related_concepts': {t: r['related_concepts'] for t, r in zip(triggers, all_recall_results)},
            'association_paths': [r['association_paths'] for r in all_recall_results]
        }
    }

def generate_associative_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """연관 기억을 활용한 응답 생성 노드"""
    user_input = state["user_input"]
    memory_results = state["retrieved_memories"]
    associated_memories = state.get("associated_memories", [])
    association_context = state.get("association_context", {})
    
    try:
        # 기억 내용을 텍스트로 요약
        if memory_results or associated_memories:
            # 연관 기억을 기반으로 응답 생성
            response = generate_associative_response(user_input, memory_results, 
                                                    associated_memories, association_context)
        else:
            # 연관 기억이 없을 때 기본 응답
            response = generate_default_response(user_input)
    except Exception as e:
        print(f"응답 생성 오류: {e}")
        response = "죄송합니다, 처리 중 오류가 발생했습니다."
    
    print(f"[응답] {response}")
    
    return {
        **state,
        "assistant_reply": response
    }

def generate_associative_response(user_input: str, direct_memories: List[Dict], 
                                indirect_memories: List[Dict], association_context: Dict) -> str:
    """연관 기억을 활용한 응답 생성"""
    
    # 질문 유형 분석
    if "누구" in user_input or "무엇" in user_input:
        # 정보 요청 응답
        if direct_memories:
            top_memory = direct_memories[0]["memory"]
            return f"기억하고 있어요! {top_memory['assistant']}"
        elif indirect_memories:
            indirect_mem = indirect_memories[0]
            related_concept = indirect_mem['associated_concept']
            return f"{related_concept}와(과) 관련해서 기억나는데, {indirect_mem['memory']['assistant']}"
    
    elif "언제" in user_input:
        # 시간 관련 응답
        for mem in direct_memories:
            if any(keyword in mem['memory']['user'] for keyword in ['월', '일', '년']):
                return f"날짜를 기억하고 있어요. {mem['memory']['assistant']}"
    
    elif "왜" in user_input:
        # 이유나 원인 응답 (연관 경로 활용)
        if association_context.get('association_paths'):
            for paths in association_context['association_paths']:
                if paths:
                    path_concept = paths[0][0]
                    return f"{path_concept}와(과) 관련이 있는 것 같아요."
    
    # 기본 응답 생성
    if direct_memories:
        top_memory = direct_memories[0]["memory"]
        return f"그것에 대해 알고 있어요. {top_memory['assistant']}"
    elif indirect_memories:
        return f"비슷한 내용으로 {indirect_memories[0]['memory']['user']}가 생각나네요."
    else:
        return "관련된 기억이 없네요. 더 자세히 말씀해주실래요?"

def create_memory_chatbot_graph_with_associations():
    """연관 기억이 포함된 LangGraph 워크플로우 생성"""
    
    # LangGraph 버전 체크 및 그래프 초기화
    try:
        import importlib.metadata
        version = importlib.metadata.version("langgraph")
        from packaging import version as pkg_version
        if pkg_version.parse(version) >= pkg_version.parse("0.1.0"):
            workflow = StateGraph(ChatbotState)
        else:
            workflow = StateGraph(state_schema=ChatbotState)
    except:
        try:
            workflow = StateGraph(ChatbotState)
        except:
            workflow = StateGraph(state_schema=ChatbotState)
    
    # 노드 추가
    workflow.add_node("extract_triggers", extract_triggers_node)
    workflow.add_node("search_associative_memory", search_associative_memory_node)  # 수정됨
    workflow.add_node("generate_associative_response", generate_associative_response_node)  # 수정됨
    workflow.add_node("store_memory", store_memory_node)
    
    # 엣지 설정
    workflow.set_entry_point("extract_triggers")
    workflow.add_edge("extract_triggers", "search_associative_memory")
    workflow.add_edge("search_associative_memory", "generate_associative_response")
    workflow.add_edge("generate_associative_response", "store_memory")
    workflow.add_edge("store_memory", END)
    
    # 그래프 컴파일
    memory_graph = workflow.compile()
    
    return memory_graph

# ----------------- 실행 예시 -----------------

def main():
    """연관 기억 챗봇 실행 예시"""
    
    # 데모 실행
    print("=== 연관 기억 시스템 데모 ===")
    demonstrate_associative_memory()
    
    print("\n\n=== 연관 기억 챗봇 대화 테스트 ===")
    
    # 챗봇 초기화
    chatbot = AssociativeMemoryChatbot()
    
    # 대화 테스트
    test_queries = [
        "강아지에 대해 뭘 알고 있어?",
        "생일이 언제야?",
        "케이크 먹고 어떻게 됐어?",
        "미키가 좋아하는 것은?",
        "생일에 뭐하면 좋을까?"
    ]
    
    for query in test_queries:
        print("\n" + "="*50)
        print(f"사용자: {query}")
        response = chatbot.process_message(query)
        print(f"챗봇: {response}")
    
    # 대화 종료 후 연관 네트워크 시각화
    print("\n연관 네트워크 시각화 생성...")
    viz_path = f"final_association_network.png"
    chatbot.get_association_visualization("미키", viz_path)
    print(f"연관 네트워크 저장: {viz_path}")

if __name__ == "__main__":
    main()